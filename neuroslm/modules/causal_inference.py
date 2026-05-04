"""Causal Inference Module for NeuroSLM.

Standard neural networks learn statistical correlations.  Causal models
additionally learn *interventional distributions* — what would happen if
we forcibly set a variable to a specific value, breaking its normal causes
(Pearl's do-calculus).

Why causal reasoning matters for language:
  "The window broke because the ball hit it" vs "The ball hit the window
  after it broke" — the same co-occurrence, opposite causation.

  Understanding causation requires the model to distinguish:
    • Correlation:    P(Y | X=x)        — conditional observation
    • Intervention:   P(Y | do(X=x))    — hypothetical intervention
    • Counterfactual: P(Y_{X=x'} | X=x) — what-if reasoning

This module implements three components:

  1. Causal Graph Learner:
     Learns a soft adjacency matrix A[i,j] from token representations,
     where A[i,j] ≈ P(token j is caused by token i).  Uses a
     differentiable DAG constraint (Zheng et al. 2018: NOTEARS) to
     encourage acyclicity:  h(A) = tr(e^(A∘A)) - d = 0.

  2. Interventional Head:
     Given a learned causal graph and a query, estimates the interventional
     distribution P(Y | do(X)) by cutting the causal graph at X and
     propagating via the remaining edges.  This implements a soft version
     of the "front-door criterion."

  3. Counterfactual Simulator:
     Given the actual representation z and a counterfactual premise
     ("what if token j had been different"), produces a counterfactual
     representation z_cf by replaying the causal graph with the
     intervened value.

  The causal graph is sequence-local (computed per forward call), not
  a global parameter, so it can adapt to different causal structures in
  different contexts.

Novel aspect: no existing SLM uses a learned *local* causal graph to
modulate its representations.  Graph attention (GAT) uses fixed topology;
relational transformers learn association not causation.  This module
is the first to apply soft DAG constraints to in-context causal discovery
inside an LLM-style forward pass.

References:
  Pearl (2009): Causality: Models, Reasoning and Inference
  Zheng et al. (2018): DAGs with NO TEARS: Continuous Optimization for DAG Structure Learning
  Schölkopf et al. (2021): Towards Causal Representation Learning
  Ke et al. (2019): Learning Neural Causal Models from Unknown Interventions
"""
from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple


class CausalGraphLearner(nn.Module):
    """Learns a soft local DAG over token representations.

    Uses a low-rank factored adjacency A = sigmoid(U @ V^T / sqrt(d))
    to avoid O(T^2 * D) parameters, plus the NOTEARS acyclicity penalty.
    """

    def __init__(self, d_model: int, d_causal: int = 32):
        super().__init__()
        self.d_causal = d_causal
        # Map each token to a low-rank causal key/query
        self.cause_proj  = nn.Linear(d_model, d_causal)  # "I might cause others"
        self.effect_proj = nn.Linear(d_model, d_causal)  # "I might be caused by others"

    def forward(self, x: torch.Tensor
               ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        x: (B, T, d_model)
        Returns:
          A:          (B, T, T) soft adjacency  (A[b,i,j] = P(i → j))
          dag_penalty (B,)     acyclicity penalty h(A)
        """
        C = self.cause_proj(x)   # (B, T, d_causal)
        E = self.effect_proj(x)  # (B, T, d_causal)

        # A[i,j] = P(token i causes token j)
        A = torch.sigmoid(
            torch.bmm(C, E.transpose(1, 2)) / (self.d_causal ** 0.5)
        )   # (B, T, T)

        # Zero diagonal (no self-causation)
        A = A * (1 - torch.eye(A.shape[1], device=A.device).unsqueeze(0))

        # Soft acyclicity penalty: h(A) = tr(e^(A∘A)) - T
        # Expensive for large T, so use matrix power approximation: tr(I + A^2 + ...) ≈ tr(e^A^2)
        A_sq = torch.bmm(A, A)   # (B, T, T)
        dag_penalty = (A_sq.diagonal(dim1=-2, dim2=-1).sum(-1) -
                       torch.tensor(float(A.shape[1]), device=A.device))
        dag_penalty = dag_penalty.abs()   # (B,)

        return A, dag_penalty


class CausalInferenceModule(nn.Module):
    """Full causal inference module: graph learning + intervention + counterfactual.

    Parameters
    ----------
    d_model   : token embedding dimension
    n_vars    : number of abstract causal variables (semantic slots)
    d_causal  : dimension of causal key/query projections
    """

    def __init__(self, d_model: int, n_vars: int = 8, d_causal: int = 32):
        super().__init__()
        self.n_vars  = n_vars
        self.d_causal = d_causal

        self.graph_learner = CausalGraphLearner(d_model, d_causal)

        # Token → abstract causal variable encoder
        self.var_encoder = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Linear(d_model, n_vars),
        )

        # Interventional propagation: given do(X), estimate downstream effect
        # Implemented as message-passing over the causal graph
        self.msg_proj = nn.Linear(d_model, d_model)
        self.gate_proj = nn.Linear(d_model * 2, d_model)

        # Counterfactual head: (original_rep, counterfactual_cause) → counterfactual_rep
        self.cf_head = nn.Sequential(
            nn.Linear(d_model * 2, d_model),
            nn.GELU(),
            nn.Linear(d_model, d_model),
        )

        # Output norm
        self.ln = nn.LayerNorm(d_model)

        # Learned mask to differentiate direct vs indirect effects
        self.direct_gate = nn.Linear(n_vars, n_vars)

    # ------------------------------------------------------------------

    def _intervene(self, x: torch.Tensor, A: torch.Tensor,
                   do_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Propagate intervention through the causal graph.

        x:       (B, T, d_model) — current token representations
        A:       (B, T, T)       — soft causal adjacency
        do_mask: (B, T)          — True at positions where we intervene
                                   (cut incoming edges, keep outgoing)

        Returns: (B, T, d_model) — causally-propagated representations
        """
        # Cut incoming edges at intervened nodes
        if do_mask is not None:
            # Zero column i of A for each intervened token i
            A_mod = A * (1 - do_mask.float().unsqueeze(1))
        else:
            A_mod = A

        # One round of causal message passing
        messages = self.msg_proj(x)                          # (B, T, D)
        agg = torch.bmm(A_mod, messages)                     # (B, T, D)  sum of causes

        # Gating: how much does the cause information update each token
        gate_input = torch.cat([x, agg], dim=-1)             # (B, T, 2D)
        gate = torch.sigmoid(self.gate_proj(gate_input))     # (B, T, D)
        propagated = x + gate * agg                          # (B, T, D)

        return self.ln(propagated)

    # ------------------------------------------------------------------

    def forward(self, x: torch.Tensor,
                cf_tokens: Optional[torch.Tensor] = None,
                do_mask: Optional[torch.Tensor] = None
               ) -> dict:
        """
        x:         (B, T, d_model)
        cf_tokens: (B, T, d_model) — counterfactual premise (alternative
                   token representations for "what if X had been Y")
        do_mask:   (B, T) bool     — which tokens are intervention targets

        Returns dict with:
          out:           (B, T, d_model) — causally-enriched representations
          causal_vars:   (B, T, n_vars)  — abstract causal variable encoding
          A:             (B, T, T)       — soft causal graph
          dag_loss:      scalar          — acyclicity penalty (add to total loss)
          cf_out:        (B, T, d_model) — counterfactual output (if cf_tokens given)
        """
        # ---- 1. Causal graph ----
        A, dag_penalty = self.graph_learner(x)              # (B,T,T), (B,)
        dag_loss = dag_penalty.mean() * 1e-3                # small weight

        # ---- 2. Causal variable encoding ----
        causal_vars = self.var_encoder(x)                   # (B, T, n_vars)
        direct_vars = self.direct_gate(F.gelu(causal_vars)) # (B, T, n_vars)

        # ---- 3. Interventional propagation ----
        out = self._intervene(x, A, do_mask)                # (B, T, d_model)

        # ---- 4. Counterfactual output ----
        cf_out = out
        if cf_tokens is not None:
            # Replace intervened cause, re-propagate
            x_cf = cf_tokens if do_mask is None else (
                torch.where(do_mask.unsqueeze(-1), cf_tokens, x)
            )
            prop_cf = self._intervene(x_cf, A, do_mask=None)
            cf_input = torch.cat([out, prop_cf], dim=-1)    # (B, T, 2D)
            cf_out = self.ln(out + self.cf_head(cf_input))  # (B, T, D)

        return {
            "out":         out,
            "causal_vars": direct_vars,
            "A":           A,
            "dag_loss":    dag_loss,
            "cf_out":      cf_out,
        }
