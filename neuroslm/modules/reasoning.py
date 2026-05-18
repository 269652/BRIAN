"""ReasoningCortex — Modern Hopfield Network for analogical pattern completion.

Biologically: lateral prefrontal cortex (lPFC) + frontoparietal network
implement relational reasoning, analogy, and rule extraction.  The key
computational property is pattern completion: given a partial or noisy
pattern, the network recovers the closest stored attractor.

Computationally: uses the Modern Hopfield Network (Ramsauer 2020) with a
learned attractor library. A high β (inverse temperature) produces a
sharp, winner-take-all retrieval — critical for logical deduction where
ambiguity must be resolved.

Reference:
  Ramsauer et al. (2020). Hopfield networks is all you need.
  ICLR 2021.  arXiv:2008.02217

Activated by κ_reason vesicles (type REASONING in VesiclePool).
"""
from __future__ import annotations
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from .brain_module import BrainModule
from .common import TransformerBlock, RMSNorm
from .fast_weight import FastWeightLayer


# Topic index for this expert (must match TopicClassifier.N_TOPICS ordering)
TOPIC_REASONING = 2


class ReasoningCortex(BrainModule):
    """Pattern-completion reasoning via Modern Hopfield attractors.

    Architecture:
      1. A learnable attractor bank (n_attractors × d_sem)
         — the "knowledge library" of reasoning patterns
      2. Hopfield update:  x_new = β × A^T × softmax(β × A × x^T)
         where A = attractor matrix.  Multiple iterations converge
         x towards the nearest attractor (energy minimum).
      3. Lateral inhibition between attractors (competitive dynamics)
         so each update selects a distinct reasoning schema.

    A higher β (sharper softmax) → more decisive pattern completion.
    β is learned via a scalar log_beta parameter (soft-plus so β > 0.5).

    Args:
        d_sem:        Semantic embedding dimension
        n_attractors: Number of learnable reasoning patterns
        base_beta:    Minimum β value (adds to soft-plus output)
        n_iters:      Hopfield update iterations (unrolled, XLA-safe)
    """

    def __init__(self, d_sem: int,
                 n_attractors: int = 64,
                 base_beta: float = 4.0,
                 n_iters: int = 3,
                 enable_hfw: bool = True,
                 n_action_types: int = 14,        # match SocialMarkovMemory N_TYPES
                 rec_rank: int = 16,
                 d_hidden: int | None = None,
                 n_blocks: int = 0,
                 max_ctx: int = 2048,
                 expert_n_heads: int = 4):
        """SRC-TEH extension args:

        d_hidden: per-token expert width.  When None (legacy/test path) only
            the d_sem Hopfield attractor path is active.  When set, the
            cortex additionally constructs `n_blocks` TransformerBlocks at
            d_hidden plus a learnable attractor-bank cross-attention layer —
            invoked via `forward_tokens(x)` from the expert-choice router.
        n_blocks: depth of the token-level expert (default 3 per SRC-TEH).
        max_ctx: max sequence length per token-level call (router caps it).
        expert_n_heads: heads used by the token-level transformer blocks.
        """
        super().__init__()
        self.d_sem = d_sem
        self.n_attractors = n_attractors
        self.base_beta = base_beta
        # Static: unroll exactly n_iters steps at compile time (XLA-safe)
        self.n_iters = min(n_iters, 4)
        self.d_hidden = d_hidden
        self.n_blocks = int(n_blocks)

        # Learnable attractor library (reasoning schema bank)
        self.attractors = nn.Parameter(
            torch.randn(n_attractors, d_sem) * (1.0 / math.sqrt(d_sem)))

        # Inverse temperature — starts moderate, grows via gradient descent
        self.log_beta = nn.Parameter(torch.zeros(1))

        # Output: per-attractor learned scale for the retrieved pattern
        self.output_scale = nn.Parameter(torch.ones(n_attractors))

        # QK-norm applied to query before Hopfield (numerics in bfloat16)
        self.query_norm = nn.LayerNorm(d_sem)

        # Projection: Hopfield output → d_sem residual
        self.out_proj = nn.Linear(d_sem, d_sem, bias=False)
        nn.init.zeros_(self.out_proj.weight)  # start as identity (no contribution)

        self.norm = nn.LayerNorm(d_sem)

        # ── Low-rank recurrent dynamics (causal attractor layer) ──────────
        # Implements:  h_{t+1} = tanh(A · B · h_t + W_in · x_t)
        # with A ∈ ℝ^{d × r}, B ∈ ℝ^{r × d}, r ≪ d. Two unrolled steps so
        # the model learns abstract relational schemas (e.g. "Insult → Offense")
        # as low-rank fixed points. Zero-init so the network starts as a
        # pure passthrough of the Hopfield retrieval.
        self.rec_rank = rec_rank
        self.rec_A = nn.Parameter(
            torch.randn(d_sem, rec_rank) * (1.0 / math.sqrt(d_sem)))
        self.rec_B = nn.Parameter(
            torch.zeros(rec_rank, d_sem))                # zero-init: identity
        self.rec_in = nn.Parameter(
            torch.eye(d_sem) * 0.0 + torch.randn(d_sem, d_sem) * 0.01)
        self.rec_norm = nn.LayerNorm(d_sem)

        # ── Action → Reaction predictor ──────────────────────────────────
        # Markov / Hopfield-style outcome distribution: given an action
        # embedding, predict a distribution over reaction-type prototypes.
        # The prototypes are learned simultaneously with the projection.
        # n_action_types matches SocialMarkovMemory's ACTION_LABELS length
        # so the two systems can hand off cleanly.
        self.n_action_types = n_action_types
        self.reaction_prototypes = nn.Parameter(
            torch.randn(n_action_types, d_sem) * (1.0 / math.sqrt(d_sem)))
        self.action_to_logits = nn.Sequential(
            nn.Linear(d_sem, d_sem),
            nn.GELU(),
            nn.Linear(d_sem, n_action_types),
        )

        # Hebbian fast weights — final binding stage.  When the cortex completes
        # a pattern, bind the answer with the current GWS broadcast so future
        # inference within the same conversation can recall it associatively
        # (dual-timescale: slow attractors × fast episodic).
        n_h = max(1, n_attractors // 16)
        while d_sem % n_h != 0 and n_h > 1:
            n_h -= 1
        self.hfw = FastWeightLayer(d_sem, decay=0.95, base_eta=0.1, n_heads=n_h) \
                   if enable_hfw else None
        self._hfw_state: torch.Tensor | None = None

        # ── Token-level expert stack (SRC-TEH) ────────────────────────────
        # Constructed only when d_hidden + n_blocks > 0.  Otherwise the
        # cortex is the lightweight (legacy) Hopfield enrichment.
        if self.d_hidden is not None and self.n_blocks > 0:
            eh = max(1, expert_n_heads)
            while self.d_hidden % eh != 0 and eh > 1:
                eh -= 1
            self.expert_blocks = nn.ModuleList([
                TransformerBlock(self.d_hidden, n_heads=eh, max_ctx=max_ctx)
                for _ in range(self.n_blocks)
            ])
            # Attractor bank at the expert width — Hopfield-style schema bank
            # for relational completion at token-level.
            self.attractors_h = nn.Parameter(
                torch.randn(n_attractors, self.d_hidden)
                * (1.0 / math.sqrt(self.d_hidden)))
            self.attr_log_beta = nn.Parameter(torch.zeros(1))
            self.attr_norm     = RMSNorm(self.d_hidden)
            self.attr_q        = nn.Linear(self.d_hidden, self.d_hidden, bias=False)
            self.attr_out      = nn.Linear(self.d_hidden, self.d_hidden, bias=False)
            nn.init.zeros_(self.attr_out.weight)
            self.expert_norm   = RMSNorm(self.d_hidden)
        else:
            self.expert_blocks = None

    # ------------------------------------------------------------------
    # Single Hopfield update step
    # ------------------------------------------------------------------
    def _hopfield_step(self, x: torch.Tensor) -> torch.Tensor:
        """One Hopfield energy-minimization step.

        x:  (B, d_sem)    — current query
        A:  (n_att, d_sem) — attractors (normalised)
        Returns: (B, d_sem) updated state
        """
        beta = F.softplus(self.log_beta) + self.base_beta  # β > base_beta

        A = F.normalize(self.attractors, dim=-1)           # (K, d)
        q = F.normalize(x.float(), dim=-1).to(x.dtype)    # (B, d) — QK-norm

        logits = beta * (q @ A.T)                          # (B, K)
        weights = F.softmax(logits, dim=-1)                # (B, K)

        # Scale attractors by learned per-attractor output weight
        scaled = A * self.output_scale.unsqueeze(-1).abs() # (K, d)
        retrieved = weights @ scaled                        # (B, d)
        return retrieved

    # ------------------------------------------------------------------
    # Lateral inhibition between attractors
    # ------------------------------------------------------------------
    def _lateral_inhibit(self, x: torch.Tensor) -> torch.Tensor:
        """Attenuate attractor dimensions that are redundant across the batch.

        Reduces "echo chamber" dynamics where all attractors collapse to one
        response. Analogous to cortical surround inhibition.
        """
        A = F.normalize(self.attractors, dim=-1)           # (K, d)
        sim = A @ A.T                                       # (K, K)
        eye = torch.eye(self.n_attractors, device=A.device, dtype=sim.dtype)
        off = (sim * (1 - eye)).clamp(min=0).mean(-1)      # (K,) — mean sim
        # Attenuate attractors that are highly similar to others
        gate = (1.0 - 0.2 * off).unsqueeze(-1)             # (K, 1)
        self.attractors.data = (
            self.attractors.data * gate.detach().to(self.attractors.dtype))
        return x

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------
    def forward(self, x: torch.Tensor,
                vesicle_gate: float = 0.0,
                gws_context: torch.Tensor | None = None,
                maturity: float | None = None) -> torch.Tensor:
        """x: (B, d_sem) or (B, S, d_sem).

        vesicle_gate ∈ [0, 1]: strength of reasoning-cortex activation.
        gws_context: optional (B, d_sem) — Global Workspace broadcast used as
                     plasticity context for the Hebbian fast-weight binding.
        maturity: optional MAT scalar in [0, 1] — convex fade-in weight applied
                  to the Hopfield-retrieved residual:
                      h_out = (1 - m_eff)·h_in + m_eff·Expert(h_in)
                  with m_eff = max(maturity, 0.05) so a 5% noise broadcast
                  flows through even at very low maturity. None preserves
                  legacy (full-weight) behaviour.

        Returns same shape as x.
        """
        if vesicle_gate < 1e-3:
            return x

        squeeze = x.dim() == 3
        if squeeze:
            x_flat = x.mean(1)  # (B, d_sem)
        else:
            x_flat = x          # (B, d_sem)

        h = self.norm(x_flat.float()).to(dtype=x_flat.dtype)
        h = self.query_norm(h.float()).to(dtype=h.dtype)

        # Unrolled Hopfield iterations (static depth → XLA-compilable)
        state = h
        if self.n_iters >= 1:
            state = self._hopfield_step(state)
        if self.n_iters >= 2:
            state = self._hopfield_step(state)
        if self.n_iters >= 3:
            state = self._hopfield_step(state)
        if self.n_iters >= 4:
            state = self._hopfield_step(state)

        # ── Low-rank recurrent dynamics on top of Hopfield retrieval ─────
        # state acts as the input drive; we unroll two recurrent steps.
        h = state
        for _ in range(2):
            rec_update = h @ self.rec_A @ self.rec_B + state @ self.rec_in
            h = torch.tanh(self.rec_norm(rec_update.float()).to(rec_update.dtype))
        state = h

        enrichment = self.out_proj(state)
        # Maturity fade-in: 5% noise floor pre-awakening, full weight at M ≈ 1.0.
        m_eff = 1.0 if maturity is None else max(float(maturity), 0.05)
        enriched   = x_flat + m_eff * vesicle_gate * enrichment

        # Hebbian fast-weight binding stage — bind GWS context ↔ retrieved schema
        if self.hfw is not None and vesicle_gate > 0.1:
            ctx = gws_context if gws_context is not None else enriched
            seq = enriched.unsqueeze(1)
            seq, self._hfw_state = self.hfw(seq, context=ctx,
                                             W_fast=self._hfw_state)
            enriched = seq.squeeze(1)

        if squeeze:
            delta = (enriched - x_flat).unsqueeze(1)  # (B, 1, d_sem)
            return x + delta
        return enriched

    # ------------------------------------------------------------------
    # Action → Reaction predictor (Modern Hopfield / Markov-style)
    # ------------------------------------------------------------------
    def predict_reaction(self,
                          action_emb: torch.Tensor,
                          temperature: float = 1.0,
                          ) -> tuple[torch.Tensor, torch.Tensor]:
        """Predict the distribution over reaction types given an action.

        action_emb: (B, d_sem) or (d_sem,).
        temperature: softmax temperature; lower → sharper prediction.

        Returns:
          probs:  (B, n_action_types) — categorical posterior over reactions.
          completed_pattern: (B, d_sem) — Modern-Hopfield pattern completion
                              onto the reaction-prototype bank.

        Two routes that complement each other:
          1. A direct MLP classifier from the action embedding to logits.
          2. A Modern-Hopfield completion over the prototype bank.
        We average both for robustness; the two routes co-train.
        """
        a = action_emb
        if a.dim() == 1:
            a = a.unsqueeze(0)

        # Route 1: direct classifier
        logits_mlp = self.action_to_logits(a) / max(1e-3, temperature)

        # Route 2: Hopfield completion over reaction prototypes
        beta = F.softplus(self.log_beta) + self.base_beta
        P = F.normalize(self.reaction_prototypes, dim=-1)         # (T, d)
        q = F.normalize(a.float(), dim=-1).to(a.dtype)            # (B, d)
        logits_hop = beta * (q @ P.T) / max(1e-3, temperature)    # (B, T)

        # Average the two route logits (geometric mean ≈ averaging logits)
        logits = 0.5 * (logits_mlp + logits_hop)
        probs = F.softmax(logits, dim=-1)

        # Completed pattern = expected prototype under the posterior
        completed = probs @ self.reaction_prototypes               # (B, d)
        return probs, completed

    def causal_aux_loss(self,
                         action_emb: torch.Tensor,
                         reaction_target: torch.Tensor,
                         ) -> torch.Tensor:
        """Cross-entropy auxiliary loss for the action → reaction predictor.

        action_emb:       (B, d_sem)
        reaction_target:  (B,) int64 — index into [0, n_action_types).

        train.py adds this loss scaled by `_aux_w_scale * w_causal`.
        Naturally suppressed during infancy.
        """
        probs, _ = self.predict_reaction(action_emb)
        return F.nll_loss(torch.log(probs + 1e-9), reaction_target.long())

    # ------------------------------------------------------------------
    # Token-level expert pass (SRC-TEH)
    # ------------------------------------------------------------------
    def forward_tokens(self, x: torch.Tensor,
                       maturity: float | None = None) -> torch.Tensor:
        """Process a routed batch of tokens through the 3-block expert.

        x: (B, C, d_hidden) — tokens pulled by the ExpertChoiceRouter.
        Returns (B, C, d_hidden) with attractor-completion residual applied.
        Falls back to identity when the token-level stack was not constructed.
        """
        if self.expert_blocks is None:
            return x

        B, C, D = x.shape
        h = x
        for blk in self.expert_blocks:
            h = blk(h)

        # Hopfield attractor cross-attention at d_hidden — each token
        # softly retrieves the best-matching reasoning schema.
        beta = F.softplus(self.attr_log_beta) + self.base_beta
        q   = self.attr_q(self.attr_norm(h.float()).to(h.dtype))      # (B, C, D)
        q   = F.normalize(q.float(), dim=-1).to(h.dtype)
        A   = F.normalize(self.attractors_h.float(), dim=-1).to(h.dtype)
        logits  = beta * torch.einsum("bcd,kd->bck", q, A)
        weights = F.softmax(logits.float(), dim=-1).to(h.dtype)
        retrieved = torch.einsum("bck,kd->bcd", weights, A)
        h = h + self.attr_out(retrieved)
        h = self.expert_norm(h.float()).to(h.dtype)

        m_eff = 1.0 if maturity is None else max(float(maturity), 0.05)
        return x + m_eff * (h - x)

    def _disabled_output(self, x: torch.Tensor, *args, **kwargs) -> torch.Tensor:
        return x
