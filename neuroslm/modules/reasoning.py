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
                 n_iters: int = 3):
        super().__init__()
        self.d_sem = d_sem
        self.n_attractors = n_attractors
        self.base_beta = base_beta
        # Static: unroll exactly n_iters steps at compile time (XLA-safe)
        self.n_iters = min(n_iters, 4)

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
                vesicle_gate: float = 0.0) -> torch.Tensor:
        """x: (B, d_sem) or (B, S, d_sem).
        vesicle_gate ∈ [0, 1]: strength of reasoning-cortex activation.
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

        enrichment = self.out_proj(state)
        enriched   = x_flat + vesicle_gate * enrichment

        if squeeze:
            delta = (enriched - x_flat).unsqueeze(1)  # (B, 1, d_sem)
            return x + delta
        return enriched

    def _disabled_output(self, x: torch.Tensor, *args, **kwargs) -> torch.Tensor:
        return x
