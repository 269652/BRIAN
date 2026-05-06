"""Global Workspace — Hopfield dynamics with ignition phase transition.

Theory: Baars/Dehaene Global Workspace + Modern Hopfield Networks (Ramsauer 2020).

The key insight: the attention mechanism IS the Hopfield update rule.
  slot^{t+1} = candidates^T softmax(β × candidates × slot^t^T)

Iterating this to convergence = Hopfield energy minimization.
The network finds the attractor closest to the current query.

Ignition (Dehaene 2011): conscious access occurs when GWS activity exceeds
a critical threshold θ, triggering a phase transition from local processing
to global broadcast. Pre-ignition: sparse, local activations. Post-ignition:
dense, widespread broadcast.

Lateral competition: slots inhibit each other proportional to cosine similarity,
ensuring each slot captures a distinct pattern (winner-take-all in feature space).
"""
from __future__ import annotations
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class GlobalWorkspace(nn.Module):
    def __init__(self, d_sem: int, n_slots: int, n_heads: int = 4,
                 gradient_checkpointing: bool = False,
                 hopfield_iters: int = 2,
                 ignition_threshold: float = 0.5):
        super().__init__()
        self.n_slots = n_slots
        self.d_sem   = d_sem
        self.gradient_checkpointing = gradient_checkpointing
        self.hopfield_iters = hopfield_iters

        self.slot_queries = nn.Parameter(torch.randn(n_slots, d_sem) * 0.02)
        # Hopfield inverse temperature β (learned, soft-plus to keep positive)
        self.log_beta = nn.Parameter(torch.zeros(1))
        # Standard MHA kept for backward compat when hopfield_iters == 0
        self.attn = nn.MultiheadAttention(d_sem, num_heads=n_heads, batch_first=True)
        self.norm = nn.LayerNorm(d_sem)

        # Ignition: sigmoid gate on mean slot activity vs threshold
        self.ignition_threshold = ignition_threshold
        # Learned per-slot output scale (starts at 1.0)
        self.output_scale = nn.Parameter(torch.ones(n_slots))

        # Last ignition probability (detached scalar, for logging / metrics)
        self._last_ignition: torch.Tensor | None = None

    # ------------------------------------------------------------------
    # Hopfield update step
    # ------------------------------------------------------------------
    def _hopfield_update(self, slots: torch.Tensor,
                         candidates: torch.Tensor) -> torch.Tensor:
        """One Hopfield update: slots ← softmax(β · C · S^T) · C

        slots:      (B, n_slots, d)
        candidates: (B, K, d)
        Returns:    (B, n_slots, d)
        """
        beta = F.softplus(self.log_beta) + 0.5   # β > 0.5
        # Energy-minimizing attention (unnormalised inner product)
        logits = torch.bmm(slots, candidates.transpose(1, 2)) * beta  # (B, n_slots, K)
        weights = F.softmax(logits, dim=-1)       # (B, n_slots, K)
        return torch.bmm(weights, candidates)     # (B, n_slots, d)

    # ------------------------------------------------------------------
    # Internal forward (called directly or via checkpoint wrapper)
    # ------------------------------------------------------------------
    def _forward(self, candidates: torch.Tensor,
                 ne_temp: torch.Tensor | None = None) -> torch.Tensor:
        B = candidates.size(0)

        # Initialise slots from learned queries
        slots = self.slot_queries.unsqueeze(0).expand(B, -1, -1)

        # Optional NE temperature scaling of initial queries
        if ne_temp is not None:
            slots = slots * ne_temp.view(B, 1, 1)

        if self.hopfield_iters > 0:
            # Iterative Hopfield convergence — fully unrolled so XLA compiles
            # this as a static graph (no Python-level loop variable at trace time).
            # hopfield_iters is fixed at construction; we unroll up to 4 steps.
            # XLA would otherwise retrace on each forward call if the loop bound
            # were a runtime tensor rather than a Python integer.
            if self.hopfield_iters >= 1:
                slots = self._hopfield_update(slots, candidates)
            if self.hopfield_iters >= 2:
                slots = self._hopfield_update(slots, candidates)
            if self.hopfield_iters >= 3:
                slots = self._hopfield_update(slots, candidates)
            if self.hopfield_iters >= 4:
                slots = self._hopfield_update(slots, candidates)

            # Lateral competition: inhibit slots that are too similar
            # cos-sim off-diagonal → suppress redundant patterns
            s_norm = F.normalize(slots, dim=-1)           # (B, n_slots, d)
            sim = torch.bmm(s_norm, s_norm.transpose(1, 2))  # (B, n_slots, n_slots)
            eye = torch.eye(self.n_slots, device=slots.device).unsqueeze(0)
            off_diag_sim = (sim * (1.0 - eye)).clamp(min=0)   # (B, n_slots, n_slots)
            mean_sim = off_diag_sim.sum(-1, keepdim=True) / max(self.n_slots - 1, 1)
            slots = slots * (1.0 - 0.15 * mean_sim)      # attenuate similar slots

            # Ignition phase transition
            # activity = mean L2 norm of slots across the slot dimension
            activity = slots.norm(dim=-1).mean(-1)        # (B,)
            # Soft sigmoid gate: pre-ignition ≈ 0.3, post-ignition ≈ 1.0
            ign_prob = torch.sigmoid(
                (activity - self.ignition_threshold) * 4.0)  # (B,)
            self._last_ignition = ign_prob.detach()
            broadcast_scale = 0.3 + 0.7 * ign_prob.view(B, 1, 1)
            slots = slots * broadcast_scale

            # Per-slot learned scale
            slots = slots * self.output_scale.unsqueeze(0).unsqueeze(-1)

        else:
            # Legacy: standard MHA (hopfield_iters == 0 disables Hopfield)
            q = self.slot_queries.unsqueeze(0).expand(B, -1, -1)
            if ne_temp is not None:
                q = q * ne_temp.view(B, 1, 1)
            slots, _ = self.attn(q, candidates, candidates, need_weights=False)

        return self.norm(slots)

    # ------------------------------------------------------------------
    # Public forward
    # ------------------------------------------------------------------
    def forward(self, candidates: torch.Tensor,
                ne_temp: torch.Tensor | None = None) -> torch.Tensor:
        """candidates: (B, K, d_sem) — embeddings competing for slot occupancy.
        Returns slots: (B, n_slots, d_sem)."""
        if self.gradient_checkpointing and self.training:
            return torch.utils.checkpoint.checkpoint(
                self._forward, candidates, ne_temp, use_reentrant=False)
        return self._forward(candidates, ne_temp)
