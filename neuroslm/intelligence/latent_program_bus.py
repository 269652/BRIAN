"""Latent Program Bus — continuous chain-of-thought channel.

The trunk emits a compact `bus_dim`-dim state at the end of each forward
pass; experts read/write to it; the trunk reads it back at the *start* of
the next forward pass.  This implements iterative reasoning without rerunning
the trunk and replaces the vesicle-based routing role of the old expert
gating (vesicles keep their plasticity/aversiveness roles — see RFC §F).

Properties:
  • Stored as a buffer (per-batch, EMA-smoothed across steps) so it survives
    a single training step's autograd graph without bridging gradients
    across steps.
  • Initialised to zero — at step 0 the trunk reads back a zero bus, so the
    refactor is gradient-equivalent to the legacy path on the very first
    step (smooth checkpoint round-trip).
  • bf16-safe: stored at fp32 internally; cast at read-back.
"""
from __future__ import annotations
import torch
import torch.nn as nn


class LatentProgramBus(nn.Module):
    """Across-step latent thought channel.

    Args:
        d_hidden:  Trunk hidden dimension (the trunk emits / reads back at d_hidden).
        bus_dim:   Compact bus width (typically 16-32).
        ema_alpha: EMA mixing for cross-step continuity (0.5 = 2-step half-life).
    """

    def __init__(self, d_hidden: int, bus_dim: int = 16,
                 ema_alpha: float = 0.5, d_sem: int | None = None):
        super().__init__()
        self.d_hidden  = d_hidden
        self.bus_dim   = bus_dim
        self.d_sem     = d_sem
        self.ema_alpha = float(ema_alpha)
        # Trunk → bus emission head (small).
        self.write_head = nn.Sequential(
            nn.Linear(d_hidden, d_hidden // 2),
            nn.GELU(),
            nn.Linear(d_hidden // 2, bus_dim),
        )
        # Bus → trunk readback (zero-init last layer so step 0 is identity).
        self.read_head = nn.Sequential(
            nn.Linear(bus_dim, d_hidden // 2),
            nn.GELU(),
            nn.Linear(d_hidden // 2, d_hidden),
        )
        nn.init.zeros_(self.read_head[-1].weight)
        nn.init.zeros_(self.read_head[-1].bias)
        # Optional bus → d_sem projection (zero-init) for code paths that need
        # to inject the bias on the d_sem `thought` channel rather than at
        # d_hidden directly.  Brain uses this to avoid changing the trunk
        # forward signature again.
        if d_sem is not None:
            self.bus_to_sem = nn.Linear(bus_dim, d_sem, bias=False)
            nn.init.zeros_(self.bus_to_sem.weight)
        else:
            self.bus_to_sem = None
        # Persistent bus state (1 vector per program; broadcast across batch).
        # Stored fp32; cast at read.
        self.register_buffer("_bus_state",
                             torch.zeros(bus_dim, dtype=torch.float32))
        self.register_buffer("_n_writes",
                             torch.zeros(1, dtype=torch.long))

    @torch.no_grad()
    def reset(self):
        """Clear bus state (call between sequences / at episode boundaries)."""
        self._bus_state.zero_()
        self._n_writes.zero_()

    def read(self, B: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        """Read back the current bus → (B, d_hidden) bias to add to trunk input."""
        # Clone so the in-place EMA update in `write()` cannot version-bump
        # a tensor still referenced by the backward graph.
        bus = self._bus_state.detach().clone().to(device=device, dtype=dtype)
        out = self.read_head(bus)                                # (d_hidden,)
        return out.unsqueeze(0).expand(B, -1)

    def read_sem(self, B: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        """Read the current bus → (B, d_sem) bias for the `thought` channel.

        Requires d_sem to have been provided at construction time.  Returns
        an empty (B, 0) tensor if no d_sem projection was registered.
        """
        if self.bus_to_sem is None:
            return torch.zeros(B, 0, device=device, dtype=dtype)
        bus = self._bus_state.detach().clone().to(device=device, dtype=dtype)
        out = self.bus_to_sem(bus)                                # (d_sem,)
        return out.unsqueeze(0).expand(B, -1)

    def write(self, trunk_summary: torch.Tensor) -> torch.Tensor:
        """Push a fresh bus state from the trunk's pooled summary.

        trunk_summary: (B, d_hidden) — typically the AttentionPool of h.
        Returns the (B, bus_dim) freshly-emitted bus token for inspection.
        Updates `_bus_state` via EMA so cross-step continuity is preserved.
        """
        bus_new = self.write_head(trunk_summary)              # (B, bus_dim)
        with torch.no_grad():
            mean_bus = bus_new.detach().mean(0).to(self._bus_state.dtype)
            a = self.ema_alpha
            self._bus_state.mul_(1.0 - a).add_(mean_bus, alpha=a)
            self._n_writes += 1
        return bus_new


__all__ = ["LatentProgramBus"]
