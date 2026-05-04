"""BrainModule: self-contained base class for every brain area.

Each area can be toggled on/off at any time:
    brain.hippo.disable()   # turn off hippocampus
    brain.pfc.enable()      # turn it back on

When disabled, `forward_safe()` returns the neutral passthrough defined by
the subclass via `_disabled_output()`.  The module's parameters still exist
and are still updated by the optimizer (unless you also freeze them), so
enabling it again restores full behaviour.

Usage pattern in subclasses:
    class Hippocampus(BrainModule):
        def forward(self, ...):
            ...                           # real algorithm
        def _disabled_output(self, query, *args, **kwargs):
            B = query.size(0)
            zeros = torch.zeros(B, self.topk, self.d_sem, device=query.device)
            ones  = torch.ones(B, device=query.device)
            return zeros, ones            # same shape as enabled forward
"""
from __future__ import annotations
import torch
import torch.nn as nn


class BrainModule(nn.Module):
    """Base class for all brain-area modules. Provides enable / disable."""

    def __init__(self):
        super().__init__()
        self.enabled: bool = True

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def enable(self) -> "BrainModule":
        self.enabled = True
        return self

    def disable(self) -> "BrainModule":
        self.enabled = False
        return self

    def toggle(self) -> "BrainModule":
        self.enabled = not self.enabled
        return self

    def forward_safe(self, *args, **kwargs):
        """Run forward() when enabled, else _disabled_output()."""
        if self.enabled:
            return self.forward(*args, **kwargs)
        return self._disabled_output(*args, **kwargs)

    def _disabled_output(self, *args, **kwargs):
        """Return neutral outputs matching forward()'s signature. Override in subclasses."""
        raise NotImplementedError(
            f"{type(self).__name__} must implement _disabled_output().")

    # ------------------------------------------------------------------
    # Convenience
    # ------------------------------------------------------------------
    def to_device(self, device) -> "BrainModule":
        return self.to(device)

    @property
    def status(self) -> str:
        return "enabled" if self.enabled else "disabled"
