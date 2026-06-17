"""GIF-7: Homeostatic Gradient Equilibrium.

Three synergistic mechanisms that stabilize optimization dynamics:

Part A — Divisive Gradient Normalization (Cortical Gain Control)
    Smooth replacement for hard gradient clipping.
    g'(t) = g(t) * c / sqrt(c² + ||g(t)||²)

Part B — Loss-Variance Metaplastic Damping (BCM Rule)
    Modulate effective LR by inverse loss variance.
    lr_eff = lr_sched * min(1, σ_ref / σ_L(t))

Part C — VBB KL Floor (Anti-Collapse Guard)
    Prevent posterior collapse with symmetric quadratic penalty.
    L_floor = γ * max(0, kl_min - KL(q||p))²

Neuroscience inspiration:
- Part A: V1 divisive normalization / cortical contrast gain control
- Part B: BCM sliding threshold for metaplasticity
- Part C: Homeostatic regulation of information flow (Turrigiano, 2012)
"""
from __future__ import annotations

import math
from collections import deque
from typing import Iterator, Optional, Tuple

import torch
import torch.nn as nn


# ═══════════════════════════════════════════════════════════════════
# Part A: Divisive Gradient Normalization
# ═══════════════════════════════════════════════════════════════════

def divisive_grad_normalize(
    parameters: Iterator[nn.Parameter],
    c: float,
) -> Tuple[float, float]:
    """Apply divisive normalization to gradients (cortical gain control).

    Instead of hard clipping at threshold c, applies the smooth function:
        scale = c / sqrt(c² + ||g||²)

    This is the exact functional form of V1 contrast normalization.

    Args:
        parameters: Model parameters (with .grad set).
        c: Semi-saturation constant. At ||g|| = c, signal is halved.
            Larger c → less aggressive normalization.

    Returns:
        (gnorm, scale): Raw gradient norm before normalization, and the
        scale factor applied.
    """
    params = [p for p in parameters if p.grad is not None]
    if not params:
        return 0.0, 1.0

    # Compute total gradient norm
    total_norm_sq = sum(
        p.grad.data.float().pow(2).sum().item() for p in params
    )
    gnorm = math.sqrt(total_norm_sq)

    if gnorm == 0.0:
        return 0.0, 1.0

    # Divisive normalization: scale = c / sqrt(c² + ||g||²)
    scale = c / math.sqrt(c * c + total_norm_sq)

    # Apply scaling to all gradients
    for p in params:
        p.grad.data.mul_(scale)

    return gnorm, scale


# ═══════════════════════════════════════════════════════════════════
# Part B: Loss-Variance Metaplastic Damping (BCM Rule)
# ═══════════════════════════════════════════════════════════════════

class LossVarianceDamper:
    """Modulate LR by inverse loss variance — BCM sliding threshold.

    When loss is oscillating wildly (high σ), lr_multiplier drops.
    When stable (σ ≈ σ_ref), lr_multiplier ≈ 1.

    Formula:
        mult = min(1, σ_ref / σ_L(t))

    Bounded below by min_mult to prevent complete stalling.
    """

    def __init__(
        self,
        window: int = 64,
        min_mult: float = 0.1,
        calibrate_at: Optional[int] = None,
    ):
        self.window = window
        self.min_mult = min_mult
        self.calibrate_at = calibrate_at
        self._buffer: deque = deque(maxlen=window)
        self._step = 0
        self.sigma_ref: float = 0.0
        self._calibrated = False

    def update(self, loss: float) -> None:
        """Record a loss value."""
        self._buffer.append(loss)
        self._step += 1
        # Auto-calibrate at specified step
        if (self.calibrate_at is not None
                and self._step == self.calibrate_at
                and not self._calibrated):
            self.calibrate()

    def calibrate(self) -> None:
        """Capture current loss std as the healthy reference σ_ref."""
        if len(self._buffer) < 2:
            return
        values = list(self._buffer)
        mean = sum(values) / len(values)
        var = sum((v - mean) ** 2 for v in values) / len(values)
        self.sigma_ref = math.sqrt(var) if var > 0 else 1e-8
        self._calibrated = True

    def _current_sigma(self) -> float:
        """Compute current loss standard deviation."""
        if len(self._buffer) < 2:
            return 0.0
        values = list(self._buffer)
        mean = sum(values) / len(values)
        var = sum((v - mean) ** 2 for v in values) / len(values)
        return math.sqrt(var) if var > 0 else 0.0

    def lr_multiplier(self) -> float:
        """Compute LR scaling factor.

        Returns 1.0 when:
          - Not enough data yet (< window samples)
          - Not calibrated yet (σ_ref = 0)
          - Loss is stable (σ ≤ σ_ref)

        Returns < 1.0 when loss variance exceeds reference.
        """
        if len(self._buffer) < self.window:
            return 1.0
        if self.sigma_ref <= 0:
            return 1.0

        sigma = self._current_sigma()
        if sigma <= self.sigma_ref:
            return 1.0

        # BCM rule: damp proportional to excess variance
        mult = self.sigma_ref / sigma
        return max(self.min_mult, mult)


# ═══════════════════════════════════════════════════════════════════
# Part C: VBB KL Floor (Anti-Collapse Guard)
# ═══════════════════════════════════════════════════════════════════

def vbb_kl_floor_loss(
    kl: torch.Tensor,
    kl_min: float = 100.0,
    gamma: float = 0.01,
) -> torch.Tensor:
    """Quadratic penalty when VBB KL drops below minimum.

    Prevents posterior collapse by penalizing low information flow.

    L_floor = γ * max(0, kl_min - kl)²

    When KL ≥ kl_min: penalty = 0 (healthy bottleneck)
    When KL < kl_min: quadratic push back up

    Args:
        kl: Current KL divergence (scalar tensor).
        kl_min: Minimum acceptable KL value.
        gamma: Penalty coefficient. 0 = disabled.

    Returns:
        Scalar loss term (add to total loss).
    """
    if gamma == 0.0:
        return torch.zeros(1, device=kl.device, dtype=kl.dtype).squeeze()

    deficit = torch.clamp(kl_min - kl, min=0.0)
    return gamma * deficit * deficit
