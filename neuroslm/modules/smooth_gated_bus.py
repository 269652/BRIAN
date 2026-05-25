"""Smooth-Gated Bus — temporally-smooth replacement for ReZero step gates.

ReZero (§5.3) uses zero-init scalar parameters `lambda_motor/mem/thought` that
grow under LM gradient. Problems observed up to 10k steps:

  1. The gates spend the first ~2-4k steps near zero — the model effectively
     runs as "trunk only" during that window, which exactly matches when
     LM loss decreases fastest. So the model learns to fit next-token loss
     using the trunk alone, never developing strong module dependence.
     By the time the gates open, the trunk has already minimized its
     loss without help — there's no gradient pressure to grow the gates
     further, so they stay small.

  2. The transition from "fully closed" (λ≈0) to "open" (λ>0.1) is
     abrupt in parameter space — small Δλ from gradient can flip
     module contribution from 0× to 5-10× its raw magnitude. This is
     the loss-spike pathology in §5.1.

  3. A checkpoint saved mid-transition (e.g. ReZero's mix_best.pt at
     step 7000) captures the model at a transient point where the
     trunk had been trained alongside small-but-growing λ — and the
     legacy-default-fallback ([[ood-eval-bug-rootcause-and-fix]]) is
     only a workaround, not a structural fix.

The smooth-gated-bus replaces the raw scalar with a **temporally smooth
sigmoid ramp** parameterized by three learnable scalars per gate:

    gate(t) = max_strength · sigmoid((t − t_center) / t_width)

  • `max_strength` (init 1.0) — asymptotic gate value as t → ∞.
  • `t_center`     (init = `default_center`, e.g. 2000) — step at which
                   gate reaches 50% of max_strength.
  • `t_width`      (init = `default_width`,  e.g. 500) — transition
                   sharpness; smaller = sharper ramp.

`t` is the current training step, threaded into the model as a buffer
via `set_training_step(step)`. At eval time `t = float('inf')` so
gates are fully open.

Why this addresses the three problems:

  1. Module contribution is **non-zero from step 0** (`gate(0) ≈
     sigmoid(−4) ≈ 0.018` with default center=2000, width=500), so
     even the earliest gradient updates have a small but real signal
     pushing on module weights. The trunk no longer has the option
     of solving the LM loss in isolation.

  2. The transition is **continuous in t**, not in λ. Δgate per step
     is bounded by `max_strength / (4 · t_width)` (the max slope of
     sigmoid). The model can never see a 0× → 5× contribution jump
     because the time variable is monotonic.

  3. Gates have a **learnable temporal schedule** instead of a single
     scalar that drifts under gradient. A checkpoint saved at step
     N captures both the model weights AND the gate's current
     temporal position, which is deterministic from t alone. No
     "transient point" pathology.

References
----------
LayerScale (Touvron 2021) — predecessor of ReZero with a smooth-init
    scalar; SGB extends to a time-varying schedule.
SLOG (Smith et al. 2023, ICLR) — temporal scheduling of regularization
    schedules; we adopt the per-gate learnable schedule idea.
Free-energy minimization (Friston) — composability argument: in a PCT
    trunk, module injections must reduce prediction error to be useful.
    A smooth schedule lets the trunk discover this gradually without
    sudden capacity injections.
"""
from __future__ import annotations
import torch
import torch.nn as nn


class SmoothTemporalGate(nn.Module):
    """A scalar gate whose value smoothly rises from ~0 to max_strength
    as the training step variable t advances through a learnable
    sigmoid schedule.

    Parameters
    ----------
    default_center : initial value of t_center (step at which gate is 50%).
    default_width  : initial value of t_width (transition sharpness).
    min_width      : softplus lower bound on t_width to prevent collapse.
    init_max       : initial value of max_strength.

    The default schedule (`center=2000, width=500`) gives:
        gate(   0) ≈ 0.018  (small but non-zero)
        gate(1000) ≈ 0.12
        gate(2000) = 0.50
        gate(3000) ≈ 0.88
        gate(5000) ≈ 0.997
    """

    def __init__(self, default_center: float = 2000.0,
                 default_width: float = 500.0,
                 min_width: float = 50.0,
                 init_max: float = 1.0):
        super().__init__()
        self.max_strength = nn.Parameter(torch.tensor(float(init_max)))
        # Store t_center directly; learnable but unconstrained (negative
        # is fine — means gate opens "in the past", i.e. fully open already).
        self.t_center = nn.Parameter(torch.tensor(float(default_center)))
        # Parameterize width through a softplus-shifted raw scalar so it
        # stays > min_width. Initialize raw so softplus(raw) + min_width
        # equals default_width.
        # softplus(x) = ln(1 + e^x); we want ln(1 + e^raw) = default_width - min_width
        # → raw = ln(e^(default_width - min_width) − 1)
        target = float(default_width - min_width)
        if target > 50:  # exp overflow guard for large widths
            raw_init = target  # softplus(x) ≈ x for large x
        else:
            raw_init = float(torch.log(torch.exp(torch.tensor(target)) - 1.0))
        self._t_width_raw = nn.Parameter(torch.tensor(raw_init))
        self._min_width = float(min_width)

    @property
    def t_width(self) -> torch.Tensor:
        return nn.functional.softplus(self._t_width_raw) + self._min_width

    def forward(self, step: float | torch.Tensor) -> torch.Tensor:
        """Return the current gate value as a 1-D tensor of shape [1].

        `step` may be a Python float, a 0-D tensor, or a scalar tensor.
        At inference, pass `float('inf')` to get the fully-open value.
        """
        if not torch.is_tensor(step):
            step_t = torch.as_tensor(step, dtype=self.t_center.dtype,
                                     device=self.t_center.device)
        else:
            step_t = step.to(device=self.t_center.device,
                             dtype=self.t_center.dtype)

        # sigmoid argument; clamp to avoid +/-inf surprises (esp at eval)
        if torch.isinf(step_t):
            # At eval, return max_strength directly (gate fully open).
            return self.max_strength.view(1)

        arg = (step_t - self.t_center) / self.t_width
        # Cheap clamp to keep sigmoid in float range
        arg = torch.clamp(arg, min=-30.0, max=30.0)
        return (self.max_strength * torch.sigmoid(arg)).view(1)

    def extra_repr(self) -> str:
        with torch.no_grad():
            return (f"max={self.max_strength.item():.3f}, "
                    f"center={self.t_center.item():.1f}, "
                    f"width={self.t_width.item():.1f}")


class SmoothGatedBus(nn.Module):
    """Container for the three module-injection gates (motor / mem /
    thought) used by Brain. Acts as a drop-in replacement for the
    `lambda_motor / lambda_mem / lambda_thought` raw scalars under
    `cfg.use_smooth_gated_bus`.

    Usage in Brain:
        if cfg.use_smooth_gated_bus:
            self.sgb = SmoothGatedBus(...)
        else:
            self.lambda_motor = nn.Parameter(zeros(1))   # legacy
            ...

        # In forward, after a training-step setter has been called:
        if self.sgb is not None:
            lam_m = self.sgb.gate('motor')
            lam_mem = self.sgb.gate('mem')
            lam_th = self.sgb.gate('thought')
        else:
            lam_m, lam_mem, lam_th = self.lambda_motor, self.lambda_mem, self.lambda_thought
    """

    GATE_NAMES = ("motor", "mem", "thought")

    def __init__(self, default_center: float = 2000.0,
                 default_width: float = 500.0):
        super().__init__()
        self.gates = nn.ModuleDict({
            name: SmoothTemporalGate(
                default_center=default_center,
                default_width=default_width,
            )
            for name in self.GATE_NAMES
        })
        # current training step, updated by Brain.set_training_step
        # buffer so it's saved/loaded with state_dict for reproducibility
        self.register_buffer(
            "_current_step",
            torch.tensor(0.0, dtype=torch.float32),
        )

    def set_step(self, step: int | float) -> None:
        """Update the internal step counter. Called by Brain.set_training_step
        once per train step. At eval, set to float('inf') for fully-open gates."""
        if torch.is_tensor(step):
            self._current_step = step.detach().to(self._current_step.dtype).reshape(())
        else:
            self._current_step = torch.tensor(float(step),
                                              dtype=self._current_step.dtype,
                                              device=self._current_step.device)

    def gate(self, name: str) -> torch.Tensor:
        """Return current value of the named gate as a [1]-shaped tensor."""
        if name not in self.gates:
            raise KeyError(f"unknown gate {name!r}; have {list(self.gates)}")
        return self.gates[name](self._current_step)

    def all_values(self) -> dict[str, float]:
        """Detached scalar dict for logging."""
        with torch.no_grad():
            return {n: float(self.gate(n).item()) for n in self.GATE_NAMES}
