"""Predictive coding residuals — propagate the prediction error.

Hierarchical predictive coding (Rao & Ballard 1999, Friston 2005)
inverts the standard "activation propagates forward" view: each layer
*predicts* the layer below; the only thing that actually moves up the
hierarchy is the **prediction residual** ``ε = x - x̂`` — what wasn't
already explained.

The canonical local update at layer ``ℓ``:

    x̂_ℓ      = g_ℓ(x_{ℓ+1})           # top-down prediction
    ε_ℓ      = x_ℓ - x̂_ℓ              # residual / "prediction error"
    x'_ℓ     = x_ℓ - α · ε_ℓ         # state update (gradient on ε² loss)

We expose this as an ``edge``-kind feature for the BRIAN DSL: the
forward consumes the lower-layer activation ``x_below`` and the
upper-layer state ``x_above`` and returns the new ``x_below`` plus
publishes the residual via the module's ``last_residual`` buffer so
downstream code (e.g. the loss head) can read it.

Two operating modes:

* ``mode="iterative"`` — one fixed-step update per forward call.
  Use inside a re-entry loop that calls the module repeatedly.
* ``mode="single"`` — collapses ``predict + residual + correct`` into
  one functional pass, returning the corrected activation directly.
  Use as a drop-in residual block in feed-forward circuits.

References
~~~~~~~~~~
* Rao, R.P.N., Ballard, D.H. — "Predictive coding in the visual cortex:
  a functional interpretation of some extra-classical receptive-field
  effects", *Nature Neurosci.* 2(1), 1999.
* Friston, K. — "A theory of cortical responses", *Phil. Trans. R. Soc.
  B* 360(1456), 2005.
* Whittington, J., Bogacz, R. — "An approximation of the error back-
  propagation algorithm in a predictive coding network with local
  Hebbian synaptic plasticity", *Neural Comp.* 29(5), 2017.

Implementation: tested by ``tests/test_predictive_coding_residual.py``
(13 contracts).
"""

from __future__ import annotations

from typing import Literal, Optional

import torch
import torch.nn as nn


_RESIDUAL_EPS = 1e-8


def predictive_residual(
    x_below: torch.Tensor,
    x_above: torch.Tensor,
    predictor: nn.Module,
    *,
    step_size: float = 0.1,
) -> tuple[torch.Tensor, torch.Tensor]:
    """One Rao-Ballard-style local update.

    Returns ``(x_below_new, residual)``. The residual is the part of
    ``x_below`` *not* explained by the top-down prediction; it's what
    propagates upward in a predictive-coding hierarchy.

    Args:
        x_below: lower-layer state ``(..., D)``.
        x_above: upper-layer state ``(..., D_above)`` — input to the
            generative ``predictor``.
        predictor: any ``nn.Module`` mapping ``x_above`` shape to
            ``x_below`` shape. Typically a single linear layer in the
            canonical Rao-Ballard formulation, but anything goes.
        step_size: ``α`` in the gradient step on ``ε² / 2``. Must be in
            ``(0, 1]``; smaller = more iterations to converge, larger
            = faster but risks oscillation. The classical PC literature
            uses ``α = 0.1``.

    Returns:
        ``(x_below_new, residual)`` where ``residual = x_below - x̂``.
        Both have the same shape as ``x_below``.
    """
    if not (0.0 < step_size <= 1.0):
        raise ValueError(
            f"step_size must be in (0, 1], got {step_size}"
        )
    x_hat = predictor(x_above)
    if x_hat.shape != x_below.shape:
        raise ValueError(
            f"predictor produced shape {tuple(x_hat.shape)} but "
            f"x_below has shape {tuple(x_below.shape)} — they must match"
        )
    residual = x_below - x_hat
    x_below_new = x_below - step_size * residual
    return x_below_new, residual


class PredictiveCodingResidual(nn.Module):
    """Predictive-coding edge: corrects ``x_below`` using ``x_above``.

    Wired as an ``edge`` endpoint in the BRIAN DSL: ``forward(x)`` takes
    a single sequence of activations ``(B, T, D)`` and treats ``x`` as
    *both* the lower-layer state and the source of the prediction
    (using a learned linear predictor). The corrected state is
    returned; the residual is exposed via ``self.last_residual`` so
    downstream losses can access it.

    This is the "edge-shaped" wrapping of the canonical PC update; for
    cross-layer use the functional :func:`predictive_residual` directly.

    Args:
        d_model: embedding dimension (D).
        predictor: optional custom predictor module. Default is a
            single ``nn.Linear(d_model, d_model, bias=True)`` — the
            Rao-Ballard linear generative model.
        step_size: gradient step on the ε² objective.
        mode: ``"single"`` for one-shot residual correction;
            ``"iterative"`` for one update per forward call (use
            inside a loop).
        n_iterations: number of inner steps when ``mode="iterative"``.
            Default ``1``.
    """

    def __init__(
        self,
        d_model: int,
        predictor: Optional[nn.Module] = None,
        *,
        step_size: float = 0.1,
        mode: Literal["single", "iterative"] = "single",
        n_iterations: int = 1,
    ) -> None:
        super().__init__()
        if d_model <= 0:
            raise ValueError(f"d_model must be positive, got {d_model}")
        if mode not in ("single", "iterative"):
            raise ValueError(
                f"mode must be 'single' or 'iterative', got {mode!r}"
            )
        if n_iterations < 1:
            raise ValueError(f"n_iterations must be ≥ 1, got {n_iterations}")
        self.d_model = d_model
        self.step_size = float(step_size)
        self.mode = mode
        self.n_iterations = int(n_iterations)
        self.predictor = predictor if predictor is not None else \
            nn.Linear(d_model, d_model, bias=True)
        # Initialise the DEFAULT predictor to ~identity-minus-noise so
        # the first forward isn't dominated by a random projection.
        # Custom predictors are left alone — caller controls init.
        if predictor is None and isinstance(self.predictor, nn.Linear):
            with torch.no_grad():
                nn.init.eye_(self.predictor.weight)
                self.predictor.weight.add_(
                    0.01 * torch.randn_like(self.predictor.weight)
                )
                if self.predictor.bias is not None:
                    nn.init.zeros_(self.predictor.bias)
        # Buffer for the most recent residual norm — useful for the
        # loss head and for diagnostics. Persistent=False so it doesn't
        # bloat checkpoints.
        self.register_buffer(
            "last_residual_norm",
            torch.zeros(1),
            persistent=False,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """``(B, T, D)`` → ``(B, T, D)`` — return corrected state.

        For the BRIAN edge contract we treat the SAME tensor as both
        ``x_below`` (state to be corrected) and ``x_above`` (source of
        the prediction). This implements "self-predicting" PC, which
        in the linear-predictor case learns to be the identity plus a
        residual sparsifier — a well-studied recurrent generative model.
        """
        if x.dim() != 3:
            raise ValueError(
                f"PredictiveCodingResidual expects (B, T, D), got "
                f"shape {tuple(x.shape)}"
            )
        x_state = x
        residual = torch.zeros_like(x_state)
        iters = self.n_iterations if self.mode == "iterative" else 1
        for _ in range(iters):
            x_state, residual = predictive_residual(
                x_state, x_state, self.predictor,
                step_size=self.step_size,
            )
        # Track the residual L2 norm for downstream consumers — detach
        # so it never gets backproped through.
        self.last_residual_norm.copy_(
            residual.detach().norm().reshape(1)
        )
        return x_state
