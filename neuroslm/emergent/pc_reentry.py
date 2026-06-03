# -*- coding: utf-8 -*-
"""C3 — Predictive-coding reentry residual (observation-only).

The `motor → sensory` synapse in `arch.neuro` realises the bowtie's
re-entry loop (the cosmetic biology already specifies it). We observe
whether this loop carries predictive structure: at each step, can the
*previous* motor activation linearly predict the *current* sensory
activation?

If yes, the reentry is functional — Karl-Friston-style free-energy
minimisation is happening for free. If no, the loop is a dead wire.

This probe owns a tiny diagonal+rank-1 linear predictor (no parameter
budget impact) that learns online via SGD on the residual ‖e_t‖². The
predictor is *isolated* from the trunk: gradients computed here do not
flow into the model's parameters. We only update our own `W` and `b`.

Telemetry: `pc_residual`, `pc_explained_var`, `pc_predictor_norm`.

When telemetry confirms the residual is informative, Phase 8 promotes
this into a real `λ_pc · ‖e_t‖²` aux loss with trunk gradients.
"""
from __future__ import annotations
from typing import Dict, Optional

import torch


class PCReentryProbe:
    """Online linear predictor for the motor→sensory loop.

    Maintains a learnable diagonal scale + rank-1 outer-product
    correction; trained by SGD on the residual norm. All computation
    runs on detached copies of the trunk activations so no gradient
    leaks into the main model.

    Parameters
    ----------
    dim : int
        Hidden dimensionality of the motor / sensory populations
        (assumed equal — they share `d_sem` in the RCC-Bowtie).
    lr : float
        SGD learning rate for the predictor.
    momentum : float
        Heavy-ball momentum for stability.
    ema : float
        EMA coefficient for the reported `pc_residual` / `pc_explained`.
    """

    def __init__(self,
                 dim: int,
                 lr: float = 1e-3,
                 momentum: float = 0.9,
                 ema: float = 0.05,
                 device: Optional[torch.device] = None):
        if dim <= 0:
            raise ValueError("dim must be positive")
        self.dim = int(dim)
        self.lr = float(lr)
        self.momentum = float(momentum)
        self.ema = float(ema)

        dev = device or torch.device("cpu")
        # Diagonal scale (init: identity) + rank-1 outer-product correction.
        self._diag = torch.ones(dim, device=dev)
        self._u = torch.zeros(dim, device=dev)
        self._v = torch.zeros(dim, device=dev)
        # Momentum buffers.
        self._m_diag = torch.zeros_like(self._diag)
        self._m_u = torch.zeros_like(self._u)
        self._m_v = torch.zeros_like(self._v)

        self._prev_motor: Optional[torch.Tensor] = None
        self._residual_ema: Optional[float] = None
        self._explained_ema: Optional[float] = None

    # ── Linear predictor ─────────────────────────────────────────────

    def predict(self, h_motor_prev: torch.Tensor) -> torch.Tensor:
        """ŝ_t = diag(d) · m_{t-1} + (m_{t-1} · v) · u

        The rank-1 correction lets the predictor capture a single
        dominant cross-channel direction beyond pure per-channel scale.
        """
        x = h_motor_prev.to(self._diag.device)
        diag_term = x * self._diag
        # (m · v) along last dim, broadcast onto u
        coeff = (x * self._v).sum(dim=-1, keepdim=True)
        rank1_term = coeff * self._u
        return diag_term + rank1_term

    # ── Step ─────────────────────────────────────────────────────────

    def step(self,
             h_motor: Optional[torch.Tensor],
             h_sensory: Optional[torch.Tensor]) -> Dict[str, float]:
        """Update the predictor and return current telemetry.

        Either tensor may be ``None`` (eval contexts where the relevant
        population is not exposed); the probe then returns stale stats.
        """
        if h_motor is None or h_sensory is None:
            return self._stats()

        # Detach: we never want a gradient leaking into the trunk.
        m = h_motor.detach().float()
        s = h_sensory.detach().float()
        if m.shape != s.shape or m.shape[-1] != self.dim:
            # Shape mismatch — silently no-op (probe is best-effort).
            return self._stats()

        if self._prev_motor is None:
            self._prev_motor = m
            return self._stats()

        # Predict s_t from m_{t-1}.
        x_prev = self._prev_motor
        pred = self.predict(x_prev)
        e = s - pred                                        # residual
        res_norm = float(e.pow(2).mean().item())
        target_var = float(s.var(unbiased=False).item()) + 1e-12
        explained = max(0.0, 1.0 - res_norm / target_var)

        # ── SGD on ‖e‖² over (diag, u, v) ───────────────────────────
        # d L/d diag = -2 · mean_{...} ( e ⊙ x_prev )
        # d L/d u    = -2 · mean_{...} ( e * (x_prev · v) )      (per dim)
        # d L/d v    = -2 · mean_{...} ( (e · u) * x_prev )
        x_flat = x_prev.reshape(-1, self.dim)
        e_flat = e.reshape(-1, self.dim)
        N = max(1, x_flat.shape[0])
        coeff_xv = (x_flat * self._v).sum(dim=-1, keepdim=True)  # (N,1)
        coeff_eu = (e_flat * self._u).sum(dim=-1, keepdim=True)  # (N,1)

        grad_diag = -2.0 * (e_flat * x_flat).sum(dim=0) / N
        grad_u    = -2.0 * (e_flat * coeff_xv).sum(dim=0) / N
        grad_v    = -2.0 * (x_flat * coeff_eu).sum(dim=0) / N

        # Heavy-ball update.
        self._m_diag = self.momentum * self._m_diag + grad_diag
        self._m_u    = self.momentum * self._m_u    + grad_u
        self._m_v    = self.momentum * self._m_v    + grad_v
        self._diag -= self.lr * self._m_diag
        self._u    -= self.lr * self._m_u
        self._v    -= self.lr * self._m_v

        # EMAs.
        if self._residual_ema is None:
            self._residual_ema = res_norm
            self._explained_ema = explained
        else:
            self._residual_ema = (1 - self.ema) * self._residual_ema + self.ema * res_norm
            self._explained_ema = (1 - self.ema) * self._explained_ema + self.ema * explained

        # Slide the buffer.
        self._prev_motor = m
        return self._stats()

    # ── Differentiable residual (Jun 2026 — C3 → trunk gradient) ─────

    def residual_diff(self,
                      h_motor: Optional[torch.Tensor],
                      h_sensory: Optional[torch.Tensor]) -> Optional[torch.Tensor]:
        """Return an autograd-tracked scalar ``||s − W·m||²`` *without*
        disturbing the internal SGD path.

        Design notes
        ------------
        - ``W`` (= diag + rank-1 (u,v)) is treated as **frozen** here:
          we detach the predictor parameters so this loss only updates
          the trunk's motor / sensory activations. The probe's own
          SGD (in :meth:`step`) keeps learning W on the detached
          tensors in parallel — clean separation of concerns.
        - Same-step formulation (``s_t`` predicted from ``m_t``, not
          ``m_{t-1}``) avoids holding a non-detached previous-motor
          tensor across training steps (which would either leak
          autograd graphs or require explicit retain_graph plumbing).
          The lagged formulation in :meth:`step` is preserved for
          telemetry because the lag-1 structure is what tells us
          whether the loop carries genuine *predictive* information
          (vs. mere copying).
        - Returns ``None`` if shapes don't match or either input is
          ``None`` — callers must guard.

        The semantics: the trunk learns to make ``h_motor`` and
        ``h_sensory`` mutually-predictable under the probe's current
        world model. Soft cycle-consistency, no extra parameters in
        the trunk, no autograd-graph accumulation.
        """
        if h_motor is None or h_sensory is None:
            return None
        if h_motor.shape != h_sensory.shape or h_motor.shape[-1] != self.dim:
            return None
        # Cast to predictor dtype/device but KEEP the autograd graph.
        m = h_motor.to(device=self._diag.device, dtype=torch.float32)
        s = h_sensory.to(device=self._diag.device, dtype=torch.float32)
        # Frozen-W prediction (detach predictor params so only the
        # trunk receives gradient from this term).
        d_det = self._diag.detach()
        u_det = self._u.detach()
        v_det = self._v.detach()
        diag_term = m * d_det
        coeff = (m * v_det).sum(dim=-1, keepdim=True)
        rank1_term = coeff * u_det
        pred = diag_term + rank1_term
        # Mean-squared residual (scalar, autograd-tracked through m & s).
        return (s - pred).pow(2).mean()

    # ── Stats ────────────────────────────────────────────────────────

    def _stats(self) -> Dict[str, float]:
        return {
            "pc_residual": float(self._residual_ema or 0.0),
            "pc_explained": float(self._explained_ema or 0.0),
            "pc_predictor_norm": float(
                (self._diag.pow(2).sum() + self._u.pow(2).sum()
                 + self._v.pow(2).sum()).sqrt().item()
            ),
        }
