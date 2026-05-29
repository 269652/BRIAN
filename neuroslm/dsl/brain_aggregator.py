# -*- coding: utf-8 -*-
"""DSL Brain aggregator — bit-identical port of Brain's total-loss formula.

This is the keystone of "DSL covers everything": it applies the exact
weighted sum that Brain's `forward_lm` builds at brain.py:1802-1824:

    total = w_lm * lm_loss
          + aux_w * ph_pred  * w_pred_coding * pred_coding
          + aux_w * ph_world * w_world       * world_loss
          + aux_w * ph_fwd   * w_forward     * fwd_reg * 0.01
          + aux_w * ph_motor * w_motor       * motor_loss
          + aux_w * ph_kl    * w_kl_world    * rssm_kl
          + aux_w * ph_novel * 0.05          * novel_aux_loss
          + aux_w * ph_cpc   * w_cpc         * cpc_loss
          + aux_w * ph_phi   * w_phi         * phi_loss_term
          + aux_w * (0.01 * id_drift + 0.01 * (1 - calm))   [orchestrator]

Components missing for a given preset (e.g. `cpc = None` in
`rcc_bowtie_30m_p4`) are skipped — same as Brain. The phase gates and
weights come from `dsl.maturity` which already mirrors brain.py:1794-1810
and brain.Brain._phase_gate bit-for-bit.

Parity is validated in `tests/dsl/test_brain_aggregator_parity.py`:
random LM + aux loss values + a fixed MAT → DSL aggregator output ==
Brain's total at atol 1e-6.
"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Optional

import torch

from neuroslm.dsl.maturity import (
    MaturityTracker, AuxWeights, TotalLossConfig, phase_gate,
)
from neuroslm.dsl.subsystems.orchestrator_adapter import (
    OrchestratorMetrics, orchestrator_aux_contribution,
)


@dataclass
class LossBundle:
    """Container for all aux loss tensors at a single forward step.

    Each field is optional: missing components are treated as zero
    (same as Brain's `_safe(None) → 0` pattern). The trunk-affecting
    set for `rcc_bowtie_30m_p4` is `lm + pred_coding + world + motor +
    forward + kl_world + phi + orchestrator-metrics`.
    """
    lm_loss: torch.Tensor
    pred_coding: Optional[torch.Tensor] = None
    world: Optional[torch.Tensor] = None
    forward: Optional[torch.Tensor] = None
    motor: Optional[torch.Tensor] = None
    kl_world: Optional[torch.Tensor] = None
    novel: Optional[torch.Tensor] = None
    cpc: Optional[torch.Tensor] = None
    phi: Optional[torch.Tensor] = None
    orchestrator: Optional[OrchestratorMetrics] = None


def _safe(t: Optional[torch.Tensor], like: torch.Tensor) -> torch.Tensor:
    """Brain's `_safe` analog: NaN/Inf → 0; None → 0 with matching device/dtype.

    Matches `brain.py:1780-1783`'s `_safe(t)` exactly.
    """
    if t is None:
        return torch.zeros((), device=like.device, dtype=like.dtype)
    if isinstance(t, torch.Tensor):
        return t.nan_to_num(0.0, posinf=0.0, neginf=0.0)
    val = float(t)
    if val != val:   # NaN check
        val = 0.0
    return torch.tensor(val, device=like.device, dtype=like.dtype)


def aggregate_total_loss(bundle: LossBundle,
                          mat: float,
                          config: Optional[TotalLossConfig] = None
                          ) -> torch.Tensor:
    """Apply Brain's brain.py:1802-1824 total-loss formula to the bundle.

    Args:
        bundle: every aux loss produced by this forward step (any can be None)
        mat:    current maturity scalar (drives the phase gates)
        config: weights + AuxWeights table; defaults to Brain's rcc_bowtie
                values from `maturity.AuxWeights`.

    Returns: scalar total loss, ready for `.backward()`.
    """
    cfg = config or TotalLossConfig()
    aw = cfg.aux
    lm = bundle.lm_loss

    total = cfg.w_lm * lm

    # ── PCH (trunk-affecting via h_lang in language model) ──
    if bundle.pred_coding is not None and bundle.pred_coding.numel() > 0:
        w = aw.scaled("pred_coding", mat)
        total = total + w * _safe(bundle.pred_coding, lm)

    # ── Sidecar aux losses (trained sub-modules; detached from trunk in p4) ──
    if bundle.world is not None and bundle.world.numel() > 0:
        w = aw.scaled("world", mat)
        total = total + w * _safe(bundle.world, lm)

    if bundle.forward is not None and bundle.forward.numel() > 0:
        # Brain's `w_forward * fwd_reg * 0.01` — the *0.01 bonus is rolled
        # into AuxWeights.forward, so `scaled("forward")` already gives
        # 0.20*0.01 * phase * aux_w. No double-application here.
        w = aw.scaled("forward", mat)
        total = total + w * _safe(bundle.forward, lm)

    if bundle.motor is not None and bundle.motor.numel() > 0:
        w = aw.scaled("motor", mat)
        total = total + w * _safe(bundle.motor, lm)

    if bundle.kl_world is not None and bundle.kl_world.numel() > 0:
        w = aw.scaled("kl_world", mat)
        total = total + w * _safe(bundle.kl_world, lm)

    if bundle.novel is not None and bundle.novel.numel() > 0:
        w = aw.scaled("novel", mat)
        total = total + w * _safe(bundle.novel, lm)

    if bundle.cpc is not None and bundle.cpc.numel() > 0:
        w = aw.scaled("cpc", mat)
        total = total + w * _safe(bundle.cpc, lm)

    if bundle.phi is not None and bundle.phi.numel() > 0:
        w = aw.scaled("phi", mat)
        total = total + w * _safe(bundle.phi, lm)

    # ── Orchestrator (id_drift + 1-calm), aux_w unscaled by phase gate ──
    if bundle.orchestrator is not None:
        total = total + orchestrator_aux_contribution(
            bundle.orchestrator, aux_w=aw.master_scale)

    return total


def brain_reference_total(bundle: LossBundle, mat: float,
                           config: Optional[TotalLossConfig] = None
                           ) -> torch.Tensor:
    """The expected total when applying Brain's *explicit* formula.

    Used by parity tests to assert `aggregate_total_loss == brain_reference`
    bit-for-bit. Brain's formula at brain.py:1802-1810 written out as a
    literal Python expression (no helpers, no dispatch) — so a divergence
    in `AuxWeights` constants or the phase-gate function is immediately
    visible as a single failing assertion.
    """
    cfg = config or TotalLossConfig()
    aw = cfg.aux
    lm = bundle.lm_loss

    def ph(center, width=0.08):
        return phase_gate(mat, center, width)

    aux_w = aw.master_scale
    # Brain's exact numeric constants from brain.py:1794-1810
    w_pred_coding  = 0.10
    w_world        = 0.30
    w_forward      = 0.20
    w_motor        = 0.05
    w_kl_world     = 0.10
    w_cpc          = 0.05
    w_phi          = 0.02

    z = torch.zeros((), device=lm.device, dtype=lm.dtype)
    total = cfg.w_lm * lm
    if bundle.pred_coding is not None:
        total = total + aux_w * ph(0.35) * w_pred_coding * _safe(bundle.pred_coding, lm)
    if bundle.world is not None:
        total = total + aux_w * ph(0.45) * w_world * _safe(bundle.world, lm)
    if bundle.forward is not None:
        total = total + aux_w * ph(0.50) * w_forward * _safe(bundle.forward, lm) * 0.01
    if bundle.motor is not None:
        total = total + aux_w * ph(0.50) * w_motor * _safe(bundle.motor, lm)
    if bundle.kl_world is not None:
        total = total + aux_w * ph(0.60) * w_kl_world * _safe(bundle.kl_world, lm)
    if bundle.novel is not None:
        total = total + aux_w * ph(0.55) * 0.05 * _safe(bundle.novel, lm)
    if bundle.cpc is not None:
        total = total + aux_w * ph(0.55) * w_cpc * _safe(bundle.cpc, lm)
    if bundle.phi is not None:
        total = total + aux_w * ph(0.60) * w_phi * _safe(bundle.phi, lm)
    if bundle.orchestrator is not None:
        total = total + aux_w * (
            0.01 * _safe(bundle.orchestrator.identity_drift, lm)
            + 0.01 * (1.0 - _safe(bundle.orchestrator.neural_calm, lm))
        )
    return total
