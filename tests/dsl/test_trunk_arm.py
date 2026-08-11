# -*- coding: utf-8 -*-
"""trunk-100m — the two-layer doctrine's first A/B arm (§14).

control-100m topology + the SmolLM expert roster + KL-distill assist +
`trunk_pretrain: true`, and NOTHING else. The single question this arm
asks: does detached-teacher distillation accelerate a vanilla trunk when
no cognition-layer loss interferes? Comparator: control-100m (identical
trunk, no experts) at matched steps/ctx/params via `brian ood compare`.

Contracts:
  A. Recipe parity with control-100m — the same SHARED_FIELDS discipline
     TestRecipeParity pins between control and SmolLM. A drifted lr or
     dropout turns the distill A/B into an optimizer study.
  B. Trunk topology identical to control-100m: block_pattern "standard",
     geometry_adapters false, cosine_head false, pred_coding_weight 0.0,
     and the SAME 100m scale dims (640/14/10, ctx 2048, batch 4).
  C. trunk_pretrain is TRUE here and FALSE on control-100m and SmolLM
     (anti-drift both directions: the doctrine arm must gate cognition;
     the existing arms must keep their historical behaviour).
  D. The expert roster + distillation stack match SmolLM's proven
     values (same teachers, same CFD funnel) — the assist is copied,
     not invented.
"""
from __future__ import annotations

from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
TRUNK_ARCH = REPO_ROOT / "architectures" / "trunk-100m"
CONTROL_ARCH = REPO_ROOT / "architectures" / "control-100m"
SMOLLM_ARCH = REPO_ROOT / "architectures" / "SmolLM"

# Same list TestRecipeParity (test_control_arm.py) pins control↔SmolLM.
SHARED_FIELDS = ("learning_rate", "weight_decay", "grad_accum",
                 "grad_clip", "z_loss", "dropout", "label_smoothing",
                 "flooding_level", "stochastic_depth", "llrd",
                 "rope_base", "optimizer", "warmup_steps",
                 "min_lr_ratio")


def _load(arch_root):
    from neuroslm.dsl.training_config import load_training_config_from_arch
    if not (arch_root / "arch.neuro").is_file():
        pytest.fail(f"arch missing: {arch_root}/arch.neuro")
    return load_training_config_from_arch(arch_root)


class TestTrunkRecipeParity:
    def test_recipe_identical_to_control(self):
        trunk = _load(TRUNK_ARCH)
        ctrl = _load(CONTROL_ARCH)
        for f in SHARED_FIELDS:
            assert getattr(trunk, f) == getattr(ctrl, f), (
                f"training-recipe drift on {f!r}: trunk-100m="
                f"{getattr(trunk, f)!r} vs control={getattr(ctrl, f)!r}")


class TestTrunkTopologyIsControl:
    def test_vanilla_trunk_knobs(self):
        trunk = _load(TRUNK_ARCH)
        assert trunk.block_pattern == "standard"
        assert trunk.geometry_adapters is False
        assert trunk.cosine_head is False
        assert trunk.pred_coding_weight == 0.0

    def test_100m_scale_dims_match_control(self):
        trunk = _load(TRUNK_ARCH)
        ctrl = _load(CONTROL_ARCH)
        tv = trunk.scales.variants["100m"]
        cv = ctrl.scales.variants["100m"]
        for f in ("d_model", "depth", "n_heads", "max_ctx",
                  "batch_size", "seq_len", "grad_accum"):
            assert getattr(tv, f) == getattr(cv, f), (
                f"100m scale drift on {f!r}: trunk-100m="
                f"{getattr(tv, f)!r} vs control={getattr(cv, f)!r}")
        assert trunk.scales.default == "100m"


class TestDoctrineFlag:
    def test_trunk_pretrain_true_here(self):
        assert _load(TRUNK_ARCH).trunk_pretrain is True

    def test_existing_arms_keep_flag_false(self):
        assert _load(CONTROL_ARCH).trunk_pretrain is False
        assert _load(SMOLLM_ARCH).trunk_pretrain is False


class TestDistillAssistStack:
    def test_expert_roster_matches_smollm(self):
        trunk = _load(TRUNK_ARCH)
        smol = _load(SMOLLM_ARCH)
        t_roster = [(e.id, e.domain) for e in trunk.multi_cortex.experts]
        s_roster = [(e.id, e.domain) for e in smol.multi_cortex.experts]
        assert t_roster == s_roster, (
            "the assist is COPIED from the proven SmolLM roster, not "
            f"invented: trunk={t_roster} vs smollm={s_roster}")
        assert all(e.freeze for e in trunk.multi_cortex.experts)

    def test_distillation_enabled_with_cfd(self):
        mc = _load(TRUNK_ARCH).multi_cortex
        assert mc.enabled is True
        assert mc.distillation_enabled is True
        assert mc.cfd_enabled is True, (
            "CFD funnel is what made the SmolLM2 teacher safe (H23/H006)"
            " — the assist ships with it")
        assert mc.distillation_temperature == pytest.approx(2.0)
        assert mc.distillation_gap_floor == pytest.approx(0.1)
        assert mc.distillation_gap_ceiling == pytest.approx(2.0)
