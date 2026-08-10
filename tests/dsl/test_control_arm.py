# -*- coding: utf-8 -*-
"""TDD contracts for the vanilla control arm (H59 falsification apparatus).

The "topology over scale" thesis (README line 12, findings.md Layer B) is
only falsifiable against a param-matched, data-matched, step-matched
vanilla transformer trained by the SAME pipeline. The legacy control
(`train.py --baseline`) is unreachable from `train_dsl.py`, so every run
since B4 has had no control arm (findings.md H12 sidebar).

Contracts:
  A. training { block_pattern: "standard" | "interleave" } — parsed,
     default "interleave" (today's behaviour), invalid value rejected.
  B. training { geometry_adapters: false } — parsed, default true.
  C. training { pred_coding_weight: 0.0 } — parsed, default -1.0
     (= keep the AuxWeights default); 0.0 kills the PCH aux term so a
     control run is pure CE (+ shared stabilisers).
  D. DSLLanguageCortex(block_pattern="standard") builds ONLY
     StandardBlock layers (no DiffBlock, no ModBlock); default builds
     the i%3 interleave exactly as before.
  E. geometry_adapters=False → every adapter is nn.Identity.
  F. architectures/control-100m parses to the vanilla configuration:
     standard blocks, no adapters, no cosine head, zero PCH aux, no
     multi-cortex experts, no novel-topology modules.
  G. Shared-recipe parity: control-100m and SmolLM declare identical
     optimizer/regularisation hyperparameters (lr, wd, grad_accum,
     grad_clip, z_loss, dropout, rope_base) — the arms differ ONLY in
     topology mechanisms, never in the training recipe.
  H. Param parity: at the 100m dims (d=640, depth=8, heads=10) the
     control trunk's trainable-param count is within 5% of the BRIAN
     trunk's, so "matched params" in findings rows is a measured fact.
"""
from __future__ import annotations

from pathlib import Path

import pytest
import torch
import torch.nn as nn

REPO_ROOT = Path(__file__).resolve().parents[2]
CONTROL_ARCH = REPO_ROOT / "architectures" / "control-100m"
SMOLLM_ARCH = REPO_ROOT / "architectures" / "SmolLM"


# ══════════════════════════════════════════════════════════════════════
# A/B/C. Config parsing
# ══════════════════════════════════════════════════════════════════════

class TestBlockPatternConfig:
    def test_default_is_interleave(self):
        from neuroslm.dsl.training_config import parse_training_config
        cfg = parse_training_config("")
        assert cfg.block_pattern == "interleave"

    def test_parses_standard(self):
        from neuroslm.dsl.training_config import parse_training_config
        cfg = parse_training_config('block_pattern: "standard"')
        assert cfg.block_pattern == "standard"

    def test_parses_interleave_explicitly(self):
        from neuroslm.dsl.training_config import parse_training_config
        cfg = parse_training_config('block_pattern: "interleave"')
        assert cfg.block_pattern == "interleave"

    def test_rejects_unknown_pattern(self):
        from neuroslm.dsl.training_config import parse_training_config
        with pytest.raises(ValueError):
            parse_training_config('block_pattern: "banana"')


class TestGeometryAdaptersConfig:
    def test_default_is_true(self):
        from neuroslm.dsl.training_config import parse_training_config
        cfg = parse_training_config("")
        assert cfg.geometry_adapters is True

    def test_parses_false(self):
        from neuroslm.dsl.training_config import parse_training_config
        cfg = parse_training_config("geometry_adapters: false")
        assert cfg.geometry_adapters is False


class TestPredCodingWeightConfig:
    def test_default_keeps_aux(self):
        from neuroslm.dsl.training_config import parse_training_config
        cfg = parse_training_config("")
        assert cfg.pred_coding_weight == pytest.approx(-1.0)

    def test_parses_zero(self):
        from neuroslm.dsl.training_config import parse_training_config
        cfg = parse_training_config("pred_coding_weight: 0.0")
        assert cfg.pred_coding_weight == pytest.approx(0.0)

    def test_parses_positive(self):
        from neuroslm.dsl.training_config import parse_training_config
        cfg = parse_training_config("pred_coding_weight: 0.25")
        assert cfg.pred_coding_weight == pytest.approx(0.25)


# ══════════════════════════════════════════════════════════════════════
# D/E. Cortex construction honours the knobs
# ══════════════════════════════════════════════════════════════════════

def _tiny_cortex(**kw):
    from neuroslm.dsl.nn_lang import build_dsl_language_cortex
    return build_dsl_language_cortex(
        vocab=100, d_model=64, depth=6, n_heads=4, max_ctx=32, **kw)


class TestStandardBlockPattern:
    def test_standard_pattern_builds_only_standard_blocks(self):
        lm = _tiny_cortex(block_pattern="standard")
        names = [type(b).__name__ for b in lm.blocks]
        assert names == ["StandardBlock"] * 6, names

    def test_default_pattern_is_the_original_interleave(self):
        lm = _tiny_cortex()
        names = [type(b).__name__ for b in lm.blocks]
        assert names == ["StandardBlock", "DiffBlock", "ModBlock"] * 2, names

    def test_standard_pattern_forward_smoke(self):
        torch.manual_seed(0)
        lm = _tiny_cortex(block_pattern="standard", geometry_adapters=False)
        lm.eval()
        ids = torch.randint(0, 100, (2, 16))
        with torch.no_grad():
            logits = lm(ids)
        assert logits.shape == (2, 16, 100)
        assert torch.isfinite(logits).all()

    def test_standard_pattern_gradient_flows(self):
        torch.manual_seed(0)
        lm = _tiny_cortex(block_pattern="standard", geometry_adapters=False)
        ids = torch.randint(0, 100, (2, 16))
        logits = lm(ids)
        logits.sum().backward()
        grads = [p.grad for p in lm.parameters() if p.grad is not None]
        assert grads and any(g.abs().sum() > 0 for g in grads)


class TestGeometryAdaptersKnob:
    def test_adapters_off_are_identity(self):
        lm = _tiny_cortex(geometry_adapters=False)
        assert all(isinstance(a, nn.Identity) for a in lm.adapters)

    def test_adapters_default_on(self):
        lm = _tiny_cortex()
        names = {type(a).__name__ for a in lm.adapters}
        assert names == {"NeuralGeometryAdapter"}, names


# ══════════════════════════════════════════════════════════════════════
# F. The control arch parses to the vanilla configuration
# ══════════════════════════════════════════════════════════════════════

class TestControlArch:
    @pytest.fixture(scope="class")
    def cfg(self):
        from neuroslm.dsl.training_config import load_training_config_from_arch
        if not (CONTROL_ARCH / "arch.neuro").is_file():
            pytest.fail(f"control arch missing: {CONTROL_ARCH}/arch.neuro")
        return load_training_config_from_arch(CONTROL_ARCH)

    def test_standard_blocks(self, cfg):
        assert cfg.block_pattern == "standard"

    def test_no_geometry_adapters(self, cfg):
        assert cfg.geometry_adapters is False

    def test_no_cosine_head(self, cfg):
        assert cfg.cosine_head is False

    def test_pch_aux_zeroed(self, cfg):
        assert cfg.pred_coding_weight == pytest.approx(0.0)

    def test_no_multi_cortex_experts(self, cfg):
        mc = getattr(cfg, "multi_cortex", None)
        assert mc is None or not getattr(mc, "enabled", False)

    def test_no_novel_topology_modules(self, cfg):
        for field in ("grid_positions", "episodic_memory",
                      "surprise_head", "nfo"):
            v = getattr(cfg, field, None)
            enabled = bool(v.get("enabled", bool(v))) if isinstance(v, dict) \
                else bool(v)
            assert not enabled, f"{field} must be OFF in the control arm"

    def test_no_pct_trunk(self, cfg):
        assert cfg.pct_trunk == pytest.approx(0.0)
        assert cfg.tonnetz_period == 0


# ══════════════════════════════════════════════════════════════════════
# G. Shared-recipe parity between the arms
# ══════════════════════════════════════════════════════════════════════

class TestRecipeParity:
    """The two arms must differ ONLY in topology mechanisms. A drifted
    lr or dropout would silently turn the A/B into an optimizer study."""

    SHARED_FIELDS = ("learning_rate", "weight_decay", "grad_accum",
                     "grad_clip", "z_loss", "dropout", "label_smoothing",
                     "flooding_level", "stochastic_depth", "llrd",
                     "rope_base", "optimizer", "warmup_steps",
                     "min_lr_ratio")

    def test_recipe_identical(self):
        from neuroslm.dsl.training_config import load_training_config_from_arch
        if not (CONTROL_ARCH / "arch.neuro").is_file():
            pytest.fail(f"control arch missing: {CONTROL_ARCH}/arch.neuro")
        ctrl = load_training_config_from_arch(CONTROL_ARCH)
        brian = load_training_config_from_arch(SMOLLM_ARCH)
        for f in self.SHARED_FIELDS:
            assert getattr(ctrl, f) == getattr(brian, f), (
                f"training-recipe drift on {f!r}: control="
                f"{getattr(ctrl, f)!r} vs SmolLM={getattr(brian, f)!r}")


# ══════════════════════════════════════════════════════════════════════
# H. Param parity at the 100m dims
# ══════════════════════════════════════════════════════════════════════

class TestParamParity:
    @staticmethod
    def _dims_of(arch_root, scale="100m"):
        from neuroslm.dsl.training_config import load_training_config_from_arch
        cfg = load_training_config_from_arch(arch_root)
        sv = cfg.scales.variants[scale]
        return cfg, dict(vocab=50257,
                         d_model=int(sv.d_model), depth=int(sv.depth),
                         n_heads=int(sv.n_heads),
                         max_ctx=128)  # ctx sizes buffers, not params

    def test_trainable_params_within_5_percent(self):
        """Instantiate both trunks at each arch's OWN declared 100m
        dims and compare trainable-parameter counts. This is the number
        findings.md rows will cite as 'param-matched'. (Measured
        2026-07-12: BRIAN trunk 134.8M at 640/8 with adapters+diff+MoD;
        vanilla standard-only is 5.15M/layer → depth 14 ≈ 136.1M.)"""
        from neuroslm.dsl.nn_lang import build_dsl_language_cortex
        ctrl_cfg, ctrl_dims = self._dims_of(CONTROL_ARCH)
        brian_cfg, brian_dims = self._dims_of(SMOLLM_ARCH)
        control = build_dsl_language_cortex(
            block_pattern=ctrl_cfg.block_pattern,
            geometry_adapters=ctrl_cfg.geometry_adapters,
            cosine_head=ctrl_cfg.cosine_head, **ctrl_dims)
        n_control = sum(p.numel() for p in control.parameters()
                        if p.requires_grad)
        del control
        brian = build_dsl_language_cortex(
            block_pattern=brian_cfg.block_pattern,
            geometry_adapters=brian_cfg.geometry_adapters,
            cosine_head=brian_cfg.cosine_head, **brian_dims)
        n_brian = sum(p.numel() for p in brian.parameters()
                      if p.requires_grad)
        del brian
        rel = abs(n_control - n_brian) / max(n_control, n_brian)
        assert rel < 0.05, (
            f"param mismatch {rel:.1%}: control={n_control/1e6:.1f}M "
            f"vs BRIAN trunk={n_brian/1e6:.1f}M — adjust control depth "
            f"or document the delta in the findings row")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
