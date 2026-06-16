# -*- coding: utf-8 -*-
"""Tests that pin the GIF (Geometric Information Funnel) config pipeline.

The gif:{} block in arch.neuro must survive the full round-trip:

    arch.neuro  →  parse_training_config  →  TrainingConfig.gif (dict)
                →  GIFController.from_config  →  3 live mechanisms

Root-cause bug (fixed 0489d22): TrainingConfig never parsed the gif:{}
block, so getattr(cfg, 'gif', None) always returned None and _build_gif
silently set self._gif = None — making GIF a no-op on every deploy.

These tests ensure the bug cannot recur.
"""
import math
import pytest
from pathlib import Path

from neuroslm.dsl.training_config import (
    TrainingConfig,
    parse_training_config,
    load_training_config_from_arch,
)
from neuroslm.emergent.gif import (
    VBBAlphaSchedule,
    OODProbe,
    IsotropySchedule,
    GIFController,
)


# ── 1. Parser: gif:{} block → TrainingConfig.gif dict ────────────────

class TestGIFBlockParsing:
    """Ensures parse_training_config extracts gif:{} into cfg.gif."""

    MINIMAL_GIF_BLOCK = """
        gif: {
            enabled: true
            vbb_alpha_min: 0.001
            vbb_alpha_max: 0.05
            vbb_ramp_start: 2000
            vbb_ramp_end: 5000
            probe_n_seqs: 50
            probe_every: 100
            probe_ema_beta: 0.9
            iso_weight_max: 0.01
        }
    """

    def test_gif_block_parsed_into_dict(self):
        cfg = parse_training_config(self.MINIMAL_GIF_BLOCK)
        assert cfg.gif is not None, (
            "cfg.gif must not be None — the parser must extract the gif:{} block"
        )
        assert isinstance(cfg.gif, dict)

    def test_gif_enabled_is_true(self):
        cfg = parse_training_config(self.MINIMAL_GIF_BLOCK)
        assert cfg.gif["enabled"] is True or cfg.gif["enabled"] == "true"

    def test_gif_vbb_alpha_min_preserved(self):
        cfg = parse_training_config(self.MINIMAL_GIF_BLOCK)
        assert float(cfg.gif["vbb_alpha_min"]) == pytest.approx(0.001)

    def test_gif_vbb_alpha_max_preserved(self):
        cfg = parse_training_config(self.MINIMAL_GIF_BLOCK)
        assert float(cfg.gif["vbb_alpha_max"]) == pytest.approx(0.05)

    def test_gif_ramp_start_preserved(self):
        cfg = parse_training_config(self.MINIMAL_GIF_BLOCK)
        assert int(cfg.gif["vbb_ramp_start"]) == 2000

    def test_gif_ramp_end_preserved(self):
        cfg = parse_training_config(self.MINIMAL_GIF_BLOCK)
        assert int(cfg.gif["vbb_ramp_end"]) == 5000

    def test_gif_probe_n_seqs_preserved(self):
        cfg = parse_training_config(self.MINIMAL_GIF_BLOCK)
        assert int(cfg.gif["probe_n_seqs"]) == 50

    def test_gif_probe_every_preserved(self):
        cfg = parse_training_config(self.MINIMAL_GIF_BLOCK)
        assert int(cfg.gif["probe_every"]) == 100

    def test_gif_iso_weight_max_preserved(self):
        cfg = parse_training_config(self.MINIMAL_GIF_BLOCK)
        assert float(cfg.gif["iso_weight_max"]) == pytest.approx(0.01)

    def test_no_gif_block_returns_none(self):
        cfg = parse_training_config("")
        assert cfg.gif is None, (
            "cfg.gif must be None when no gif:{} block is declared"
        )

    def test_gif_disabled_still_parsed(self):
        body = "gif: { enabled: false }"
        cfg = parse_training_config(body)
        assert cfg.gif is not None, "gif:{} block must still be parsed even when disabled"

    def test_gif_adaptive_flag_preserved(self):
        body = """
            gif: {
                enabled: true
                adaptive: true
                target_gap_ratio: 1.5
                ramp_gain: 0.0002
            }
        """
        cfg = parse_training_config(body)
        assert cfg.gif is not None
        assert cfg.gif.get("adaptive") in (True, "true")


# ── 2. GIFController.from_config — DSL key names ────────────────────

class TestGIFControllerFromDSLKeys:
    """Ensures GIFController reads the DSL-native key names from arch.neuro."""

    def _make_cfg_with_gif(self, **overrides):
        gif = {
            "enabled": True,
            "vbb_alpha_min": 0.001,
            "vbb_alpha_max": 0.05,
            "vbb_ramp_start": 2000,
            "vbb_ramp_end": 5000,
            "probe_n_seqs": 50,
            "probe_every": 100,
            "probe_ema_beta": 0.9,
            "iso_weight_max": 0.01,
        }
        gif.update(overrides)
        cfg = TrainingConfig()
        cfg.gif = gif
        return cfg

    def test_controller_enabled(self):
        cfg = self._make_cfg_with_gif()
        ctrl = GIFController.from_config(cfg)
        assert ctrl.enabled is True

    def test_controller_disabled_when_gif_none(self):
        cfg = TrainingConfig()
        cfg.gif = None
        ctrl = GIFController.from_config(cfg)
        assert ctrl.enabled is False

    def test_controller_disabled_when_enabled_false(self):
        cfg = TrainingConfig()
        cfg.gif = {"enabled": False}
        ctrl = GIFController.from_config(cfg)
        assert ctrl.enabled is False

    def test_vbb_alpha_at_step_0(self):
        cfg = self._make_cfg_with_gif()
        ctrl = GIFController.from_config(cfg)
        assert ctrl.vbb_alpha(0) == pytest.approx(0.001)

    def test_vbb_alpha_before_ramp(self):
        cfg = self._make_cfg_with_gif()
        ctrl = GIFController.from_config(cfg)
        assert ctrl.vbb_alpha(1999) == pytest.approx(0.001)

    def test_vbb_alpha_at_ramp_end(self):
        cfg = self._make_cfg_with_gif()
        ctrl = GIFController.from_config(cfg)
        assert ctrl.vbb_alpha(5000) == pytest.approx(0.05)

    def test_vbb_alpha_after_ramp(self):
        cfg = self._make_cfg_with_gif()
        ctrl = GIFController.from_config(cfg)
        assert ctrl.vbb_alpha(10000) == pytest.approx(0.05)

    def test_vbb_alpha_mid_ramp(self):
        cfg = self._make_cfg_with_gif()
        ctrl = GIFController.from_config(cfg)
        # step 3500 = 50% through the 2000-5000 ramp
        alpha = ctrl.vbb_alpha(3500)
        expected = 0.001 + (0.05 - 0.001) * 0.5
        assert alpha == pytest.approx(expected, abs=1e-6)

    def test_isotropy_weight_before_ramp(self):
        cfg = self._make_cfg_with_gif()
        ctrl = GIFController.from_config(cfg)
        assert ctrl.isotropy_weight(1999) == pytest.approx(0.0)

    def test_isotropy_weight_at_ramp_end(self):
        cfg = self._make_cfg_with_gif()
        ctrl = GIFController.from_config(cfg)
        assert ctrl.isotropy_weight(5000) == pytest.approx(0.01)

    def test_isotropy_weight_after_ramp(self):
        cfg = self._make_cfg_with_gif()
        ctrl = GIFController.from_config(cfg)
        assert ctrl.isotropy_weight(10000) == pytest.approx(0.01)

    def test_probe_every_from_dsl_key(self):
        cfg = self._make_cfg_with_gif(probe_every=200)
        ctrl = GIFController.from_config(cfg)
        assert ctrl.ood_probe.probe_every == 200

    def test_probe_n_seqs_from_dsl_key(self):
        cfg = self._make_cfg_with_gif(probe_n_seqs=25)
        ctrl = GIFController.from_config(cfg)
        assert ctrl.ood_probe.n_seqs == 25

    def test_probe_ema_alpha_from_beta(self):
        """probe_ema_beta=0.9 → ema_alpha=0.1 (alpha = 1 - beta)."""
        cfg = self._make_cfg_with_gif(probe_ema_beta=0.9)
        ctrl = GIFController.from_config(cfg)
        assert ctrl.ood_probe.ema_alpha == pytest.approx(0.1)


# ── 3. GIFController.from_config — Python-native key names ──────────

class TestGIFControllerFromPythonKeys:
    """Ensures the Python-native key names also work (back-compat)."""

    def _make_cfg_with_python_keys(self):
        cfg = TrainingConfig()
        cfg.gif = {
            "enabled": True,
            "vbb_alpha_start": 0.002,
            "vbb_alpha_end": 0.08,
            "vbb_alpha_ramp_start": 1000,
            "vbb_alpha_ramp_end": 4000,
            "ood_probe_seqs": 30,
            "ood_probe_every": 50,
            "ood_probe_ema_alpha": 0.2,
            "isotropy_weight_max": 0.02,
            "isotropy_ramp_start": 1000,
            "isotropy_ramp_end": 4000,
        }
        return cfg

    def test_vbb_alpha_python_keys(self):
        cfg = self._make_cfg_with_python_keys()
        ctrl = GIFController.from_config(cfg)
        assert ctrl.vbb_alpha(0) == pytest.approx(0.002)
        assert ctrl.vbb_alpha(4000) == pytest.approx(0.08)

    def test_isotropy_python_keys(self):
        cfg = self._make_cfg_with_python_keys()
        ctrl = GIFController.from_config(cfg)
        assert ctrl.isotropy_weight(0) == pytest.approx(0.0)
        assert ctrl.isotropy_weight(4000) == pytest.approx(0.02)

    def test_probe_python_keys(self):
        cfg = self._make_cfg_with_python_keys()
        ctrl = GIFController.from_config(cfg)
        assert ctrl.ood_probe.n_seqs == 30
        assert ctrl.ood_probe.probe_every == 50
        assert ctrl.ood_probe.ema_alpha == pytest.approx(0.2)


# ── 4. Integration: arch.neuro file → GIFController ─────────────────

class TestGIFEndToEnd:
    """Full round-trip: arch.neuro on disk → GIFController.enabled."""

    def test_arch_with_gif_block_enables_controller(self, tmp_path):
        (tmp_path / "arch.neuro").write_text("""
            architecture test_gif { d_sem: 256 }

            training {
                gif: {
                    enabled: true
                    vbb_alpha_min: 0.001
                    vbb_alpha_max: 0.05
                    vbb_ramp_start: 2000
                    vbb_ramp_end: 5000
                    probe_n_seqs: 50
                    probe_every: 100
                    probe_ema_beta: 0.9
                    iso_weight_max: 0.01
                }
            }
        """, encoding="utf-8")

        cfg = load_training_config_from_arch(tmp_path)
        assert cfg.gif is not None, "gif must be parsed from arch.neuro"
        ctrl = GIFController.from_config(cfg)
        assert ctrl.enabled is True
        assert ctrl.vbb_alpha(0) == pytest.approx(0.001)
        assert ctrl.vbb_alpha(5000) == pytest.approx(0.05)
        assert ctrl.isotropy_weight(5000) == pytest.approx(0.01)
        assert ctrl.ood_probe.n_seqs == 50

    def test_arch_without_gif_block_disables_controller(self, tmp_path):
        (tmp_path / "arch.neuro").write_text("""
            architecture test_nogif { d_sem: 256 }
            training { grad_accum: 4 }
        """, encoding="utf-8")

        cfg = load_training_config_from_arch(tmp_path)
        assert cfg.gif is None
        ctrl = GIFController.from_config(cfg)
        assert ctrl.enabled is False

    def test_live_smollm_arch_has_gif(self):
        """Pin: the SmolLM arch that deploys to vast.ai MUST have gif enabled.

        This is the exact bug that caused 41205925 + 41208325 to train
        without GIF — the config was present but never parsed.
        """
        arch_root = Path(__file__).resolve().parents[2] / "architectures" / "SmolLM"
        if not (arch_root / "arch.neuro").is_file():
            pytest.skip("SmolLM arch not present in this checkout")
        cfg = load_training_config_from_arch(arch_root)
        assert cfg.gif is not None, (
            "SmolLM arch.neuro must have a gif:{} block — "
            "otherwise every deploy is a no-op GIF run"
        )
        ctrl = GIFController.from_config(cfg)
        assert ctrl.enabled is True, (
            "GIFController must be enabled for SmolLM — "
            "check gif.enabled in architectures/SmolLM/arch.neuro"
        )


# ── 5. VBBAlphaSchedule unit contracts ───────────────────────────────

class TestVBBAlphaSchedule:
    def test_default_schedule(self):
        s = VBBAlphaSchedule()
        assert s(0) == pytest.approx(0.001)
        assert s(3000) == pytest.approx(0.05)

    def test_clamps_below_start(self):
        s = VBBAlphaSchedule(ramp_start=100, ramp_end=200)
        assert s(50) == pytest.approx(s.alpha_start)

    def test_clamps_above_end(self):
        s = VBBAlphaSchedule(ramp_start=100, ramp_end=200)
        assert s(300) == pytest.approx(s.alpha_end)

    def test_monotonically_increasing(self):
        s = VBBAlphaSchedule()
        prev = s(0)
        for step in range(100, 6000, 100):
            cur = s(step)
            assert cur >= prev, f"alpha must be monotonic: step {step}"
            prev = cur

    def test_progress_kwarg_overrides_step(self):
        """When progress= is given, step is ignored."""
        s = VBBAlphaSchedule(ramp_start=500, ramp_end=3000)
        # progress=0.5 → midpoint regardless of step
        mid = s(0, progress=0.5)
        expected = 0.001 + (0.05 - 0.001) * 0.5
        assert mid == pytest.approx(expected, abs=1e-6)

    def test_progress_clamps_to_01(self):
        s = VBBAlphaSchedule()
        assert s(0, progress=-1.0) == pytest.approx(0.001)
        assert s(0, progress=2.0) == pytest.approx(0.05)


# ── 6. IsotropySchedule unit contracts ───────────────────────────────

class TestIsotropySchedule:
    def test_zero_before_ramp(self):
        s = IsotropySchedule()
        assert s(0) == pytest.approx(0.0)
        assert s(499) == pytest.approx(0.0)

    def test_max_at_ramp_end(self):
        s = IsotropySchedule()
        assert s(3000) == pytest.approx(0.01)

    def test_clamped_after_ramp(self):
        s = IsotropySchedule()
        assert s(99999) == pytest.approx(0.01)

    def test_monotonically_increasing(self):
        s = IsotropySchedule()
        prev = s(0)
        for step in range(100, 6000, 100):
            cur = s(step)
            assert cur >= prev, f"isotropy must be monotonic: step {step}"
            prev = cur

    def test_progress_kwarg(self):
        s = IsotropySchedule(weight_max=0.02)
        assert s(0, progress=0.5) == pytest.approx(0.01)


# ── 7. Adaptive GIF ramp ────────────────────────────────────────────

class TestAdaptiveGIFRamp:
    """Tests for the gap-ratio-driven adaptive ramp controller."""

    def _make_adaptive_ctrl(self, **overrides):
        gif = {
            "enabled": True,
            "adaptive": True,
            "target_gap_ratio": 1.5,
            "ramp_gain": 0.001,       # high gain for test visibility
            "min_ramp_speed": 0.0001,
            "vbb_alpha_min": 0.001,
            "vbb_alpha_max": 0.05,
            "vbb_ramp_start": 500,
            "vbb_ramp_end": 3000,
            "probe_n_seqs": 10,
            "probe_every": 100,
            "probe_ema_beta": 0.9,
            "iso_weight_max": 0.01,
        }
        gif.update(overrides)
        cfg = TrainingConfig()
        cfg.gif = gif
        return GIFController.from_config(cfg)

    def test_adaptive_flag_parsed(self):
        ctrl = self._make_adaptive_ctrl()
        assert ctrl.adaptive is True

    def test_static_mode_default(self):
        """adaptive defaults to False when not specified."""
        cfg = TrainingConfig()
        cfg.gif = {"enabled": True, "vbb_alpha_min": 0.001}
        ctrl = GIFController.from_config(cfg)
        assert ctrl.adaptive is False

    def test_progress_starts_at_zero(self):
        ctrl = self._make_adaptive_ctrl()
        assert ctrl.progress == 0.0

    def test_update_advances_progress(self):
        ctrl = self._make_adaptive_ctrl()
        # Simulate: OOD probe returns CE=6.0, train EMA=4.0
        # gap_ratio = exp(6-4) = exp(2) ≈ 7.39  >> target 1.5
        ctrl.ood_probe._n_evals = 1
        ctrl.ood_probe._ema = 6.0
        ctrl.update(step=100, lm_loss_ema=4.0)
        assert ctrl.progress > 0.0, "progress must advance when gap > target"

    def test_progress_monotonically_increasing(self):
        """Progress should never decrease across updates."""
        ctrl = self._make_adaptive_ctrl()
        ctrl.ood_probe._n_evals = 1
        ctrl.ood_probe._ema = 5.5
        prev = ctrl.progress
        for step in range(100, 1100, 100):
            ctrl.update(step=step, lm_loss_ema=4.0)
            assert ctrl.progress >= prev, (
                f"progress must be monotonic: step {step}"
            )
            prev = ctrl.progress

    def test_high_gap_accelerates_ramp(self):
        """With a large gap, progress should advance faster."""
        ctrl_high = self._make_adaptive_ctrl()
        ctrl_low = self._make_adaptive_ctrl()

        # Both have probe data
        ctrl_high.ood_probe._n_evals = 1
        ctrl_low.ood_probe._n_evals = 1

        # High gap: OOD=7.0, train=4.0 → gap=exp(3)≈20
        ctrl_high.ood_probe._ema = 7.0
        ctrl_high.update(step=100, lm_loss_ema=4.0)

        # Low gap: OOD=4.5, train=4.0 → gap=exp(0.5)≈1.65
        ctrl_low.ood_probe._ema = 4.5
        ctrl_low.update(step=100, lm_loss_ema=4.0)

        assert ctrl_high.progress > ctrl_low.progress, (
            "higher gap ratio must produce faster ramp advancement"
        )

    def test_gap_below_target_creeps(self):
        """When gap < target, only min_ramp_speed advances progress."""
        ctrl = self._make_adaptive_ctrl(min_ramp_speed=0.001)
        ctrl.ood_probe._n_evals = 1
        # gap = exp(4.2 - 4.0) = exp(0.2) ≈ 1.22 < target 1.5
        ctrl.ood_probe._ema = 4.2
        ctrl.update(step=100, lm_loss_ema=4.0)
        assert ctrl.progress == pytest.approx(0.001, abs=1e-6), (
            "with gap < target, only min_ramp_speed should apply"
        )

    def test_static_floor_respected(self):
        """Progress never falls below the static step-based schedule."""
        ctrl = self._make_adaptive_ctrl(min_ramp_speed=0.0)
        # At step 1750 (50% through 500-3000 ramp), static floor = 0.5
        ctrl.update(step=1750, lm_loss_ema=4.0)
        assert ctrl.progress >= 0.5 - 1e-6, (
            "progress must respect the static floor"
        )

    def test_vbb_alpha_uses_progress(self):
        """In adaptive mode, vbb_alpha uses the progress variable."""
        ctrl = self._make_adaptive_ctrl()
        ctrl._progress = 0.5
        alpha = ctrl.vbb_alpha(0)  # step doesn't matter in adaptive
        expected = 0.001 + (0.05 - 0.001) * 0.5
        assert alpha == pytest.approx(expected, abs=1e-6)

    def test_isotropy_uses_progress(self):
        ctrl = self._make_adaptive_ctrl()
        ctrl._progress = 1.0
        assert ctrl.isotropy_weight(0) == pytest.approx(0.01)

    def test_progress_clamped_at_1(self):
        ctrl = self._make_adaptive_ctrl(ramp_gain=1.0)
        ctrl.ood_probe._n_evals = 1
        ctrl.ood_probe._ema = 10.0  # massive gap
        for _ in range(100):
            ctrl.update(step=5000, lm_loss_ema=2.0)
        assert ctrl.progress == pytest.approx(1.0), (
            "progress must clamp at 1.0"
        )

    def test_gap_ratio_telemetry(self):
        ctrl = self._make_adaptive_ctrl()
        ctrl.ood_probe._n_evals = 1
        ctrl.ood_probe._ema = 5.0
        ctrl.update(step=100, lm_loss_ema=4.0)
        expected_gap = math.exp(5.0 - 4.0)
        assert ctrl.last_gap_ratio == pytest.approx(expected_gap, rel=1e-3)

    def test_no_ood_data_uses_min_speed(self):
        """Before probe loads, only min_ramp_speed advances."""
        ctrl = self._make_adaptive_ctrl(min_ramp_speed=0.01)
        ctrl.update(step=100, lm_loss_ema=4.0)
        assert ctrl.progress == pytest.approx(0.01, abs=1e-6)

    def test_live_smollm_arch_is_adaptive(self):
        """Pin: SmolLM must deploy with adaptive GIF."""
        arch_root = Path(__file__).resolve().parents[2] / "architectures" / "SmolLM"
        if not (arch_root / "arch.neuro").is_file():
            pytest.skip("SmolLM arch not present")
        cfg = load_training_config_from_arch(arch_root)
        ctrl = GIFController.from_config(cfg)
        assert ctrl.adaptive is True, (
            "SmolLM GIF must be adaptive — check gif.adaptive in arch.neuro"
        )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
