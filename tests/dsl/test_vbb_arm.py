# -*- coding: utf-8 -*-
"""TDD contracts for the VBB-waist arm (H59/H60 falsification ladder, arm-2).

Motivated by live evidence from the topology-100m deploy (H60 fix
applied): trunk representational health recovered (PR 1.5->8-10, R^2
0.47->0.93) but the OOD gap widened monotonically anyway
(gap_v2 1.30->2.77 through step 4500) while traindist ppl kept dropping
and wikitext ppl plateaued — an overfitting signature, not a
representation-collapse one. The Variational Bowtie Bottleneck is the
next rung: an explicit information bottleneck at the trunk's motor pole
meant to cap I(X; h_motor) and force generalizable-only statistics.

Contracts:
  A. architectures/vbb-100m parses with vbb_alpha > 0 and
     pc_reentry_weight > 0 (both required: the harness gates the whole
     PC-reentry call site on pc_reentry_weight>0, and the VBB-vs-legacy
     branch inside it on vbb_alpha>0 — declaring only one is a no-op).
  B. Every other training-recipe field (recipe + block_pattern +
     geometry_adapters + cosine_head + pred_coding_weight) is identical
     to topology-100m — this arm changes ONLY the VBB/pc_reentry block.
  C. The harness actually builds live VBB modules (sigma_head, log_beta)
     when constructed from this config — not just parsed, but wired.
  D. Trainable params are within a few percent of topology-100m's (VBB
     adds only a few hundred K params on a 135M trunk).
"""
from __future__ import annotations

from pathlib import Path

import pytest
import torch
import torch.nn as nn

REPO_ROOT = Path(__file__).resolve().parents[2]
TOPOLOGY_ARCH = REPO_ROOT / "architectures" / "topology-100m"
VBB_ARCH = REPO_ROOT / "architectures" / "vbb-100m"


def _load(arch_root):
    from neuroslm.dsl.training_config import load_training_config_from_arch
    if not (arch_root / "arch.neuro").is_file():
        pytest.fail(f"arch missing: {arch_root}/arch.neuro")
    return load_training_config_from_arch(arch_root)


# ══════════════════════════════════════════════════════════════════════
# A. vbb-100m parses with both required knobs on
# ══════════════════════════════════════════════════════════════════════

class TestVBBArmConfig:
    @pytest.fixture(scope="class")
    def cfg(self):
        return _load(VBB_ARCH)

    def test_vbb_alpha_positive(self, cfg):
        assert cfg.vbb_alpha > 0.0

    def test_pc_reentry_weight_positive(self, cfg):
        """The whole PC-reentry call site in compute_loss is gated on
        this being > 0 — without it, vbb_alpha is parsed but never
        reaches _compute_pc_reentry_loss at all."""
        assert cfg.pc_reentry_weight > 0.0

    def test_pc_reentry_nt_gate_on(self, cfg):
        assert cfg.pc_reentry_nt_gate is True

    def test_anti_collapse_guards_set(self, cfg):
        """free_bits / log_beta_max / entropy_eta — the MDRV-VBB
        safeguards against beta/sigma co-collapse. Copied from SmolLM's
        proven config, not invented."""
        assert cfg.vbb_free_bits > 0.0
        assert cfg.vbb_log_beta_max > 0.0
        assert cfg.vbb_entropy_eta > 0.0

    def test_matches_smollm_proven_values_except_alpha(self):
        """vbb_alpha is deliberately fixed (not GIF-ramped) for a clean
        single-variable test; every other VBB field must match SmolLM's
        already-deployed, already-tested config exactly — these are not
        invented numbers."""
        smollm_cfg = _load(REPO_ROOT / "architectures" / "SmolLM")
        vbb_cfg = _load(VBB_ARCH)
        for f in ("vbb_beta_init", "pc_reentry_weight", "pc_reentry_nt_gate",
                 "vbb_free_bits", "vbb_log_beta_max", "vbb_entropy_eta",
                 "vbb_curvature", "motor_curvature"):
            assert getattr(vbb_cfg, f) == getattr(smollm_cfg, f), (
                f"{f!r}: vbb-100m={getattr(vbb_cfg, f)!r} vs "
                f"SmolLM={getattr(smollm_cfg, f)!r} — should match the "
                f"proven config exactly")


# ══════════════════════════════════════════════════════════════════════
# B. Single-variable discipline vs topology-100m
# ══════════════════════════════════════════════════════════════════════

class TestVBBArmIsolatesOneVariable:
    RECIPE_FIELDS = ("learning_rate", "weight_decay", "grad_accum",
                     "grad_clip", "z_loss", "dropout", "label_smoothing",
                     "flooding_level", "stochastic_depth", "llrd",
                     "rope_base", "optimizer", "warmup_steps",
                     "min_lr_ratio")

    def test_recipe_identical_to_topology(self):
        topo = _load(TOPOLOGY_ARCH)
        vbb = _load(VBB_ARCH)
        for f in self.RECIPE_FIELDS:
            assert getattr(vbb, f) == getattr(topo, f), (
                f"training-recipe drift on {f!r}: vbb-100m="
                f"{getattr(vbb, f)!r} vs topology-100m={getattr(topo, f)!r}")

    def test_block_topology_identical_to_topology(self):
        topo = _load(TOPOLOGY_ARCH)
        vbb = _load(VBB_ARCH)
        assert vbb.block_pattern == topo.block_pattern == "interleave"
        assert vbb.geometry_adapters == topo.geometry_adapters is True
        assert vbb.cosine_head == topo.cosine_head
        assert vbb.pred_coding_weight == pytest.approx(topo.pred_coding_weight)

    def test_topology_has_vbb_off(self):
        """The comparison baseline (topology-100m) must NOT have picked
        up VBB by accident — confirms this really is the only delta."""
        topo = _load(TOPOLOGY_ARCH)
        assert topo.vbb_alpha == pytest.approx(0.0)
        assert topo.pc_reentry_weight == pytest.approx(0.0)

    def test_no_multi_cortex_or_novel_topology(self):
        cfg = _load(VBB_ARCH)
        mc = getattr(cfg, "multi_cortex", None)
        assert mc is None or not getattr(mc, "enabled", False)
        for field in ("grid_positions", "episodic_memory",
                      "surprise_head", "nfo"):
            v = getattr(cfg, field, None)
            enabled = bool(v.get("enabled", bool(v))) if isinstance(v, dict) \
                else bool(v)
            assert not enabled, f"{field} must stay OFF — VBB is the only add"


# ══════════════════════════════════════════════════════════════════════
# C. The harness actually builds live VBB modules from this config
# ══════════════════════════════════════════════════════════════════════

class TestVBBArmBuildsLiveModules:
    def test_harness_constructs_vbb_sigma_head_and_log_beta(self):
        from neuroslm.harness import BRIANHarness
        cfg = _load(VBB_ARCH)

        class _StubLM(nn.Module):
            def __init__(self, d):
                super().__init__()
                self.proj = nn.Linear(d, d)

        h = BRIANHarness.from_language_model(
            language_model=_StubLM(640), vocab_size=50257, d_sem=640,
            training_config=cfg,
        )
        assert h._vbb_sigma_head is not None
        assert isinstance(h._vbb_sigma_head, nn.Linear)
        assert h._vbb_log_beta is not None
        assert isinstance(h._vbb_log_beta, nn.Parameter)

    def test_vbb_loss_activates_on_stashed_activations(self):
        from neuroslm.harness import BRIANHarness
        cfg = _load(VBB_ARCH)

        class _StubLM(nn.Module):
            def __init__(self, d):
                super().__init__()
                self.proj = nn.Linear(d, d)
                self._last_h_motor = None
                self._last_h_sensory = None

        h = BRIANHarness.from_language_model(
            language_model=_StubLM(640), vocab_size=50257, d_sem=640,
            training_config=cfg,
        )
        mu = torch.randn(2, 4, 640, requires_grad=True)
        s = torch.randn(2, 4, 640, requires_grad=True)
        h.language_model._last_h_motor = mu
        h.language_model._last_h_sensory = s
        loss = h._compute_pc_reentry_loss(base_weight=cfg.pc_reentry_weight)
        assert loss is not None
        assert loss.requires_grad
        loss.backward()
        assert mu.grad is not None and mu.grad.abs().sum() > 0
        assert h._vbb_sigma_head.weight.grad is not None


# ══════════════════════════════════════════════════════════════════════
# D. Param parity vs topology-100m
# ══════════════════════════════════════════════════════════════════════

class TestVBBArmParamParity:
    def test_trunk_params_match_topology(self):
        """VBB params live on the HARNESS, not the trunk cortex itself —
        the underlying DSLLanguageCortex is unchanged, so trainable
        params at the cortex level must match topology-100m exactly."""
        from neuroslm.dsl.nn_lang import build_dsl_language_cortex
        from neuroslm.dsl.training_config import load_training_config_from_arch

        topo_cfg = load_training_config_from_arch(TOPOLOGY_ARCH)
        vbb_cfg = load_training_config_from_arch(VBB_ARCH)
        topo_sv = topo_cfg.scales.variants[topo_cfg.scales.default]
        vbb_sv = vbb_cfg.scales.variants[vbb_cfg.scales.default]

        dims = dict(vocab=50257, max_ctx=128)
        topo = build_dsl_language_cortex(
            d_model=int(topo_sv.d_model), depth=int(topo_sv.depth),
            n_heads=int(topo_sv.n_heads),
            block_pattern=topo_cfg.block_pattern,
            geometry_adapters=topo_cfg.geometry_adapters,
            cosine_head=topo_cfg.cosine_head, **dims)
        n_topo = sum(p.numel() for p in topo.parameters() if p.requires_grad)
        del topo
        vbb = build_dsl_language_cortex(
            d_model=int(vbb_sv.d_model), depth=int(vbb_sv.depth),
            n_heads=int(vbb_sv.n_heads),
            block_pattern=vbb_cfg.block_pattern,
            geometry_adapters=vbb_cfg.geometry_adapters,
            cosine_head=vbb_cfg.cosine_head, **dims)
        n_vbb = sum(p.numel() for p in vbb.parameters() if p.requires_grad)
        del vbb
        assert n_topo == n_vbb, (
            "vbb-100m's trunk cortex must be byte-identical in param "
            "count to topology-100m's — VBB lives on the harness")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
