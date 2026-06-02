# -*- coding: utf-8 -*-
"""Tests for the new `regularization { ... }` block in arch.neuro.

Five interventions, declared math-first in the `regularization` block
and parsed into structured sub-configs that the harness consumes:

    regularization {
        dar      { enabled: true,  lambda: 1.0, hidden: 64, grl_alpha: 0.1 }
        pcc      { enabled: true,  k: 8, n_negatives: 64, tau: 0.07, layers: [4,5,6,7] }
        isotropy { enabled: true,  weight: 0.01, buffer: 4096 }
        cmd      { enabled: true,  weight: 0.05, divergence: "jsd",
                   heads: ["lm", "narrative"] }
        adaptive_mixture { enabled: true, target_entropy: 4.5,
                           probe_interval: 100, gamma: 2.0,
                           min_ratio: 0.10, max_ratio: 0.80 }
    }

This file is TDD-first: each test asserts the parser produces the right
structured config. Harness wiring lands in PR2.
"""
import pytest

from neuroslm.dsl.regularization import (
    RegularizationConfig,
    DARConfig,
    PCCConfig,
    IsotropyConfig,
    CMDConfig,
    AdaptiveMixtureConfig,
    parse_regularization_block,
)


# ── Defaults: empty block → every intervention disabled ───────────────

class TestDefaults:
    def test_empty_block_all_disabled(self):
        cfg = parse_regularization_block("")
        assert cfg.dar.enabled is False
        assert cfg.pcc.enabled is False
        assert cfg.isotropy.enabled is False
        assert cfg.cmd.enabled is False
        assert cfg.adaptive_mixture.enabled is False

    def test_missing_subblock_uses_defaults(self):
        cfg = parse_regularization_block(
            'isotropy: { enabled: true, weight: 0.02 }'
        )
        assert cfg.isotropy.enabled is True
        # Unspecified interventions retain disabled defaults
        assert cfg.dar.enabled is False
        assert cfg.pcc.enabled is False
        assert cfg.cmd.enabled is False
        assert cfg.adaptive_mixture.enabled is False


# ── Intervention A: Distributional Adversarial Reweighting ────────────

class TestDAR:
    def test_basic(self):
        cfg = parse_regularization_block(
            'dar: { enabled: true, lambda: 1.5, hidden: 128, grl_alpha: 0.2 }'
        )
        assert cfg.dar.enabled is True
        assert cfg.dar.lam == 1.5
        assert cfg.dar.hidden == 128
        assert cfg.dar.grl_alpha == 0.2

    def test_defaults_when_enabled(self):
        cfg = parse_regularization_block('dar: { enabled: true }')
        assert cfg.dar.enabled is True
        assert cfg.dar.lam == 1.0
        assert cfg.dar.hidden == 64
        assert cfg.dar.grl_alpha == 0.1


# ── Intervention B: Predictive Contrastive Coding (replaces PCT) ──────

class TestPCC:
    def test_basic(self):
        cfg = parse_regularization_block(
            'pcc: { enabled: true, k: 8, n_negatives: 64, tau: 0.07, '
            'layers: [4, 5, 6, 7] }'
        )
        assert cfg.pcc.enabled is True
        assert cfg.pcc.k == 8
        assert cfg.pcc.n_negatives == 64
        assert cfg.pcc.tau == pytest.approx(0.07)
        assert cfg.pcc.layers == [4, 5, 6, 7]

    def test_defaults(self):
        cfg = parse_regularization_block('pcc: { enabled: true }')
        assert cfg.pcc.k == 4
        assert cfg.pcc.n_negatives == 64
        assert cfg.pcc.tau == pytest.approx(0.1)
        assert cfg.pcc.layers == []  # empty = all layers


# ── Intervention C: Isotropy whitening ────────────────────────────────

class TestIsotropy:
    def test_basic(self):
        cfg = parse_regularization_block(
            'isotropy: { enabled: true, weight: 0.01, buffer: 4096 }'
        )
        assert cfg.isotropy.enabled is True
        assert cfg.isotropy.weight == pytest.approx(0.01)
        assert cfg.isotropy.buffer == 4096

    def test_distance_default_frobenius(self):
        cfg = parse_regularization_block('isotropy: { enabled: true }')
        assert cfg.isotropy.distance == "frobenius"


# ── Intervention D: Cross-Module Disagreement ────────────────────────

class TestCMD:
    def test_basic(self):
        cfg = parse_regularization_block(
            'cmd: { enabled: true, weight: 0.05, divergence: "jsd", '
            'heads: ["lm", "narrative"] }'
        )
        assert cfg.cmd.enabled is True
        assert cfg.cmd.weight == pytest.approx(0.05)
        assert cfg.cmd.divergence == "jsd"
        assert cfg.cmd.heads == ["lm", "narrative"]

    def test_rejects_unknown_divergence(self):
        with pytest.raises(ValueError, match="divergence"):
            parse_regularization_block(
                'cmd: { enabled: true, divergence: "wasserstein_42" }'
            )


# ── Intervention E: Adaptive mixture controller ───────────────────────

class TestAdaptiveMixture:
    def test_basic(self):
        cfg = parse_regularization_block(
            'adaptive_mixture: { enabled: true, target_entropy: 4.5, '
            'probe_interval: 100, gamma: 2.0, min_ratio: 0.1, max_ratio: 0.8 }'
        )
        am = cfg.adaptive_mixture
        assert am.enabled is True
        assert am.target_entropy == pytest.approx(4.5)
        assert am.probe_interval == 100
        assert am.gamma == pytest.approx(2.0)
        assert am.min_ratio == pytest.approx(0.1)
        assert am.max_ratio == pytest.approx(0.8)

    def test_min_below_max(self):
        with pytest.raises(ValueError, match="min_ratio"):
            parse_regularization_block(
                'adaptive_mixture: { enabled: true, min_ratio: 0.9, max_ratio: 0.5 }'
            )


# ── End-to-end: full block ────────────────────────────────────────────

class TestFullBlock:
    def test_all_five_interventions_together(self):
        body = '''
            dar:      { enabled: true, lambda: 1.0 }
            pcc:      { enabled: true, k: 4 }
            isotropy: { enabled: true, weight: 0.01 }
            cmd:      { enabled: true, weight: 0.05, divergence: "jsd" }
            adaptive_mixture: { enabled: true, target_entropy: 4.5 }
        '''
        cfg = parse_regularization_block(body)
        assert cfg.dar.enabled
        assert cfg.pcc.enabled
        assert cfg.isotropy.enabled
        assert cfg.cmd.enabled
        assert cfg.adaptive_mixture.enabled

    def test_any_enabled(self):
        cfg = parse_regularization_block('isotropy: { enabled: true }')
        assert cfg.any_enabled() is True

        cfg2 = parse_regularization_block('')
        assert cfg2.any_enabled() is False
