# -*- coding: utf-8 -*-
"""RED tests for GIF-6: Cosine LM Head.

Mathematical specification:
    z_i = τ · (h̄ · w̄_i)    where h̄ = h/‖h‖, w̄_i = w_i/‖w_i‖

Eliminates magnitude as a degree of freedom in the final projection.
All predictions must come from angular proximity between the hidden
state and the token embedding — a geometric property that generalises
across domains.

τ is a learnable scalar temperature (init √d_model ≈ 22.6 for d=512).
Logit magnitude is bounded by τ, so the model cannot encode domain-
specific confidence in the norms of h or W_head.

DSL equation: lib/gif.neuro :: gif_cosine_lm_head
Python atom:  neuroslm/dsl/nn_ops.py :: cosine_lm_head
Config:       training { cosine_head: true }
"""
import math
import pytest
import torch
import torch.nn.functional as F

from neuroslm.dsl import nn_ops


# ══════════════════════════════════════════════════════════════════════
# 1. nn_ops.cosine_lm_head atom
# ══════════════════════════════════════════════════════════════════════

class TestCosineHeadOp:
    """Pin the nn_ops atom: cosine_lm_head(h, weight, temperature)."""

    def test_output_shape(self):
        """Output shape matches standard linear head: (B, T, V)."""
        B, T, D, V = 2, 16, 64, 100
        h = torch.randn(B, T, D)
        w = torch.randn(V, D)
        tau = torch.tensor(8.0)
        logits = nn_ops.cosine_lm_head(h, w, tau)
        assert logits.shape == (B, T, V)

    def test_logits_bounded_by_temperature(self):
        """All logit values must lie in [-τ, +τ]."""
        B, T, D, V = 4, 32, 128, 500
        h = torch.randn(B, T, D) * 100  # large norms shouldn't matter
        w = torch.randn(V, D) * 50
        tau = torch.tensor(15.0)
        logits = nn_ops.cosine_lm_head(h, w, tau)
        assert logits.max().item() <= tau.item() + 1e-5
        assert logits.min().item() >= -tau.item() - 1e-5

    def test_invariant_to_hidden_norm(self):
        """Scaling h should not change logits (norm-invariance)."""
        B, T, D, V = 2, 8, 32, 50
        h = torch.randn(B, T, D)
        w = torch.randn(V, D)
        tau = torch.tensor(10.0)
        logits_1 = nn_ops.cosine_lm_head(h, w, tau)
        logits_2 = nn_ops.cosine_lm_head(h * 7.3, w, tau)
        torch.testing.assert_close(logits_1, logits_2, atol=1e-5, rtol=1e-5)

    def test_invariant_to_weight_norm(self):
        """Scaling W should not change logits (norm-invariance)."""
        B, T, D, V = 2, 8, 32, 50
        h = torch.randn(B, T, D)
        w = torch.randn(V, D)
        tau = torch.tensor(10.0)
        logits_1 = nn_ops.cosine_lm_head(h, w, tau)
        logits_2 = nn_ops.cosine_lm_head(h, w * 0.01, tau)
        torch.testing.assert_close(logits_1, logits_2, atol=1e-5, rtol=1e-5)

    def test_temperature_scales_logits(self):
        """Doubling τ doubles all logit values."""
        B, T, D, V = 2, 8, 32, 50
        h = torch.randn(B, T, D)
        w = torch.randn(V, D)
        logits_1 = nn_ops.cosine_lm_head(h, w, torch.tensor(5.0))
        logits_2 = nn_ops.cosine_lm_head(h, w, torch.tensor(10.0))
        torch.testing.assert_close(logits_2, logits_1 * 2, atol=1e-4, rtol=1e-4)

    def test_gradient_flows_through_all_inputs(self):
        """Gradients must flow through h, W, and τ."""
        B, T, D, V = 2, 8, 32, 50
        h = torch.randn(B, T, D, requires_grad=True)
        w = torch.randn(V, D, requires_grad=True)
        tau = torch.tensor(10.0, requires_grad=True)
        logits = nn_ops.cosine_lm_head(h, w, tau)
        logits.sum().backward()
        assert h.grad is not None and h.grad.abs().sum() > 0
        assert w.grad is not None and w.grad.abs().sum() > 0
        assert tau.grad is not None and tau.grad.abs().item() > 0

    def test_cosine_similarity_semantics(self):
        """When h and w[i] are aligned, logit should be τ (max)."""
        D, V = 32, 3
        h = torch.zeros(1, 1, D)
        h[0, 0, 0] = 1.0  # unit vector along dim 0
        w = torch.zeros(V, D)
        w[0, 0] = 1.0   # aligned with h → cos = 1
        w[1, 1] = 1.0   # orthogonal → cos = 0
        w[2, 0] = -1.0  # anti-aligned → cos = -1
        tau = torch.tensor(20.0)
        logits = nn_ops.cosine_lm_head(h, w, tau)
        assert logits[0, 0, 0].item() == pytest.approx(20.0, abs=1e-4)
        assert logits[0, 0, 1].item() == pytest.approx(0.0, abs=1e-4)
        assert logits[0, 0, 2].item() == pytest.approx(-20.0, abs=1e-4)

    def test_numerical_stability_near_zero(self):
        """Near-zero hidden states should not produce NaN."""
        B, T, D, V = 2, 8, 32, 50
        h = torch.randn(B, T, D) * 1e-10
        w = torch.randn(V, D)
        tau = torch.tensor(10.0)
        logits = nn_ops.cosine_lm_head(h, w, tau)
        assert not torch.isnan(logits).any()
        assert not torch.isinf(logits).any()


# ══════════════════════════════════════════════════════════════════════
# 2. Config: training { cosine_head: true }
# ══════════════════════════════════════════════════════════════════════

class TestCosineHeadConfig:
    """Pin config parsing for cosine_head flag."""

    def test_default_is_false(self):
        from neuroslm.dsl.training_config import parse_training_config
        cfg = parse_training_config("")
        assert cfg.cosine_head is False

    def test_parses_true(self):
        from neuroslm.dsl.training_config import parse_training_config
        cfg = parse_training_config("cosine_head: true")
        assert cfg.cosine_head is True

    def test_parses_false(self):
        from neuroslm.dsl.training_config import parse_training_config
        cfg = parse_training_config("cosine_head: false")
        assert cfg.cosine_head is False

    def test_smollm_arch_has_cosine_head(self):
        """Pin: SmolLM arch declares cosine_head: true."""
        from pathlib import Path
        from neuroslm.dsl.training_config import load_training_config_from_arch
        arch_root = Path(__file__).resolve().parents[2] / "architectures" / "SmolLM"
        if not (arch_root / "arch.neuro").is_file():
            pytest.skip("SmolLM arch not present")
        cfg = load_training_config_from_arch(arch_root)
        assert cfg.cosine_head is True, (
            "SmolLM must have cosine_head: true — check arch.neuro"
        )


# ══════════════════════════════════════════════════════════════════════
# 3. Model integration: DSLLanguageModel
# ══════════════════════════════════════════════════════════════════════

class TestDSLLanguageModelCosineHead:
    """Pin that DSLLanguageModel uses cosine head when configured."""

    def test_has_head_temperature_param(self):
        from neuroslm.dsl.nn_lang import build_language_model
        lm = build_language_model(
            vocab=100, d_model=64, depth=2, n_heads=4, max_ctx=32,
            cosine_head=True,
        )
        assert hasattr(lm, "head_temperature")
        assert lm.head_temperature.requires_grad
        # Init should be √d_model
        expected = math.sqrt(64)
        assert lm.head_temperature.item() == pytest.approx(expected, rel=1e-4)

    def test_no_temperature_when_disabled(self):
        from neuroslm.dsl.nn_lang import build_language_model
        lm = build_language_model(
            vocab=100, d_model=64, depth=2, n_heads=4, max_ctx=32,
            cosine_head=False,
        )
        assert not hasattr(lm, "head_temperature") or lm.head_temperature is None

    def test_logits_bounded_by_temperature(self):
        from neuroslm.dsl.nn_lang import build_language_model
        torch.manual_seed(42)
        lm = build_language_model(
            vocab=100, d_model=64, depth=2, n_heads=4, max_ctx=32,
            cosine_head=True,
        )
        lm.eval()
        ids = torch.randint(0, 100, (2, 16))
        with torch.no_grad():
            logits = lm(ids)
        tau = lm.head_temperature.item()
        assert logits.max().item() <= tau + 0.1
        assert logits.min().item() >= -tau - 0.1

    def test_gradient_reaches_temperature(self):
        from neuroslm.dsl.nn_lang import build_language_model
        torch.manual_seed(42)
        lm = build_language_model(
            vocab=100, d_model=64, depth=2, n_heads=4, max_ctx=32,
            cosine_head=True,
        )
        ids = torch.randint(0, 100, (2, 16))
        logits = lm(ids)
        logits.sum().backward()
        assert lm.head_temperature.grad is not None
        assert lm.head_temperature.grad.abs().item() > 0

    def test_default_cosine_head_is_false(self):
        """build_language_model without cosine_head uses linear head."""
        from neuroslm.dsl.nn_lang import build_language_model
        torch.manual_seed(42)
        lm = build_language_model(
            vocab=100, d_model=64, depth=2, n_heads=4, max_ctx=32,
        )
        # Should have lm_head but no temperature
        assert hasattr(lm, "lm_head")
        assert not hasattr(lm, "head_temperature") or lm.head_temperature is None


# ══════════════════════════════════════════════════════════════════════
# 4. Model integration: DSLLanguageCortex
# ══════════════════════════════════════════════════════════════════════

class TestDSLLanguageCortexCosineHead:
    """Pin that DSLLanguageCortex uses cosine head when configured."""

    def test_cortex_has_temperature_param(self):
        from neuroslm.dsl.nn_lang import build_dsl_language_cortex
        lm = build_dsl_language_cortex(
            vocab=100, d_model=64, depth=2, n_heads=4, max_ctx=32,
            cosine_head=True,
        )
        assert hasattr(lm, "head_temperature")
        assert lm.head_temperature.requires_grad

    def test_cortex_logits_bounded(self):
        from neuroslm.dsl.nn_lang import build_dsl_language_cortex
        torch.manual_seed(42)
        lm = build_dsl_language_cortex(
            vocab=100, d_model=64, depth=2, n_heads=4, max_ctx=32,
            cosine_head=True,
        )
        lm.eval()
        ids = torch.randint(0, 100, (2, 16))
        with torch.no_grad():
            logits = lm(ids)
        tau = lm.head_temperature.item()
        assert logits.max().item() <= tau + 0.1
        assert logits.min().item() >= -tau - 0.1

    def test_cortex_default_is_linear(self):
        from neuroslm.dsl.nn_lang import build_dsl_language_cortex
        lm = build_dsl_language_cortex(
            vocab=100, d_model=64, depth=2, n_heads=4, max_ctx=32,
        )
        assert not hasattr(lm, "head_temperature") or lm.head_temperature is None


# ══════════════════════════════════════════════════════════════════════
# 5. DSL equation presence
# ══════════════════════════════════════════════════════════════════════

class TestCosineHeadDSLEquation:
    """Pin that lib/gif.neuro exports the cosine head equation."""

    def test_gif_neuro_has_cosine_equation(self):
        from pathlib import Path
        gif_path = Path(__file__).resolve().parents[2] / "lib" / "gif.neuro"
        if not gif_path.is_file():
            pytest.skip("lib/gif.neuro not present")
        text = gif_path.read_text()
        assert "gif_cosine_lm_head" in text

    def test_arch_imports_cosine_equation(self):
        from pathlib import Path
        arch_path = (Path(__file__).resolve().parents[2]
                     / "architectures" / "SmolLM" / "arch.neuro")
        if not arch_path.is_file():
            pytest.skip("arch.neuro not present")
        text = arch_path.read_text(encoding="utf-8")
        assert "gif_cosine_lm_head" in text


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
