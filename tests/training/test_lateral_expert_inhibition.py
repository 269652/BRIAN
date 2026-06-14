# -*- coding: utf-8 -*-
"""TDD contract — Item 4: lateral expert inhibition (Mexican-hat / WTA).

Goal
====
Force the experts to *specialise* by introducing competitive inhibition
between their routing weights. Currently the softmax router produces
weights that sum to 1, but two experts can comfortably share 0.45/0.45
forever — there's no pressure to actually pick one. Lateral inhibition
adds the brain's classic ``Mexican-hat`` pattern: when one expert fires
strongly, it suppresses the others; the GABA NT channel sets the
overall inhibition strength.

Formula (locked here, math-first)
=================================
Given router weights ``w ∈ Δ^N`` per token (output of ``ThalamicRouter``):

    confidence_i = w_i                           # already in [0, 1]
    rival_mass_i = Σ_{j≠i} confidence_j           # mass on other experts
    suppressed_i = w_i / (1 + κ · rival_mass_i)  # divisive inhibition
    w'           = suppressed / Σ suppressed     # renormalise to simplex

    κ = κ_base · clamp(GABA_level / 0.5, 0.0, 4.0)

* ``κ_base`` is the new ``lateral_inhibition_kappa`` knob in
  ``multi_cortex``. Default 0.0 → identity (back-compat).
* GABA-modulated: at GABA baseline (≈ 0.15) the multiplier is 0.3,
  so even with κ_base = 0.5 the effective κ is mild (0.15). At GABA
  saturation (0.5+) the inhibition reaches κ_base × (≥1).
* Divisive (not subtractive) so we can never go negative; the
  renormalise puts us back on the simplex regardless of κ.

Why divisive:
* Real cortical inhibition is subtractive shunting at the membrane,
  not "subtract a constant." Divisive normalisation
  (Carandini & Heeger 2012, "Normalization as a canonical neural
  computation") is the standard model.
* Subtractive would force a `max(0, ·)` floor which has zero gradient
  for suppressed experts — pathological for training.

Contract pinned in this file
============================
A. New ``LateralInhibition`` module accepts ``kappa_base`` and a runtime
   ``set_nt_levels`` setter, takes ``(B, T, N)`` weights and returns
   ``(B, T, N)`` weights still on the simplex.
B. With ``kappa_base = 0`` OR ``GABA = 0`` the output is bit-identical
   to the input.
C. With ``kappa_base > 0`` and ``GABA > 0``: the **gini coefficient**
   (a sharpness measure) of the output is strictly higher than the
   input — peaks get peakier, troughs get troughier.
D. The output is always on the simplex (sums to 1 within float tol).
E. The output is always non-negative.
F. Gradient flows through the inhibition (no detach / no hard max).
G. ``LMExpertEnsemble`` accepts a ``lateral_inhibition`` arg and applies
   it between the router and the per-expert weighted sum.
H. The DSL ``multi_cortex { lateral_inhibition_kappa: 0.5 }`` parses.
"""
from __future__ import annotations

import pytest
import torch


# ──────────────────────────────────────────────────────────────────────
# A) Module exists with the right ctor + setter
# ──────────────────────────────────────────────────────────────────────


class TestModuleSurface:

    def test_lateral_inhibition_module_importable(self):
        from neuroslm.cortex import LateralInhibition
        assert LateralInhibition is not None

    def test_ctor_accepts_kappa_base(self):
        from neuroslm.cortex import LateralInhibition
        li = LateralInhibition(kappa_base=0.5)
        assert li.kappa_base == pytest.approx(0.5)

    def test_default_kappa_is_zero(self):
        from neuroslm.cortex import LateralInhibition
        li = LateralInhibition()
        assert li.kappa_base == 0.0

    def test_ctor_rejects_negative_kappa(self):
        from neuroslm.cortex import LateralInhibition
        with pytest.raises(ValueError):
            LateralInhibition(kappa_base=-0.1)

    def test_set_nt_levels_exists_and_ignores_unknown(self):
        from neuroslm.cortex import LateralInhibition
        li = LateralInhibition(kappa_base=0.5)
        li.set_nt_levels({"GABA": 0.3, "DA": 0.2, "5HT": 0.5})
        # No exception means the setter accepted the dict.


# ──────────────────────────────────────────────────────────────────────
# B) Back-compat: kappa=0 OR GABA=0 → identity
# ──────────────────────────────────────────────────────────────────────


class TestBackCompat:

    def _weights(self, n: int = 3):
        torch.manual_seed(0)
        # A non-uniform softmax over (B=2, T=4, N).
        logits = torch.randn(2, 4, n)
        return torch.softmax(logits, dim=-1)

    def test_kappa_zero_is_identity(self):
        from neuroslm.cortex import LateralInhibition
        li = LateralInhibition(kappa_base=0.0)
        w = self._weights()
        out = li(w)
        assert torch.allclose(out, w, atol=1e-7)

    def test_gaba_zero_is_identity(self):
        from neuroslm.cortex import LateralInhibition
        li = LateralInhibition(kappa_base=1.0)
        li.set_nt_levels({"GABA": 0.0})
        w = self._weights()
        out = li(w)
        assert torch.allclose(out, w, atol=1e-7), (
            "GABA=0 must zero out the inhibition multiplier (no effect)"
        )

    def test_no_nt_set_defaults_to_baseline(self):
        """Default GABA = 0.15 (DrivenNTSystem baseline) gives mild
        inhibition; output should not be identity but must still be on
        the simplex."""
        from neuroslm.cortex import LateralInhibition
        li = LateralInhibition(kappa_base=0.5)
        w = self._weights()
        out = li(w)
        sums = out.sum(dim=-1)
        assert torch.allclose(sums, torch.ones_like(sums), atol=1e-5)


# ──────────────────────────────────────────────────────────────────────
# C, D, E) Sharpening + simplex + non-negativity
# ──────────────────────────────────────────────────────────────────────


def _gini(p: torch.Tensor) -> torch.Tensor:
    """Per-row Gini coefficient — higher = sharper distribution.

    Standard discrete Gini for non-negative values:

        G = (Σ (2i - N - 1) · sorted_p_i) / (N · Σ p_i)        (i = 1..N)

    For probability rows (Σ p = 1), the denominator collapses to N.
    Sanity checks:
      * uniform  [0.25]·4 → G = 0
      * peaked   [0,0,0,1] → G = 0.75
      * mid      [0.05, 0.15, 0.3, 0.5] → G = 0.375
    """
    p = p.flatten(0, -2)                                       # (M, N)
    M, N = p.shape
    sorted_p, _ = torch.sort(p, dim=-1)
    idx = torch.arange(1, N + 1, dtype=p.dtype, device=p.device)
    coeffs = 2.0 * idx - (N + 1.0)                             # e.g. [-3,-1,1,3]
    row_sum = sorted_p.sum(dim=-1).clamp(min=1e-9)
    return (coeffs.unsqueeze(0) * sorted_p).sum(dim=-1) / (N * row_sum)


class TestInhibitionSharpens:

    def _weights(self):
        torch.manual_seed(1)
        logits = torch.randn(2, 4, 4)
        return torch.softmax(logits, dim=-1)

    def test_high_gaba_increases_sharpness(self):
        from neuroslm.cortex import LateralInhibition
        li = LateralInhibition(kappa_base=1.0)
        w = self._weights()
        li.set_nt_levels({"GABA": 0.8})       # high inhibition
        out = li(w)
        # Sanity: not identity.
        assert not torch.allclose(out, w, atol=1e-3)
        # Sharpness should rise on average.
        gini_in = _gini(w).mean().item()
        gini_out = _gini(out).mean().item()
        assert gini_out > gini_in + 1e-4, (
            f"high GABA must sharpen routing: gini in={gini_in:.4f} "
            f"out={gini_out:.4f}"
        )

    def test_output_on_simplex_for_all_inputs(self):
        from neuroslm.cortex import LateralInhibition
        li = LateralInhibition(kappa_base=2.0)
        torch.manual_seed(2)
        for gaba in (0.0, 0.1, 0.5, 0.9):
            li.set_nt_levels({"GABA": gaba})
            for _ in range(5):
                w = torch.softmax(torch.randn(3, 5, 4), dim=-1)
                out = li(w)
                sums = out.sum(dim=-1)
                assert torch.allclose(
                    sums, torch.ones_like(sums), atol=1e-5
                ), f"sum off simplex at gaba={gaba}: {sums}"

    def test_output_nonneg(self):
        from neuroslm.cortex import LateralInhibition
        li = LateralInhibition(kappa_base=10.0)  # extreme inhibition
        torch.manual_seed(3)
        li.set_nt_levels({"GABA": 1.0})
        for _ in range(5):
            w = torch.softmax(torch.randn(3, 5, 4), dim=-1)
            out = li(w)
            assert (out >= -1e-7).all(), f"output went negative: {out.min()}"


# ──────────────────────────────────────────────────────────────────────
# F) Gradient flows (no detach, no zero-grad block)
# ──────────────────────────────────────────────────────────────────────


class TestGradientFlow:

    def test_grad_flows_through_inhibition(self):
        from neuroslm.cortex import LateralInhibition
        torch.manual_seed(4)
        logits = torch.randn(2, 3, 4, requires_grad=True)
        w = torch.softmax(logits, dim=-1)

        li = LateralInhibition(kappa_base=0.5)
        li.set_nt_levels({"GABA": 0.5})
        out = li(w)
        loss = out.sum()
        loss.backward()
        assert logits.grad is not None
        assert torch.isfinite(logits.grad).all()
        # The gradient should not be the trivial all-zero of a detached path.
        assert logits.grad.abs().sum().item() > 0.0


# ──────────────────────────────────────────────────────────────────────
# G) LMExpertEnsemble wires it in
# ──────────────────────────────────────────────────────────────────────


class TestEnsembleWires:

    def test_ensemble_accepts_lateral_inhibition_arg(self):
        """LMExpertEnsemble must accept a `lateral_inhibition` kwarg
        and apply it between the router output and the weighted sum."""
        import inspect
        from neuroslm.experts import LMExpertEnsemble
        sig = inspect.signature(LMExpertEnsemble.__init__)
        assert "lateral_inhibition" in sig.parameters, (
            "LMExpertEnsemble.__init__ must take a lateral_inhibition kwarg"
        )

    def test_build_lm_expert_ensemble_accepts_kappa(self):
        """The factory should accept and wire `lateral_inhibition_kappa`."""
        import inspect
        from neuroslm.experts import build_lm_expert_ensemble
        sig = inspect.signature(build_lm_expert_ensemble)
        assert "lateral_inhibition_kappa" in sig.parameters


# ──────────────────────────────────────────────────────────────────────
# H) DSL parse round-trip
# ──────────────────────────────────────────────────────────────────────


class TestDSLParse:

    def test_parser_reads_kappa(self):
        from neuroslm.dsl.training_config import _parse_multi_cortex
        mc = _parse_multi_cortex("""{
            enabled: true,
            lateral_inhibition_kappa: 0.5,
            experts: [
                { id: "gpt2", domain: "general" }
            ],
            trunk_tokenizer: "gpt2"
        }""")
        assert hasattr(mc, "lateral_inhibition_kappa")
        assert mc.lateral_inhibition_kappa == pytest.approx(0.5)

    def test_parser_default_zero(self):
        from neuroslm.dsl.training_config import _parse_multi_cortex
        mc = _parse_multi_cortex("""{
            enabled: true,
            experts: [
                { id: "gpt2", domain: "general" }
            ],
            trunk_tokenizer: "gpt2"
        }""")
        assert mc.lateral_inhibition_kappa == 0.0
