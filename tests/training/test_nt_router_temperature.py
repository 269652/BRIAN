# -*- coding: utf-8 -*-
"""TDD contract — Item 2: NT → ThalamicRouter temperature coupling.

Goal
====
Norepinephrine (NE) is the brain's *gain / uncertainty* channel. High NE
sharpens routing (one expert wins decisively); low NE diffuses it
(mixture stays soft). Map the standardised NE level to the softmax
temperature applied to the router logits BEFORE the simplex projection.

Formula (locked here, math-first)
=================================
With ``z_NE = 2 * (NE - 0.5)`` the unit-mean-centred NE signal in
``[-1, +1]`` (NE itself lives in ``[0, 1]`` from the homeostat sigmoid):

    T(NE) = 1 / clamp(1 + k_NE · z_NE, 0.1, 10.0)

    logits_T = logits / T(NE)
             = logits · clamp(1 + k_NE · z_NE, 0.1, 10.0)

``k_NE`` is the new ``router_temp_nt_gain`` knob in
``multi_cortex { ... }``. Default 0.0 → identity (back-compat).

Why this shape:
* NE at baseline (≈ 0.20 per ``DrivenNTSystem`` defaults) gives
  ``z_NE ≈ -0.6``; with ``k_NE=0.5`` the multiplier is
  ``1 - 0.3 = 0.7`` → routing slightly softened (mixture mode).
* NE at 0.8 (stress regime) gives ``z_NE = 0.6``; with ``k_NE=0.5`` the
  multiplier is ``1.3`` → routing sharper (winner-take-most).
* The clamp guarantees temperature ∈ ``[0.1, 10.0]`` so no NaN.
* Multiplying logits is mathematically identical to dividing by T but
  saves one floating-point divide per token per cortex.

Contract pinned in this file
============================
A. ``ThalamicRouter`` accepts a ``router_temp_nt_gain`` ctor arg
   (default 0.0) and a runtime setter ``set_nt_levels(dict)``.
B. With ``router_temp_nt_gain = 0`` and/or no NT levels set,
   forward is bit-identical to the current behaviour
   (back-compat — every existing test must still pass).
C. With ``router_temp_nt_gain > 0`` and NE pinned high, the routing
   distribution's max increases (sharper).
D. With ``router_temp_nt_gain > 0`` and NE pinned low, the routing
   distribution's max decreases (softer).
E. The DSL ``multi_cortex { router_temp_nt_gain: 0.5 }`` parses and
   round-trips into ``MultiCortexConfig.router_temp_nt_gain``.
F. The temperature multiplier is clamped to ``[0.1, 10.0]``.
"""
from __future__ import annotations

import math
import pytest
import torch

from neuroslm.cortex import (
    DomainLexicon,
    ThalamicRouter,
)


# Tiny shared fixture: a router over a 32-token vocab and 3 domains.
def _make_router(temp_gain: float = 0.0, bias: float = 0.0):
    vocab = 32
    domains = ["a", "b", "c"]
    # Equal-size lexical chunks per domain.
    lex = DomainLexicon(domain_token_map={
        d: list(range(i * 10, (i + 1) * 10))
        for i, d in enumerate(domains)
    })
    return ThalamicRouter(
        vocab_size=vocab,
        d_model=16,
        domains=domains,
        lexicon=lex,
        lexical_bias_weight=bias,
        bema_tau=0.0,
        router_temp_nt_gain=temp_gain,
    )


def _routing_distribution(router, ids):
    """Return ``(B, T, N)`` softmax probabilities from the router."""
    return router(ids)


class TestRouterAcceptsTempGain:
    """A) Constructor and runtime setter exist."""

    def test_ctor_accepts_router_temp_nt_gain(self):
        r = _make_router(temp_gain=0.5)
        assert getattr(r, "router_temp_nt_gain") == pytest.approx(0.5)

    def test_default_temp_gain_is_zero(self):
        r = _make_router()
        assert r.router_temp_nt_gain == 0.0

    def test_ctor_rejects_negative_temp_gain(self):
        # A negative gain would flip the NE polarity and silently
        # make routing softer when NE is high — fail loudly.
        with pytest.raises(ValueError):
            _make_router(temp_gain=-0.1)

    def test_set_nt_levels_runtime_setter_exists(self):
        r = _make_router(temp_gain=0.5)
        assert callable(getattr(r, "set_nt_levels", None))
        r.set_nt_levels({"NE": 0.8})
        # No exception means the setter accepted the dict.

    def test_set_nt_levels_ignores_unknown_keys(self):
        r = _make_router(temp_gain=0.5)
        # Passing extra channels (DA, ACh, ...) must NOT raise — the
        # router only consumes NE; everything else is ignored so the
        # caller can pass the whole NTSystem.levels() dict.
        r.set_nt_levels({"NE": 0.5, "DA": 0.3, "GABA": 0.1})


class TestBackCompat:
    """B) gain=0 OR no NT set → bit-identical to current behaviour."""

    def test_zero_gain_is_bit_identical_to_pre_change(self):
        torch.manual_seed(0)
        r = _make_router(temp_gain=0.0)
        ids = torch.randint(0, 32, (2, 8))
        out_no_nt = r(ids).clone()
        # Setting NT to extreme values must not move the output when
        # the gain is 0.
        r.set_nt_levels({"NE": 1.0})
        out_with_nt = r(ids).clone()
        assert torch.allclose(out_no_nt, out_with_nt), \
            "gain=0 must make NT level a no-op"

    def test_gain_positive_no_nt_set_is_identity(self):
        """If the harness never calls set_nt_levels, the router defaults
        to NE=0.5 (the centre of the sigmoid range), giving z_NE=0 and
        a multiplier of exactly 1.0 — i.e. identity."""
        torch.manual_seed(0)
        r_off = _make_router(temp_gain=0.0)
        r_on = _make_router(temp_gain=0.5)
        ids = torch.randint(0, 32, (2, 8))
        out_off = r_off(ids).clone()
        out_on = r_on(ids).clone()
        # Different router_temp_nt_gain → different layer instances → different
        # router_embed init; we have to compare the *temperature path* by
        # setting both routers to the same params.
        r_on.router_embed.weight.data.copy_(r_off.router_embed.weight.data)
        r_on.learnable_logits.weight.data.copy_(r_off.learnable_logits.weight.data)
        # No NT set → centre point → multiplier 1 → bit-identical.
        out_on2 = r_on(ids)
        assert torch.allclose(out_off, out_on2), (
            "with no NT levels set, gain>0 must still be identity (centre point)"
        )


class TestTemperatureCoupling:
    """C, D, F) Sharpness moves the right way with NE; clamp holds."""

    def _max_weight(self, router, ids) -> float:
        """Return the mean of the per-token max weight."""
        return router(ids).amax(dim=-1).mean().item()

    def test_high_ne_sharpens_routing(self):
        """C: high NE → larger max weight (sharper distribution)."""
        torch.manual_seed(1)
        r = _make_router(temp_gain=0.5, bias=2.0)  # bias creates non-uniform logits
        ids = torch.arange(30, dtype=torch.long).unsqueeze(0)  # (1, 30)

        r.set_nt_levels({"NE": 0.5})  # centre
        sharp_centre = self._max_weight(r, ids)

        r.set_nt_levels({"NE": 0.95})  # high stress
        sharp_high = self._max_weight(r, ids)

        assert sharp_high > sharp_centre + 1e-4, (
            f"high NE must sharpen routing (max weight): "
            f"centre={sharp_centre:.4f} high={sharp_high:.4f}"
        )

    def test_low_ne_softens_routing(self):
        """D: low NE → smaller max weight (softer distribution)."""
        torch.manual_seed(1)
        r = _make_router(temp_gain=0.5, bias=2.0)
        ids = torch.arange(30, dtype=torch.long).unsqueeze(0)

        r.set_nt_levels({"NE": 0.5})
        sharp_centre = self._max_weight(r, ids)

        r.set_nt_levels({"NE": 0.05})  # low arousal
        sharp_low = self._max_weight(r, ids)

        assert sharp_low < sharp_centre - 1e-4, (
            f"low NE must soften routing (max weight): "
            f"centre={sharp_centre:.4f} low={sharp_low:.4f}"
        )

    def test_temperature_multiplier_is_clamped(self):
        """F: multiplier ∈ [0.1, 10.0] — no NaN / div-by-zero. Use
        an extreme gain to push the raw multiplier outside the bounds
        and confirm the output stays finite."""
        r = _make_router(temp_gain=100.0, bias=2.0)  # absurd gain
        ids = torch.arange(30, dtype=torch.long).unsqueeze(0)

        # Both extremes must produce finite, simplex-valid output.
        for ne in (0.0, 1.0):
            r.set_nt_levels({"NE": ne})
            out = r(ids)
            assert torch.isfinite(out).all(), f"output NaN/Inf at NE={ne}"
            sums = out.sum(dim=-1)
            assert torch.allclose(sums, torch.ones_like(sums), atol=1e-5), (
                f"router output not on simplex at NE={ne}, sums={sums}"
            )


class TestDSLParse:
    """E) DSL block ``router_temp_nt_gain: 0.5`` round-trips."""

    def test_parser_reads_router_temp_nt_gain(self):
        from neuroslm.dsl.training_config import _parse_multi_cortex
        mc = _parse_multi_cortex("""{
            enabled: true,
            router_temp_nt_gain: 0.5,
            experts: [
                { id: "gpt2", domain: "general", freeze: true }
            ],
            trunk_tokenizer: "gpt2"
        }""")
        assert hasattr(mc, "router_temp_nt_gain"), (
            "MultiCortexConfig must expose `router_temp_nt_gain`"
        )
        assert mc.router_temp_nt_gain == pytest.approx(0.5)

    def test_parser_default_is_zero(self):
        from neuroslm.dsl.training_config import _parse_multi_cortex
        mc = _parse_multi_cortex("""{
            enabled: true,
            experts: [
                { id: "gpt2", domain: "general" }
            ],
            trunk_tokenizer: "gpt2"
        }""")
        assert mc.router_temp_nt_gain == 0.0
