# -*- coding: utf-8 -*-
"""TDD contract — harness wiring: push NT levels to every consumer.

Items 2, 3, 4 added three NT-modulated knobs:

  * **Item 2** — ``ThalamicRouter.set_nt_levels({NE: ...})``
    (routing-softmax temperature)
  * **Item 3** — ``BRIANHarness._nt_levels_for_distill`` consumed by
    ``_distillation_lambda`` (5HT/DA-modulated distillation strength)
  * **Item 4** — ``LateralInhibition.set_nt_levels({GABA: ...})``
    (κ-gated Mexican-hat WTA inhibition)

…and three matching DSL knobs that activate them in ``arch.neuro``.

Without a wiring path that pushes the homeostat's live NT levels to
each of those consumers every step, the knobs do nothing — the
defaults stay forever at centre values (NE=0.5, 5HT=0.5, DA=0.5,
GABA=0.0) and the new mechanisms can't actually express their effect.

This file pins the wiring contract:

A. ``BRIANHarness.distribute_nt_levels(levels)`` exists and is a
   single seam that fans out the live NT dict to all consumers:
   - ``self._nt_levels_for_distill`` is updated.
   - If ``self.multi_cortex`` exposes ``set_nt_levels``, it's called.
B. Calling ``distribute_nt_levels(None)`` is a no-op (no exception)
   so callers can be permissive.
C. ``compute_loss(ids, targets, nt_levels=levels)`` auto-distributes
   when ``nt_levels`` is supplied, so the existing call site in
   ``train_step`` doesn't need a separate hook.
D. Polymorphic against ``MultiCortexEnsemble`` (legacy ensemble that
   has no ``set_nt_levels``): ``distribute_nt_levels`` MUST NOT raise.
"""
from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest


# ──────────────────────────────────────────────────────────────────────
# A) distribute_nt_levels exists on the harness class
# ──────────────────────────────────────────────────────────────────────


class TestDistributeMethod:

    def test_method_exists_on_class(self):
        from neuroslm.harness import BRIANHarness
        assert hasattr(BRIANHarness, "distribute_nt_levels"), (
            "BRIANHarness must expose a `distribute_nt_levels(levels)` "
            "method as the single seam for NT distribution"
        )

    def test_distributes_to_distill_attr(self):
        """Calling distribute_nt_levels stores the dict on
        `_nt_levels_for_distill` so the next `_distillation_lambda`
        call picks it up."""
        from neuroslm.harness import BRIANHarness

        # Reuse a stub harness — we only care about the method body.
        stub = SimpleNamespace(_nt_levels_for_distill={}, multi_cortex=None)
        BRIANHarness.distribute_nt_levels(stub, {"DA": 0.7, "5HT": 0.3})
        assert stub._nt_levels_for_distill == {"DA": 0.7, "5HT": 0.3}

    def test_distributes_to_multi_cortex_when_supported(self):
        from neuroslm.harness import BRIANHarness

        mc = MagicMock()
        mc.set_nt_levels = MagicMock()
        stub = SimpleNamespace(_nt_levels_for_distill={}, multi_cortex=mc)
        levels = {"NE": 0.8, "GABA": 0.4, "5HT": 0.5, "DA": 0.2}
        BRIANHarness.distribute_nt_levels(stub, levels)
        mc.set_nt_levels.assert_called_once_with(levels)

    def test_polymorphic_when_multi_cortex_lacks_set_nt_levels(self):
        """The legacy `MultiCortexEnsemble` doesn't have
        `set_nt_levels` — distribute must silently skip the push
        instead of raising AttributeError."""
        from neuroslm.harness import BRIANHarness

        legacy_mc = SimpleNamespace()       # no set_nt_levels attribute
        stub = SimpleNamespace(
            _nt_levels_for_distill={}, multi_cortex=legacy_mc)
        # No exception ⇒ pass.
        BRIANHarness.distribute_nt_levels(stub, {"NE": 0.5})
        assert stub._nt_levels_for_distill == {"NE": 0.5}

    def test_none_is_noop(self):
        from neuroslm.harness import BRIANHarness

        mc = MagicMock()
        mc.set_nt_levels = MagicMock()
        before = {"5HT": 0.5}
        stub = SimpleNamespace(_nt_levels_for_distill=dict(before),
                               multi_cortex=mc)
        BRIANHarness.distribute_nt_levels(stub, None)
        # State unchanged.
        assert stub._nt_levels_for_distill == before
        mc.set_nt_levels.assert_not_called()


# ──────────────────────────────────────────────────────────────────────
# B) Integration: compute_loss(..., nt_levels=...) auto-distributes
# ──────────────────────────────────────────────────────────────────────


class TestComputeLossAutoDistribute:
    """compute_loss is the harness-internal seam through which
    `nt_levels` already flows (train_step → compute_loss → forward).
    We pin that compute_loss calls distribute_nt_levels first thing,
    so adding `nt_levels` to a callsite is sufficient — no extra
    plumbing needed elsewhere."""

    def test_compute_loss_calls_distribute(self):
        """When compute_loss is called with `nt_levels=...`, the
        harness must invoke `self.distribute_nt_levels(nt_levels)`
        before running forward + computing the loss.

        We verify this by binding a recorder onto a small mutable
        stub instance (so `self.distribute_nt_levels` resolves to it)
        and using a sentinel exception to short-circuit the rest of
        compute_loss — no need to stand up a real harness.
        """
        from neuroslm.harness import BRIANHarness
        import torch

        recorded = {"calls": []}

        class _Sentinel(Exception):
            pass

        # A mutable mini-harness stub. We provide everything the
        # first dozen lines of compute_loss touch (the sentinel fires
        # well before the heavier sections).
        class _Stub:
            language_model = None       # short-circuits MAT mech-mult block
            multi_cortex = None
            _observer = None
            # `compute_loss` reads `self.training_config.genetics.enabled`
            # before the forward call. Provide a tiny stub that returns
            # False so the genetics pre-step short-circuits.
            training_config = SimpleNamespace(
                genetics=SimpleNamespace(enabled=False)
            )

            def __init__(self):
                self._nt_levels_for_distill = {}

            def distribute_nt_levels(self, levels):
                recorded["calls"].append(
                    dict(levels) if levels else None)

            def __call__(self, *_args, **_kwargs):
                raise _Sentinel()

        stub = _Stub()
        levels = {"NE": 0.7, "GABA": 0.6, "5HT": 0.3, "DA": 0.4}

        with pytest.raises(_Sentinel):
            BRIANHarness.compute_loss(
                stub,
                torch.zeros((1, 4), dtype=torch.long),
                torch.zeros((1, 4), dtype=torch.long),
                nt_levels=levels,
            )

        assert recorded["calls"], (
            "compute_loss must invoke distribute_nt_levels when "
            "nt_levels is supplied — got no call"
        )
        assert recorded["calls"][0] == levels
