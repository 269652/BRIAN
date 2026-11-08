# -*- coding: utf-8 -*-
"""Global workspace content store (architecture.md §14.13).

Percepts, recalled memories and selected thoughts compete for a few
slots by salience; items fade each tick and the weakest is displaced.
The surviving contents are what thought generation sees."""
from __future__ import annotations

import pytest

from neuroslm.cognition.workspace import GlobalWorkspace


def _ws(capacity=4, decay=0.5):
    return GlobalWorkspace(capacity=capacity, decay=decay)


class TestCompetition:
    def test_holds_only_the_most_salient_items(self):
        ws = _ws(capacity=4)
        for i, s in enumerate([0.9, 0.2, 0.7, 0.5, 0.8]):
            ws.offer(f"item {i}", "memory", [1.0], s)
        assert {it.text for it in ws.items()} == {
            "item 0", "item 2", "item 3", "item 4"}

    def test_a_weaker_newcomer_does_not_get_in(self):
        ws = _ws(capacity=2)
        ws.offer("strong", "percept", [1.0], 1.0)
        ws.offer("medium", "memory", [1.0], 0.6)
        ws.offer("weak", "thought", [1.0], 0.1)
        assert "weak" not in ws.texts()

    def test_equal_salience_displaces_the_older_item(self):
        ws = _ws(capacity=2)
        ws.offer("old", "memory", [1.0], 0.5)
        ws.offer("strong", "percept", [1.0], 1.0)
        ws.offer("new", "memory", [1.0], 0.5)
        assert ws.texts() == {"strong", "new"}

    def test_items_are_ordered_most_salient_last(self):
        ws = _ws()
        ws.offer("b", "memory", [1.0], 0.6)
        ws.offer("a", "percept", [1.0], 1.0)
        ws.offer("c", "thought", [1.0], 0.2)
        assert [it.text for it in ws.items()] == ["c", "b", "a"]


class TestFadingAndRefresh:
    def test_decay_scales_every_salience(self):
        ws = _ws(decay=0.5)
        ws.offer("x", "memory", [1.0], 0.8)
        ws.decay()
        assert ws.items()[0].salience == pytest.approx(0.4)

    def test_reoffering_refreshes_without_duplicating(self):
        ws = _ws(decay=0.5)
        ws.offer("x", "memory", [1.0], 0.8)
        ws.decay()
        ws.offer("x", "memory", [1.0], 0.6)
        assert len(ws.items()) == 1
        assert ws.items()[0].salience == pytest.approx(0.6)

    def test_a_weaker_reoffer_does_not_lower_salience(self):
        ws = _ws()
        ws.offer("x", "thought", [1.0], 0.9)
        ws.offer("x", "thought", [1.0], 0.0)
        assert ws.items()[0].salience == pytest.approx(0.9)

    def test_drop_removes_one_source(self):
        ws = _ws()
        ws.offer("p", "percept", [1.0], 1.0)
        ws.offer("t", "thought", [1.0], 0.9)
        ws.drop("thought")
        assert ws.texts() == {"p"}


class TestIgnitionSustains:
    """Ignited contents do not fade: ``sustain`` undoes the tick's
    decay for every item held before it (H69 (e): only the new thought
    was sustained, so ignition crowded the percepts out)."""

    def test_sustain_undoes_this_ticks_fading(self):
        ws = _ws(decay=0.5)
        ws.offer("x", "sight", [1.0], 0.8)
        ws.decay()
        ws.sustain()
        assert ws.items()[0].salience == pytest.approx(0.8)

    def test_newcomers_keep_the_salience_they_were_offered_at(self):
        ws = _ws(decay=0.5)
        ws.decay()
        ws.offer("y", "thought", [1.0], 0.3)
        ws.sustain()
        assert ws.items()[0].salience == pytest.approx(0.3)

    def test_sustain_does_not_compound(self):
        ws = _ws(decay=0.5)
        ws.offer("x", "memory", [1.0], 0.8)
        ws.decay()
        ws.sustain()
        ws.sustain()
        assert ws.items()[0].salience == pytest.approx(0.8)

    def test_a_refresh_above_the_old_salience_is_kept(self):
        ws = _ws(decay=0.5)
        ws.offer("x", "memory", [1.0], 0.4)
        ws.decay()
        ws.offer("x", "memory", [1.0], 0.9)
        ws.sustain()
        assert ws.items()[0].salience == pytest.approx(0.9)


class TestThoughtSalience:
    def test_divisive_normalisation_against_the_run_baseline(self):
        from neuroslm.cognition.workspace import thought_salience
        assert thought_salience(0.1, 0.1, ignited=False) == pytest.approx(0.5)
        assert thought_salience(0.0, 0.1, ignited=False) == 0.0
        assert thought_salience(0.3, 0.1, ignited=False) == pytest.approx(0.75)

    def test_ignition_sustains_at_full_salience(self):
        from neuroslm.cognition.workspace import thought_salience
        assert thought_salience(0.0, 0.1, ignited=True) == 1.0

    def test_no_baseline_yet_counts_as_typical(self):
        from neuroslm.cognition.workspace import thought_salience
        assert thought_salience(0.2, None, ignited=False) == pytest.approx(0.5)

    def test_an_exact_repeat_is_not_typical_novelty(self):
        """1 − cos of identical float vectors is ~2e-16, not 0: a run
        of exact repeats must read as zero novelty, not as 0/0 = 0.5
        (it hid the loop from the reorient streak)."""
        from neuroslm.cognition.workspace import divisive_normalise
        assert divisive_normalise(2.2e-16, 2.2e-16) == 0.0
        assert divisive_normalise(2.2e-16, None) == 0.0


class TestStateVector:
    def test_salience_weighted_mean_of_item_latents(self):
        ws = _ws()
        ws.offer("a", "memory", [1.0, 0.0], 0.75)
        ws.offer("b", "thought", [0.0, 1.0], 0.25)
        assert ws.state_vector() == pytest.approx([0.75, 0.25])

    def test_items_without_a_vector_are_skipped(self):
        ws = _ws()
        ws.offer("a", "memory", [], 0.9)
        ws.offer("b", "thought", [0.0, 2.0], 0.5)
        assert ws.state_vector() == pytest.approx([0.0, 2.0])

    def test_empty_or_silent_workspace_has_no_state(self):
        ws = _ws()
        assert ws.state_vector() is None
        ws.offer("a", "memory", [1.0], 0.0)
        assert ws.state_vector() is None
