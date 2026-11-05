"""Embodied loop (neuroslm/cognition/embodiment.py): the mind walks the
grounded park. Each step the park's percepts are sensed, the mind
thinks, and the BG chooses the next movement — which the park executes.
Grounding metrics are computed against the world state."""
from __future__ import annotations

import math
import random

import pytest

from neuroslm.environments.park import Park, ParkObject

from test_cognitive_runtime import _ScriptedGen, _mk_runtime


def _rt(thoughts=("a passing thought",), seed=0):
    from neuroslm.cognition.runtime import MindConfig
    return _mk_runtime(_ScriptedGen(list(thoughts)), seed=seed,
                       cfg=MindConfig(n_candidates=1, recall_k=0))


def _bench(oid="bench1", x=5.0, y=0.0, colour="red"):
    return ParkObject(oid, "bench", x, y, radius=0.8,
                      attrs={"colour": colour})


def _tree(oid="tree1", x=5.0, y=0.0):
    return ParkObject(oid, "tree", x, y, radius=1.0,
                      attrs={"species": "oak"})


class TestWalkLoop:
    def test_percepts_reach_the_mind_as_sights(self):
        from neuroslm.cognition.embodiment import walk
        park = Park.from_objects([_bench(x=4.0)], agent=(0.0, 0.0, 0.0))
        rt = _rt()
        walk(rt, park, ticks=1)
        held = {it.text for it in rt.workspace.items()
                if it.source == "sight"}
        assert held == {p.text for p in park.last_percepts
                        if p.modality == "visual"}

    def test_the_chosen_movement_is_executed_by_the_park(self):
        from neuroslm.cognition.embodiment import walk
        park = Park(seed=3)
        recs = walk(_rt(), park, ticks=6)
        for prev, nxt in zip(recs, recs[1:]):
            assert nxt.action == prev.next_action

    def test_approach_is_offered_only_for_what_is_seen(self):
        from neuroslm.cognition.embodiment import movement_options
        park = Park.from_objects([_bench("front", x=4.0),
                                  _bench("back", x=-4.0)],
                                 agent=(0.0, 0.0, 0.0))
        park.step("look")
        acts = {a for a, _ in movement_options(park)}
        assert "approach:front" in acts and "approach:back" not in acts
        assert {"look", "head:0", "head:90", "head:180", "head:-90"} <= acts

    def test_records_carry_pose_percepts_and_thought(self):
        from neuroslm.cognition.embodiment import walk
        recs = walk(_rt(), Park(seed=1), ticks=3)
        assert len(recs) == 3
        r = recs[-1]
        assert r.thought == "a passing thought"
        assert isinstance(r.seen_ids, tuple) and len(r.pose) == 3


class TestPlaceNovelty:
    """Exploration by place novelty: the mind remembers which ground it
    has viewed — from its own pose and field of view, never from the
    world's objects — and values a move by the never-viewed ground in
    the direction it would face."""

    def test_viewed_ground_is_not_novel_unviewed_ground_is(self):
        from neuroslm.cognition.embodiment import PlaceMemory
        pm = PlaceMemory(size=60.0)
        pm.view(30.0, 30.0, 0.0, reach=100.0)
        assert pm.novelty_ahead(30.0, 30.0, 0.0) == 0.0
        assert pm.novelty_ahead(30.0, 30.0, math.pi) > 0.8

    def test_novelty_counts_ground_beyond_the_view_distance(self):
        # what lies that way, all the way to the fence — a short
        # viewed cone must not hide a large unexplored region behind it
        from neuroslm.cognition.embodiment import PlaceMemory
        pm = PlaceMemory(size=60.0)
        pm.view(5.0, 30.0, 0.0, reach=10.0)
        assert pm.novelty_ahead(5.0, 30.0, 0.0) > 0.7

    def test_novelty_ignores_ground_outside_the_park(self):
        from neuroslm.cognition.embodiment import PlaceMemory
        pm = PlaceMemory(size=60.0)
        pm.view(1.0, 30.0, 0.0)
        assert pm.novelty_ahead(1.0, 30.0, math.pi) == 0.0, (
            "nothing to discover beyond the fence")

    def test_coverage_of_the_park_ground(self):
        from neuroslm.cognition.embodiment import PlaceMemory
        pm = PlaceMemory(size=60.0)
        assert pm.coverage() == 0.0
        pm.view(30.0, 30.0, 0.0)
        assert 0.0 < pm.coverage() < 0.5

    def test_ground_behind_a_seen_obstruction_is_not_viewed(self):
        from neuroslm.cognition.embodiment import PlaceMemory
        pm = PlaceMemory(size=60.0)
        # a tree 5 m due east of (30, 30), half-width 0.3 rad
        pm.view(30.0, 30.0, 0.0, occluders=[(5.0, 0.0, 0.3)])
        cell = lambda x, y: pm._viewed[int(x // pm.cell), int(y // pm.cell)]
        assert cell(33.0, 30.5) and not cell(45.0, 30.5)
        assert cell(45.0, 38.0), "off to the side stays visible"

    def test_facing_the_fence_is_not_inflated(self):
        # a fraction of a fence-facing cone (few in-park cells) made the
        # fence look like the most unexplored direction (stuck 200 ticks)
        from neuroslm.cognition.embodiment import PlaceMemory, _place_values
        park = Park.from_objects([], agent=(58.0, 30.0, 0.0))
        vals = _place_values(PlaceMemory(park.size), park)
        assert vals["head:180"] == pytest.approx(1.0)
        assert vals["head:0"] < 0.2

    def test_a_bump_steers_the_mind_another_way(self):
        from neuroslm.cognition.embodiment import PlaceMemory, walk
        for seed in range(10):
            park = Park.from_objects([_tree(x=31.6, y=30.0)],
                                     agent=(30.0, 30.0, 0.0))
            pm = PlaceMemory(park.size)
            pm._viewed[:] = True
            pm._viewed[18:, 12:18] = False      # unexplored, beyond the tree
            # going around a tree can take a few blocked tries
            recs = walk(_rt(seed=seed), park, ticks=6, action="head:0",
                        places=pm)
            assert recs[0].next_action != "head:0", "just bumped that way"
            assert any(r.pose[:2] != (30.0, 30.0) for r in recs)

    def test_a_blocked_way_is_not_offered_even_when_nothing_else_pulls(self):
        # value 0 for blocked tied with every explored direction; random
        # tie-breaking kept choosing it (stuck 83 ticks)
        from neuroslm.cognition.embodiment import PlaceMemory, walk
        for seed in range(100):
            park = Park.from_objects([_tree(x=31.6, y=30.0)],
                                     agent=(30.0, 30.0, 0.0))
            pm = PlaceMemory(park.size)
            pm._viewed[:] = True                 # nothing left to explore
            recs = walk(_rt(seed=seed), park, ticks=1, action="head:0",
                        places=pm)
            assert recs[0].next_action not in ("head:0",)

    def test_the_last_unexplored_corner_still_pulls(self):
        # place values are divisively normalised over the options' pool:
        # a small absolute remainder is still the best direction
        from neuroslm.cognition.embodiment import PlaceMemory, _place_values
        park = Park.from_objects([], agent=(30.0, 30.0, 0.0))
        pm = PlaceMemory(park.size)
        pm._viewed[:] = True
        pm._viewed[-3:, 12:18] = False          # a small patch due east
        vals = _place_values(pm, park)
        assert vals["head:0"] == pytest.approx(1.0)
        # walking distance: the way back still reaches it, the long way
        assert vals["head:180"] < vals["head:0"]

    def test_moves_toward_unviewed_ground_are_preferred(self):
        rt = _rt()
        rt._boredom = 1.0
        opts = [("walk", None), ("turn_left", None)]
        n = sum(rt.choose_movement(opts, expected={"walk": 1.0,
                                                   "turn_left": 0.0})[0]
                == "walk" for _ in range(300))
        assert n / 300 > 0.7


class TestCognitiveMap:
    """Obstacles seen (solid percepts) or felt (bumps) go into the place
    memory; exploration value follows walking distance around them, so
    the mind does not oscillate against the near side of a pond whose far
    side is unexplored."""

    def test_a_seen_solid_object_is_mapped(self):
        from neuroslm.cognition.embodiment import PlaceMemory
        pm = PlaceMemory(size=60.0)
        # a pond 10 m due east of (20, 30), angular half-width asin(4/10)
        pm.mark_obstacle(20.0, 30.0, 0.0, 10.0, 0.0, math.asin(0.4))
        cell = lambda x, y: pm._obstacle[int(x // pm.cell), int(y // pm.cell)]
        assert cell(30.0, 30.0) and cell(33.0, 30.0)
        assert not cell(20.0, 30.0) and not cell(30.0, 40.0)

    def test_value_follows_walking_distance_not_the_straight_line(self):
        from neuroslm.cognition.embodiment import PlaceMemory
        pm = PlaceMemory(size=60.0)
        pm._viewed[:] = True
        pm._viewed[25:, 13:17] = False          # unexplored: x ≥ 50
        pm._obstacle[20, :26] = True            # wall at x≈40, gap at y > 52
        # straight east hits the wall; north-east leads round it
        east = pm.value_toward(30.0, 30.0, 0.0, 1.5)
        north = pm.value_toward(30.0, 30.0, math.pi / 2, 1.5)
        assert north > east

    def test_ground_sealed_off_by_obstacles_does_not_pull(self):
        from neuroslm.cognition.embodiment import PlaceMemory
        pm = PlaceMemory(size=60.0)
        pm._viewed[:] = True
        pm._viewed[5, 5] = False
        pm._obstacle[4:7, 4:7] = True
        pm._obstacle[5, 5] = False              # an unviewed hole, walled in
        assert pm.value_toward(30.0, 30.0, math.pi * 1.25, 1.5) == 0.0

    def test_a_step_into_an_obstacle_is_worth_nothing(self):
        from neuroslm.cognition.embodiment import PlaceMemory
        pm = PlaceMemory(size=60.0)
        pm._obstacle[15, 15] = True             # the cell at (31, 31)
        assert pm.value_toward(29.6, 31.0, 0.0, 1.5) == 0.0

    def test_percepts_say_what_is_solid(self):
        p = Park.from_objects(
            [ParkObject("pond", "pond", 8.0, 0.0, radius=3.0),
             ParkObject("dog", "dog", 4.0, 2.0, radius=0.4, solid=False)],
            agent=(0.0, 0.0, 0.0))
        q = {x.object_ids[0]: x for x in p.perceive()
             if x.modality == "visual"}
        assert q["pond"].solid and not q["dog"].solid

    def test_the_mind_sees_everything_in_the_park(self):
        # every static object, in every park; people and the dog move and
        # are met by chance, so they are reported, not guaranteed
        from neuroslm.cognition.embodiment import walk, grounding_report
        for seed in range(5):
            park = Park(seed=seed)
            recs = walk(_rt(seed=seed), park, ticks=300)
            rep = grounding_report(recs, park)
            assert rep["static_coverage"] == 1.0, (seed, rep)
            assert rep["coverage"] >= 0.95, (seed, rep)


class TestGroundingReport:
    def test_coverage_is_the_share_of_objects_ever_seen(self):
        from neuroslm.cognition.embodiment import WalkRecord, grounding_report
        park = Park.from_objects([_bench("a"), _bench("b", x=-5.0)],
                                 agent=(0.0, 0.0, 0.0))
        recs = [WalkRecord(1, "look", "look", (0, 0, 0), ("a",),
                           ("A red bench, a few metres ahead.",), "hm")]
        assert grounding_report(recs, park)["coverage"] == pytest.approx(0.5)

    def test_static_coverage_leaves_out_the_movers(self):
        from neuroslm.cognition.embodiment import WalkRecord, grounding_report
        dog = ParkObject("dog", "dog", 9.0, 9.0, 0.4, solid=False, speed=2.5)
        park = Park.from_objects([_bench("a"), dog], agent=(0.0, 0.0, 0.0))
        recs = [WalkRecord(1, "look", "look", (0, 0, 0), ("a",), (), "hm")]
        rep = grounding_report(recs, park)
        assert rep["static_coverage"] == 1.0 and rep["coverage"] == 0.5

    def test_mentioning_an_unseen_kind_is_confabulation(self):
        from neuroslm.cognition.embodiment import WalkRecord, grounding_report
        park = Park.from_objects(
            [_bench("a"), ParkObject("f", "fountain", 20.0, 20.0, 1.5)],
            agent=(0.0, 0.0, 0.0))
        recs = [WalkRecord(1, "look", "look", (0, 0, 0), ("a",),
                           ("A red bench, a few metres ahead.",),
                           "I could sit on that bench by the fountain.")]
        rep = grounding_report(recs, park)
        assert rep["mentions"] == 2
        assert rep["grounded_mentions"] == 1
        assert rep["confabulated"] == ["fountain"]
        assert rep["grounded_rate"] == pytest.approx(0.5)

    def test_a_kind_seen_earlier_stays_grounded(self):
        from neuroslm.cognition.embodiment import WalkRecord, grounding_report
        park = Park.from_objects([_bench("a")], agent=(0.0, 0.0, 0.0))
        recs = [WalkRecord(1, "look", "turn_left", (0, 0, 0), ("a",),
                           ("A red bench, a few metres ahead.",), "nice"),
                WalkRecord(2, "turn_left", "look", (0, 0, 1), (), (),
                           "that red bench was nice")]
        rep = grounding_report(recs, park)
        assert rep["confabulated"] == [] and rep["grounded_rate"] == 1.0

    def test_no_mentions_has_no_rate(self):
        from neuroslm.cognition.embodiment import WalkRecord, grounding_report
        park = Park.from_objects([_bench("a")], agent=(0.0, 0.0, 0.0))
        recs = [WalkRecord(1, "look", "look", (0, 0, 0), (), (), "hmm")]
        assert grounding_report(recs, park)["grounded_rate"] is None


class TestSenseReturnsSalience:
    def test_sense_reports_the_percept_salience(self):
        rt = _rt()
        assert rt.sense("the river bank", "sight") == pytest.approx(0.5)
        assert rt.sense("the river bank", "sight") == pytest.approx(0.0)
