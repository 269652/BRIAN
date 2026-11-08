"""Embodied loop (neuroslm/cognition/embodiment.py): the mind walks the
grounded park. Each step the park's percepts are sensed, the mind
thinks, and the BG chooses the next movement — which the park executes.
Grounding metrics are computed against the world state."""
from __future__ import annotations

import math
import random

import pytest

from neuroslm.environments.park import Park, ParkObject

from test_cognitive_runtime import _ScriptedGen, _mk_runtime, _score_map,     _vec_for


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


class TestProprioception:
    """H69 (h): 30 % of the raw world model's ungrounded nouns were
    compass words. The body knows its heading and where it is in the
    park (path integration); proprioception tells the mind, in words
    that are true."""

    def _say(self, x, y, h):
        from neuroslm.cognition.embodiment import proprioception
        return proprioception(Park.from_objects([], agent=(x, y, h)))

    def test_heading_is_a_compass_direction(self):
        # forward = (cos θ, sin θ); +y is north
        assert self._say(30, 30, math.pi / 2) == \
            "I am facing north, in the middle of the park."
        assert "facing east" in self._say(30, 30, 0.0)
        assert "facing north-east" in self._say(30, 30, math.pi / 4)
        assert "facing south" in self._say(30, 30, -math.pi / 2)
        assert "facing south-west" in self._say(30, 30, -3 * math.pi / 4)

    def test_position_is_a_part_of_the_park(self):
        assert "in the north-west of the park" in self._say(5, 55, 0.0)
        assert "in the middle of the park" in self._say(30, 30, 0.0)
        assert "in the south of the park" in self._say(30, 5, 0.0)

    def test_a_near_fence_is_named_by_its_side(self):
        assert "The park fence is close to the east." in \
            self._say(58, 30, 0.0)
        assert "close to the north and the west" in self._say(2, 58, 0.0)
        assert "fence" not in self._say(30, 30, 0.0)

    def test_the_walk_senses_the_body_when_it_changes(self):
        from neuroslm.cognition.embodiment import proprioception, walk
        park = Park.from_objects([], agent=(30.0, 30.0, math.pi / 2))
        rt = _rt()
        sensed = []
        real = rt.sense
        rt.sense = lambda text, source="sight", key=None: (
            sensed.append((source, text)), real(text, source, key))[1]
        walk(rt, park, ticks=1, action="look")
        body = [t for s, t in sensed if s == "body"]
        assert body == [proprioception(park)]
        assert "north" in rt._perceived_nouns

    def test_an_unchanged_body_is_not_sensed_again(self):
        from neuroslm.cognition.embodiment import PlaceMemory, walk
        park = Park.from_objects([], agent=(30.0, 30.0, math.pi / 2))
        rt, places = _rt(), PlaceMemory(park.size)
        n = []
        real = rt.sense
        rt.sense = lambda text, source="sight", key=None: (
            n.append(source), real(text, source, key))[1]
        walk(rt, park, ticks=1, action="look", places=places)
        walk(rt, park, ticks=1, action="look", places=places)
        assert n.count("body") == 1

    def test_the_body_percept_grounds_what_it_names(self):
        from neuroslm.cognition.embodiment import grounding_report, walk
        park = Park.from_objects([], agent=(30.0, 30.0, math.pi / 2))
        recs = walk(_rt(), park, ticks=1, action="look")
        rep = grounding_report(recs, park, models={
            "world": [(recs[-1].tick, "I am facing north."),
                      (recs[-1].tick, "I am facing south.")]})
        assert rep["models"]["world"]["grounded_versions"] == 1
        assert rep["models"]["world"]["ungrounded"] == ["south"]

    def test_can_be_switched_off(self):
        from neuroslm.cognition.embodiment import walk
        from neuroslm.cognition.runtime import MindConfig
        rt = _mk_runtime(_ScriptedGen(["x"]),
                         cfg=MindConfig(n_candidates=1, recall_k=0,
                                        proprioception=False))
        walk(rt, Park.from_objects([], agent=(30.0, 30.0, 0.0)), ticks=1)
        assert "east" not in rt._perceived_nouns


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

    def test_a_less_bored_mind_still_explores_decisively(self):
        # run 59316881: boredom ≈ 0.5 under R1 (the scripted mind's goes
        # to 1); with frontier value 1/(1+d) the neighbouring headings
        # were too close for the motor softmax — it paced back and forth
        # near the west fence and saw 31 % of the park in 120 ticks
        from neuroslm.cognition.embodiment import walk, grounding_report
        covs = []
        for seed in range(4):
            rt, park = _rt(seed=seed), Park(seed=seed)

            def hold(rec, res, rt=rt):
                rt._boredom = 0.5
            rt._boredom = 0.5
            recs = walk(rt, park, ticks=120, on_tick=hold)
            covs.append(grounding_report(recs, park)["static_coverage"])
        assert sum(covs) / len(covs) >= 0.9, covs

    def test_the_frontier_is_discounted_per_cell_of_travel(self):
        from neuroslm.cognition.embodiment import PlaceMemory
        pm = PlaceMemory(size=60.0)
        pm._viewed[:] = True
        pm._viewed[25, 15] = False               # the frontier at x≈51
        near = pm.value_toward(40.0, 31.0, 0.0, 1.5)
        far = pm.value_toward(40.0, 31.0, math.pi, 1.5)
        assert far / near == pytest.approx(pm.discount ** 3)   # 7 vs 4 cells


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


class TestSceneGrounding:
    """Run 59316881: the world model claimed "a red and white traffic
    light", "a street scene… buildings", "a camera on my head" — and the
    grounding rate still read 1.0, because it only counted the park's own
    kinds. Scene nouns (any concrete thing a text can claim is there) are
    grounded only when a percept so far has named them."""

    def test_unperceived_scene_nouns_are_ungrounded(self):
        from neuroslm.cognition.situation import ungrounded_nouns
        seen = "A red bench, a few metres ahead. A fountain splashing."
        assert ungrounded_nouns(
            "I am by the bench, near a traffic light and a building.",
            seen) == ["traffic light", "building"]

    def test_plurals_and_case_match_their_noun(self):
        from neuroslm.cognition.situation import ungrounded_nouns
        assert ungrounded_nouns("Two Benches and the Fountains.",
                                "A bench. A fountain.") == []

    def test_a_compound_is_not_also_counted_as_its_head(self):
        from neuroslm.cognition.situation import ungrounded_nouns
        assert ungrounded_nouns("A traffic light.", "A lamp.") == [
            "traffic light"]
        assert ungrounded_nouns("The flower bed.", "A blue flower bed.") == []

    def test_weather_compass_and_life_events_are_claims_too(self):
        # run 59316881: "a sunny day… 25°C… air pressure 1013 hPa",
        # "Looking towards the north", "the upcoming meeting… study session"
        from neuroslm.cognition.situation import ungrounded_nouns
        assert ungrounded_nouns(
            "It is sunny; the temperature is mild and the air pressure "
            "high. I face north before my meeting and study session.",
            "A red bench.") == ["sunny", "temperature", "air pressure",
                                "north", "meeting", "study session"]

    def test_the_report_scores_thoughts_and_models(self):
        from neuroslm.cognition.embodiment import WalkRecord, grounding_report
        park = Park.from_objects([_bench("a")], agent=(0.0, 0.0, 0.0))
        recs = [WalkRecord(1, "look", "look", (0, 0, 0), ("a",),
                           ("A red bench, a few metres ahead.",),
                           "The bench is by the street."),
                WalkRecord(2, "look", "look", (0, 0, 0), (), (),
                           "That bench again.")]
        rep = grounding_report(recs, park, models={
            "world": [(1, "I am by a bench."),
                      (2, "I am at a crossroads with a camera.")]})
        assert rep["scene_mentions"] == 3
        assert rep["scene_grounded_rate"] == pytest.approx(2 / 3)
        assert rep["scene_ungrounded"] == ["street"]
        w = rep["models"]["world"]
        assert w["versions"] == 2 and w["grounded_versions"] == 1
        assert w["ungrounded"] == ["crossroads", "camera"]

    def test_a_noun_is_grounded_only_once_perceived(self):
        from neuroslm.cognition.embodiment import WalkRecord, grounding_report
        park = Park.from_objects([_bench("a")], agent=(0.0, 0.0, 0.0))
        recs = [WalkRecord(1, "look", "look", (0, 0, 0), (), (),
                           "Where is a bench?"),
                WalkRecord(2, "look", "look", (0, 0, 0), ("a",),
                           ("A red bench, close.",), "")]
        rep = grounding_report(recs, park, models={"world": [(1, "A bench.")]})
        assert rep["scene_ungrounded"] == ["bench"]
        assert rep["models"]["world"]["grounded_versions"] == 0


class TestSenseReturnsSalience:
    def test_sense_reports_the_percept_salience(self):
        rt = _rt()
        assert rt.sense("the river bank", "sight") == pytest.approx(0.5)
        assert rt.sense("the river bank", "sight") == pytest.approx(0.0)


_LINES = ("the bench by the river", "a dog near the fountain",
          "rain on the lamp", "the willow tree", "a quiet moon")


def _pure_gen(prompt, max_new_tokens):
    """A pure function of the prompt — like a reseeded sampler: equal
    states give equal thoughts."""
    import zlib
    return _LINES[zlib.crc32(prompt.encode()) % len(_LINES)]


def _real_rt():
    from neuroslm.cognition.runtime import CognitiveRuntime, MindConfig
    from neuroslm.memory.episodic import EpisodicMemory
    return CognitiveRuntime(
        generate_fn=_pure_gen, score_fn=_score_map({}), embed_fn=_vec_for,
        memory=EpisodicMemory(maxlen=64), rng=random.Random(0),
        cfg=MindConfig(n_candidates=2, world_model=False, self_model=False))


class TestTwinWalks:
    """PCI during the walk (Casali et al. 2013, by twin runs): fork the
    whole mind, its body and its map mid-walk, perturb the twin, and
    read the response as the twin's divergence from the main walk. An
    unperturbed twin must replay the walk exactly — that is the null
    the method rests on."""

    def test_a_fork_is_independent_but_shares_the_trunk(self):
        rt = _real_rt()
        rt.tick()
        twin = rt.fork()
        n_mem = len(rt.memory.all())
        twin.tick()
        twin.memory.add("only the twin", content_vec=_vec_for("moon"))
        assert len(rt.memory.all()) == n_mem
        assert twin._gen is rt._gen and twin._embed is rt._embed

    def _main_and_twin(self, pulse, at=3, k=4):
        from neuroslm.cognition.embodiment import PlaceMemory, twin_walk, walk
        rt, park = _real_rt(), Park(seed=0)
        places = PlaceMemory(park.size)
        main, twin = {}, {}

        def on_tick(rec, r):
            main[rec.tick] = r
            if rec.tick == at:
                for j, tr in enumerate(twin_walk(rt, park, places,
                                                 rec.next_action, k,
                                                 pulse=pulse)):
                    twin[at + 1 + j] = tr
        walk(rt, park, at + k, places=places, on_tick=on_tick)
        return main, twin

    def test_an_unperturbed_twin_replays_the_walk(self):
        main, twin = self._main_and_twin(None)
        assert sorted(twin) == [4, 5, 6, 7]
        for t, tr in twin.items():
            assert tr.thought == main[t].thought
            assert tr.nt_levels == main[t].nt_levels

    def test_a_pulse_makes_the_twin_diverge(self):
        main, twin = self._main_and_twin(lambda r: r.nt.perturb("DA", 2.0))
        assert twin[4].nt_levels["DA"] != pytest.approx(
            main[4].nt_levels["DA"], abs=1e-3)
