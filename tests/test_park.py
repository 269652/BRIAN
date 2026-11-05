"""Grounded park environment (neuroslm/environments/park.py).

The mind walks through a persistent park. What it perceives is derived
from the world's geometry — field of view, distance, occlusion — by a
pluggable PerceptChannel, so every percept traces back to a real
object (the grounding contract). A rendered-image channel can replace
the geometry channel later without touching the world or the loop.

Conventions: heading θ in radians, forward = (cos θ, sin θ); bearing is
the signed angle from forward, positive = left.
"""
from __future__ import annotations

import math

import pytest

from neuroslm.environments.park import (
    PARK_ACTIONS, GeometryChannel, Park, ParkObject, Percept)


def _bench(oid="bench1", x=5.0, y=0.0, colour="red"):
    return ParkObject(oid, "bench", x, y, radius=0.8,
                      attrs={"colour": colour})


def _tree(oid="tree1", x=5.0, y=0.0, species="birch"):
    return ParkObject(oid, "tree", x, y, radius=1.0,
                      attrs={"species": species})


def _park(objects, x=0.0, y=0.0, heading=0.0, **kw):
    return Park.from_objects(objects, agent=(x, y, heading), **kw)


def _seen(park):
    return {oid for p in park.perceive() if p.modality == "visual"
            for oid in p.object_ids}


class TestLayout:
    def test_same_seed_same_park(self):
        a, b = Park(seed=3), Park(seed=3)
        assert [(o.id, o.x, o.y) for o in a.objects] == \
            [(o.id, o.x, o.y) for o in b.objects]

    def test_different_seeds_differ(self):
        a, b = Park(seed=3), Park(seed=4)
        assert [(o.x, o.y) for o in a.objects] != \
            [(o.x, o.y) for o in b.objects]

    def test_a_park_has_the_furniture_of_a_park(self):
        kinds = {o.kind for o in Park(seed=0).objects}
        assert {"tree", "bench", "pond", "lamp", "dog", "person"} <= kinds

    def test_objects_are_inside_the_bounds_and_do_not_overlap(self):
        p = Park(seed=1)
        for o in p.objects:
            assert o.radius <= o.x <= p.size - o.radius
            assert o.radius <= o.y <= p.size - o.radius
        solid = [o for o in p.objects if o.solid]
        for i, a in enumerate(solid):
            for b in solid[i + 1:]:
                assert math.hypot(a.x - b.x, a.y - b.y) >= a.radius + b.radius

    def test_the_agent_starts_on_free_ground(self):
        p = Park(seed=2)
        assert not p._blocked(p.agent_x, p.agent_y)


class TestFieldOfView:
    def test_an_object_straight_ahead_is_seen_ahead(self):
        p = _park([_bench(x=5.0, y=0.0)])
        (pc,) = [q for q in p.perceive() if q.modality == "visual"]
        assert pc.object_ids == ("bench1",)
        assert "ahead" in pc.text

    def test_bearing_left_and_right(self):
        p = _park([_bench("L", x=5.0, y=3.0), _bench("R", x=5.0, y=-3.0)])
        txt = {q.object_ids[0]: q.text for q in p.perceive()
               if q.modality == "visual"}
        assert "left" in txt["L"] and "right" in txt["R"]

    def test_outside_the_field_of_view_is_not_seen(self):
        p = _park([_bench(x=0.0, y=-5.0)])       # 90° right, FOV ±60°
        assert _seen(p) == set()

    def test_beyond_view_distance_is_not_seen(self):
        p = _park([_bench(x=60.0, y=0.0)], size=80.0)
        assert _seen(p) == set()

    def test_a_nearer_object_occludes_one_behind_it(self):
        p = _park([_tree(x=5.0, y=0.0), _bench("hidden", x=10.0, y=0.0),
                   _bench("beside", x=10.0, y=4.0)])
        assert _seen(p) == {"tree1", "beside"}

    def test_a_percept_carries_where_it_is_and_whether_it_blocks_the_view(self):
        p = _park([_tree(x=5.0, y=0.0), _bench("b", x=4.0, y=4.0)])
        q = {x.object_ids[0]: x for x in p.perceive()
             if x.modality == "visual"}
        t, b = q["tree1"], q["b"]
        assert t.bearing == pytest.approx(0.0)
        assert t.extent == pytest.approx(math.asin(1.0 / 5.0))
        assert t.occludes and not b.occludes
        assert b.bearing == pytest.approx(math.pi / 4)

    def test_near_objects_show_detail_far_ones_only_their_kind(self):
        near = _park([_bench(x=3.0, colour="red")])
        far = _park([_bench(x=18.0, colour="red")])
        (n,) = [q.text for q in near.perceive() if q.modality == "visual"]
        (f,) = [q.text for q in far.perceive() if q.modality == "visual"]
        assert "red" in n
        assert "bench" in f and "red" not in f


class TestActions:
    def test_the_action_repertoire(self):
        assert {"walk", "turn_left", "turn_right", "look",
                "approach", "head"} <= set(PARK_ACTIONS)

    def test_walk_moves_forward(self):
        p = _park([], x=10.0, y=10.0, heading=0.0)
        p.step("walk")
        assert p.agent_x == pytest.approx(10.0 + p.stride)
        assert p.agent_y == pytest.approx(10.0)

    def test_turning_rotates_by_the_turn_angle_and_wraps(self):
        p = _park([], heading=0.0)
        p.step("turn_left")
        assert p.heading == pytest.approx(math.pi / 4)
        for _ in range(8):
            p.step("turn_right")
        assert p.heading == pytest.approx(math.pi / 4), "wrapped to (−π, π]"

    def test_heading_off_turns_and_steps_in_one_tick(self):
        p = _park([], x=10.0, y=10.0, heading=0.0)
        p.step("head:90")
        assert p.heading == pytest.approx(math.pi / 2)
        assert (p.agent_x, p.agent_y) == pytest.approx((10.0, 10.0 + p.stride))

    def test_heading_into_an_obstacle_turns_but_does_not_move(self):
        p = _park([_tree(x=10.0, y=11.5)], x=10.0, y=10.0, heading=0.0)
        f = p.step("head:90")
        assert p.heading == pytest.approx(math.pi / 2)
        assert (p.agent_x, p.agent_y) == (10.0, 10.0)
        assert "bump" in f.tactile

    def test_a_malformed_heading_is_rejected(self):
        with pytest.raises(ValueError):
            _park([]).step("head:north")

    def test_walking_into_an_obstacle_is_blocked_and_felt(self):
        p = _park([_tree(x=11.5, y=10.0)], x=10.0, y=10.0)
        frame = p.step("walk")
        assert (p.agent_x, p.agent_y) == (10.0, 10.0)
        assert "bump" in frame.tactile

    def test_cannot_walk_out_of_the_park(self):
        p = _park([], x=0.5, y=10.0, heading=math.pi)
        p.step("walk")
        assert p.agent_x >= 0.0

    def test_approach_closes_on_a_visible_object(self):
        p = _park([_bench(x=12.0, y=0.0)], x=0.0, y=0.0)
        d0 = math.hypot(12.0 - p.agent_x, -p.agent_y)
        p.step("approach:bench1")
        d1 = math.hypot(12.0 - p.agent_x, -p.agent_y)
        assert d1 < d0
        for _ in range(20):
            p.step("approach:bench1")
        d = math.hypot(12.0 - p.agent_x, -p.agent_y)
        assert d <= 0.8 + p.stride and not p._blocked(p.agent_x, p.agent_y)

    def test_cannot_approach_what_is_not_seen(self):
        # grounding: the act needs a percept, not world knowledge
        p = _park([_bench(x=-8.0, y=0.0)])        # behind the agent
        p.step("approach:bench1")
        assert (p.agent_x, p.agent_y) == (0.0, 0.0)

    def test_unknown_actions_are_rejected(self):
        with pytest.raises(ValueError):
            _park([]).step("fly")


class TestDynamics:
    def test_the_dog_wanders_but_stays_in_the_park(self):
        p = Park(seed=0)
        dog = next(o for o in p.objects if o.kind == "dog")
        start = (dog.x, dog.y)
        for _ in range(30):
            p.step("look")
        assert (dog.x, dog.y) != start
        assert 0.0 <= dog.x <= p.size and 0.0 <= dog.y <= p.size

    def test_a_barking_dog_is_heard_behind_the_agent(self):
        dog = ParkObject("dog1", "dog", -6.0, 0.0, radius=0.4, solid=False,
                         attrs={"colour": "brown"}, sound="barking")
        p = _park([dog], size=40.0)
        heard = [q for q in p.perceive() if q.modality == "auditory"]
        assert heard and "behind" in heard[0].text
        assert "dog1" in heard[0].object_ids
        assert _seen(p) == set()

    def test_time_passes_and_the_light_changes(self):
        p = Park(seed=0, start_minute=17 * 60, minutes_per_tick=10.0)
        first = p.step("look").time_of_day
        for _ in range(24):
            last = p.step("look").time_of_day
        assert first != last


class TestGrounding:
    def test_every_percept_names_real_objects(self):
        p = Park(seed=5)
        ids = {o.id for o in p.objects}
        for _ in range(15):
            p.step("walk" if p.tick % 3 else "turn_left")
            for q in p.perceive():
                assert set(q.object_ids) <= ids

    def test_step_returns_a_sensory_frame_built_from_the_percepts(self):
        p = _park([_bench(x=4.0, colour="green")])
        f = p.step("look")
        assert f.environment == "park"
        assert "green bench" in f.visual
        assert p.last_percepts and isinstance(p.last_percepts[0], Percept)

    def test_the_channel_is_pluggable(self):
        class Blind:
            def perceive(self, park):
                return [Percept("darkness", (), "visual")]
        p = _park([_bench(x=4.0)], channel=Blind())
        assert p.step("look").visual == "darkness"

    def test_registered_with_the_environment_factory(self):
        from neuroslm.environments.virtual_world import create_environment
        assert isinstance(create_environment("park", seed=1), Park)

    def test_geometry_channel_is_the_default(self):
        assert isinstance(Park(seed=0).channel, GeometryChannel)
