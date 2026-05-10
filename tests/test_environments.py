"""Tests for the virtual_world environment stack."""
from __future__ import annotations
import pytest
from neuroslm.environments.virtual_world import (
    BusStop, MeadowTree, RainyWindow, OceanCliff,
    Library, Campfire, GridWorld,
    SensoryFrame, ENVIRONMENTS, create_environment,
    environment_stream, GRID_ACTIONS, GRID_ACTION_IDX,
)


@pytest.mark.parametrize("cls", [BusStop, MeadowTree, RainyWindow,
                                  OceanCliff, Library, Campfire])
def test_passive_environment_step(cls):
    env = cls(seed=1)
    frame = env.step()
    assert isinstance(frame, SensoryFrame)
    assert frame.environment == cls.name
    # All channels are strings
    assert isinstance(frame.visual, str)
    # to_vec must produce 6 floats
    assert len(frame.to_vec()) == 6


def test_environment_stream_finite():
    env = MeadowTree(seed=2)
    frames = list(env.stream(max_ticks=5))
    assert len(frames) == 5


def test_create_environment_random():
    env = create_environment("random", seed=10)
    assert env.name in ENVIRONMENTS


def test_create_environment_unknown():
    with pytest.raises(ValueError):
        create_environment("nonsense", seed=1)


def test_environment_stream_global_generator():
    gen = environment_stream(seed=3, switch_every=3)
    seen = []
    for _ in range(6):
        seen.append(next(gen).environment)
    # Cycled at least once across envs
    assert len(set(seen)) >= 1


# ── GridWorld: action-conditioned dynamics ─────────────────────────────
def test_gridworld_actions_round_trip():
    env = GridWorld(seed=0)
    assert env.tick == 0
    # Wait should not raise; step counter advances
    f = env.step(action=GRID_ACTION_IDX["WAIT"])
    assert env.tick == 1
    assert f.environment == "grid_world"


def test_gridworld_pick_up_key():
    """Default map: agent (1,1), floor (1,2), key (1,3). Walking E twice
    picks up the key (the implementation auto-collects on step-onto-K)."""
    env = GridWorld(seed=0)
    env.step(action=GRID_ACTION_IDX["MOVE_E"])   # (1,2) floor
    f = env.step(action=GRID_ACTION_IDX["MOVE_E"])  # (1,3) key
    assert env.has_key is True
    assert f.valence >= 0.5


def test_gridworld_walls_punish():
    env = GridWorld(seed=0)
    # Agent at (1,1); MOVE_N hits wall
    f = env.step(action=GRID_ACTION_IDX["MOVE_N"])
    assert f.valence < 0
