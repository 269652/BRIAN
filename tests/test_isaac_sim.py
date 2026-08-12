# -*- coding: utf-8 -*-
"""Isaac Sim sensory bridge (§15): pumps a simulated environment's
camera / articulation / (optional) audio streams through the §15
cortices into a CognitiveRuntime's ``observe_sensory`` SENSE path.

``OmniverseIsaacSimClient`` (the real Isaac Sim API wrapper) isn't
exercised here — it requires a running Isaac Sim process and its
bundled ``isaacsim`` Python API, neither available in this test
environment. ``SensoryBridge`` is dependency-injected against a fake
client double instead (the same pattern ``build_runtime_from_hf_lm``
uses for its LM) — the mechanism under test is the bridge's polling
and cortex wiring, which is real and fully exercised. The cortices
themselves are ALSO injected here with cheap deterministic fakes (real
math already pinned directly in tests/sensory/test_cortices.py) so
this suite never triggers a real CLIP download from HF Hub.
"""
import random

import numpy as np
import pytest

from neuroslm.cognition.runtime import CognitiveRuntime, MindConfig
from neuroslm.memory.episodic import EpisodicMemory
from neuroslm.connectors.isaac_sim import SensoryBridge


def _vec_for(text):
    axes = ("code", "weather", "music", "launch", "coffee", "river",
            "chess", "moon")
    v = [0.05] * len(axes)
    low = (text or "").lower()
    for i, w in enumerate(axes):
        if w in low:
            v[i] += 1.0
    return v


class _ScriptedGen:
    def __init__(self, outputs):
        self.outputs = list(outputs)
        self.i = 0
        self.prompts = []

    def __call__(self, prompt, n_tok):
        self.prompts.append(prompt)
        out = self.outputs[self.i % len(self.outputs)]
        self.i += 1
        return out


def _mk_runtime(cfg=None):
    from neuroslm.cognition.runtime import ThoughtScore

    def score_fn(text):
        return ThoughtScore(mean_nll=1.0, entropy_norm=0.5)

    return CognitiveRuntime(
        generate_fn=_ScriptedGen(["a thought"]),
        score_fn=score_fn,
        embed_fn=_vec_for,
        memory=EpisodicMemory(maxlen=64),
        cfg=cfg or MindConfig(n_candidates=1),
        rng=random.Random(0),
    )


class _FakeIsaacClient:
    """Stand-in for OmniverseIsaacSimClient: same call surface
    (``get_frame`` / ``get_joint_state`` / ``get_audio_chunk``),
    deterministic canned data, no Isaac Sim process required."""

    def __init__(self, frame=None, joints=None, audio=None,
                fail_frame=False):
        self._frame = frame
        self._joints = joints
        self._audio = audio
        self._fail_frame = fail_frame

    def get_frame(self):
        if self._fail_frame:
            raise RuntimeError("simulated sensor glitch")
        return self._frame

    def get_joint_state(self):
        return self._joints

    def get_audio_chunk(self):
        return self._audio


def _frame(fill=128):
    return np.ones((16, 16, 3), dtype="uint8") * fill


def _fake_cortex_for(rt, axis=0):
    """Deterministic stand-in for a real §15 cortex: content-dependent
    (so distinct inputs -> distinct vectors) without any network call
    or heavy model construction — SensoryBridge's own polling/wiring
    logic is what these tests exercise, not the cortices themselves
    (those are pinned directly in tests/sensory/test_cortices.py)."""
    dim = rt.embed_dim()

    def _cortex(raw):
        if hasattr(raw, "mean"):
            key = float(raw.mean())
        elif isinstance(raw, dict):
            flat = [x for v in raw.values() for x in (v or [])]
            key = float(sum(flat)) if flat else 0.0
        else:
            key = float(sum(raw)) if raw else 0.0
        vec = [0.0] * dim
        vec[axis % dim] = key or 1e-6
        return vec

    return _cortex


class TestSensoryBridge:
    def _bridge(self, rt, client, **kw):
        kw.setdefault("visual_cortex", _fake_cortex_for(rt, axis=0))
        kw.setdefault("acoustic_cortex", _fake_cortex_for(rt, axis=1))
        kw.setdefault("proprioceptive_cortex", _fake_cortex_for(rt, axis=2))
        return SensoryBridge(rt, client, **kw)

    def test_pumps_visual_and_proprioceptive_by_default(self):
        rt = _mk_runtime()
        client = _FakeIsaacClient(
            frame=_frame(200),
            joints={"positions": [0.1, 0.2], "velocities": [0.0, 0.0]})
        bridge = self._bridge(rt, client)
        attended = bridge.pump()
        assert attended["visual"] is True
        assert attended["proprioceptive"] is True
        assert "acoustic" not in attended  # client has no audio this cycle

    def test_percepts_land_in_episodic_memory_as_observed(self):
        rt = _mk_runtime()
        client = _FakeIsaacClient(
            frame=_frame(50),
            joints={"positions": [0.0], "velocities": [1.0]})
        bridge = self._bridge(rt, client)
        bridge.pump()
        modalities = {(e.get("context") or {}).get("modality")
                     for e in rt.memory.all()}
        assert modalities == {"visual", "proprioceptive"}
        for e in rt.memory.all():
            assert (e.get("context") or {}).get("kind") == "observed"

    def test_content_vec_dim_matches_runtime_embed_dim(self):
        rt = _mk_runtime()
        client = _FakeIsaacClient(
            frame=_frame(90),
            joints={"positions": [0.3], "velocities": [0.1]})
        bridge = self._bridge(rt, client)
        bridge.pump()
        for e in rt.memory.all():
            assert len(e["content_vec"]) == rt.embed_dim()

    def test_audio_pumped_when_client_provides_it(self):
        rt = _mk_runtime()
        client = _FakeIsaacClient(
            frame=_frame(10), joints={"positions": [0.0], "velocities": [2.0]},
            audio=[0.1, -0.1, 0.2, -0.2] * 100)
        bridge = self._bridge(rt, client)
        attended = bridge.pump()
        assert attended["acoustic"] is True
        assert any((e.get("context") or {}).get("modality") == "acoustic"
                  for e in rt.memory.all())

    def test_missing_modality_is_skipped_not_crashed(self):
        rt = _mk_runtime()
        client = _FakeIsaacClient(frame=None, joints=None, audio=None)
        bridge = self._bridge(rt, client)
        attended = bridge.pump()
        assert attended == {}

    def test_sensor_failure_on_one_modality_does_not_crash_the_pump(self):
        rt = _mk_runtime()
        client = _FakeIsaacClient(
            fail_frame=True, joints={"positions": [0.5], "velocities": [0.0]})
        bridge = self._bridge(rt, client)
        attended = bridge.pump()
        assert "visual" not in attended
        assert attended["proprioceptive"] is True

    def test_habituated_repeat_frame_is_not_re_stored(self):
        rt = _mk_runtime()
        client = _FakeIsaacClient(frame=_frame(77), joints=None)
        bridge = self._bridge(rt, client)
        bridge.pump()
        attended = bridge.pump()  # identical frame again
        assert attended["visual"] is False
        assert len(rt.memory.all()) == 1

    def test_injected_cortices_are_used_over_defaults(self):
        rt = _mk_runtime()
        calls = []

        def fake_visual_cortex(frame):
            calls.append(frame)
            return [1.0] + [0.0] * (rt.embed_dim() - 1)

        client = _FakeIsaacClient(frame=_frame(5), joints=None)
        bridge = self._bridge(rt, client, visual_cortex=fake_visual_cortex)
        bridge.pump()
        assert len(calls) == 1


class TestOmniverseIsaacSimClientGuardedImport:
    def test_raises_a_clear_error_without_the_isaacsim_package(self):
        from neuroslm.connectors.isaac_sim import OmniverseIsaacSimClient
        with pytest.raises(RuntimeError, match="isaacsim"):
            OmniverseIsaacSimClient(camera_prim_path="/World/Camera",
                                    articulation_prim_path="/World/Robot")
