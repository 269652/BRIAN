"""World and self models (§14.16): two short texts the mind keeps about
where it is and who it is, rewritten by the trunk at event boundaries
from the old model + the evidence since, kept with their history, and
held as the global workspace's context frame — in every workspace
prompt and in the workspace's latent state."""
from __future__ import annotations

import pytest

from test_cognitive_runtime import _FakeNT, _mk_runtime, _vec_for


class _ModelGen:
    """Answers model-update prompts with a fixed model; thoughts
    otherwise. Records every prompt."""

    def __init__(self, world="I am by the river near the old mill.",
                 me="I am a curious walker, heading to the mill."):
        self.world, self.me = world, me
        self.prompts = []

    def __call__(self, prompt, max_new_tokens):
        self.prompts.append(prompt)
        if "about my surroundings" in prompt:
            return self.world
        if "about myself" in prompt:
            return self.me
        return "a passing thought"


def _rt(gen=None, **kw):
    from neuroslm.cognition.runtime import MindConfig
    gen = gen or _ModelGen()
    return gen, _mk_runtime(gen, cfg=MindConfig(n_candidates=1, recall_k=0,
                                                **kw), models=True)


class TestWorkspaceContextFrame:
    def test_context_enters_the_state_but_not_the_slots(self):
        from neuroslm.cognition.workspace import GlobalWorkspace
        ws = GlobalWorkspace(capacity=2)
        ws.offer("a thought", "thought", [1.0, 0.0], 1.0)
        before = ws.state_vector()
        ws.set_context("world", "by the river", [0.0, 1.0])
        after = ws.state_vector()
        assert after[1] > before[1], "the frame shifts the latent state"
        assert len(ws.items()) == 1
        assert ws.contexts() == {"world": "by the river"}

    def test_context_alone_is_a_state(self):
        from neuroslm.cognition.workspace import GlobalWorkspace
        ws = GlobalWorkspace(capacity=2)
        ws.set_context("self", "a walker", [0.0, 1.0])
        assert ws.state_vector() == pytest.approx([0.0, 1.0])


class TestModelsEvolve:
    def test_the_first_evidence_builds_both_models(self):
        gen, rt = _rt()
        rt.sense("The river, close ahead.", "sight")
        rt.tick()
        assert rt.world_model.text == gen.world
        assert rt.self_model.text == gen.me

    def test_the_update_prompt_holds_the_old_model_and_the_evidence(self):
        gen, rt = _rt()
        rt.sense("The river, close ahead.", "sight")
        rt.tick()
        rt.sense("An old mill, a few metres to the left.", "sight")
        rt.note_movement("approach:mill")
        rt.hierarchy.boundary = lambda *a, **k: True     # an event ends
        rt.tick()
        world_p = [p for p in gen.prompts if "about my surroundings" in p][-1]
        self_p = [p for p in gen.prompts if "about myself" in p][-1]
        assert gen.world in world_p, "the previous model"
        assert "An old mill" in world_p, "what was perceived since"
        assert gen.me in self_p
        assert "a passing thought" in self_p, "what I thought"
        assert "approach:mill" in self_p, "what I did"

    def test_the_body_state_reaches_the_self_model(self):
        gen, rt = _rt()
        rt.nt = _FakeNT(DA=0.40, GABA=0.15)
        rt.sense("The river, close ahead.", "sight")
        rt.tick()
        self_p = [p for p in gen.prompts if "about myself" in p][-1]
        assert "DA" in self_p and "above" in self_p

    def test_models_keep_their_history(self):
        gen, rt = _rt()
        rt.sense("The river, close ahead.", "sight")
        rt.tick()
        gen.world = "Now I stand at the mill door."
        rt.sense("A mill door, close.", "sight")
        rt.hierarchy.boundary = lambda *a, **k: True
        rt.tick()
        hist = [t for _, t in rt.world_model.history]
        assert hist[-2:] == ["I am by the river near the old mill.",
                             "Now I stand at the mill door."]

    def test_no_update_without_an_event_boundary(self):
        gen, rt = _rt()
        rt.sense("The river, close ahead.", "sight")
        rt.tick()
        n = len(gen.prompts)
        rt.hierarchy.boundary = lambda *a, **k: False
        rt.sense("Another tree.", "sight")
        rt.tick()
        assert not any("about my surroundings" in p
                       for p in gen.prompts[n:])

    def test_a_rambling_update_is_trimmed_to_whole_sentences(self):
        gen = _ModelGen(world="I am by the river. " * 60)
        _, rt = _rt(gen, model_max_chars=60)
        rt.sense("The river, close ahead.", "sight")
        rt.tick()
        t = rt.world_model.text
        assert len(t) <= 60 and t.endswith(".")


class TestModelsFrameTheWorkspace:
    def test_both_models_are_in_the_next_thought_prompt(self):
        gen, rt = _rt()
        rt.sense("The river, close ahead.", "sight")
        rt.tick()
        rt.tick()
        p = [p for p in gen.prompts
             if "about my" not in p][-1]
        assert f"Where I am: {gen.world}" in p
        assert f"Who I am: {gen.me}" in p

    def test_both_models_are_in_the_workspace_state(self):
        gen, rt = _rt()
        rt.sense("The river, close ahead.", "sight")
        rt.tick()
        assert set(rt.workspace.contexts()) == {"world", "self"}

    def test_switched_off_there_are_no_models(self):
        gen, rt = _rt(world_model=False, self_model=False)
        rt.sense("The river, close ahead.", "sight")
        rt.tick()
        assert rt.world_model is None and rt.self_model is None
        assert not any("about my" in p for p in gen.prompts)
