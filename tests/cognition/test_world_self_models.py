"""World and self models (§14.16): two short texts the mind keeps about
where it is and who it is, rewritten by the trunk at event boundaries
from the old model + the evidence since, kept with their history, and
held as the global workspace's context frame — in every workspace
prompt and in the workspace's latent state."""
from __future__ import annotations

import pytest

from test_cognitive_runtime import _FakeNT, _mk_runtime, _vec_for


_W, _S = "Where I am now: I am", "Who I am now: I am"


class _ModelGen:
    """Completes model-update stems with a fixed continuation; thoughts
    otherwise. Records every prompt. ``world``/``me`` are the resulting
    models (stem's "I am" + continuation)."""

    def __init__(self, world="I am by the river near the old mill.",
                 me="I am a curious walker, heading to the mill."):
        self.world, self.me = world, me
        self.prompts = []

    def __call__(self, prompt, max_new_tokens):
        self.prompts.append(prompt)
        if prompt.endswith(_W):
            return self.world[len("I am"):]
        if prompt.endswith(_S):
            return self.me[len("I am"):]
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

    def test_the_update_prompts_hold_their_evidence(self):
        gen, rt = _rt()
        rt.sense("The river, close ahead.", "sight")
        rt.tick()
        rt.sense("An old mill, a few metres to the left.", "sight")
        rt.note_movement("approach:mill")
        rt.hierarchy.boundary = lambda *a, **k: True     # an event ends
        rt.tick()
        world_p = [p for p in gen.prompts if p.endswith(_W)][-1]
        self_p = [p for p in gen.prompts if p.endswith(_S)][-1]
        assert "The river, close ahead." in world_p, "seen earlier"
        assert "An old mill" in world_p, "seen since"
        assert gen.me in self_p, "the self model carries on"
        assert "a passing thought" in self_p, "what I thought"
        assert "approach:mill" in self_p, "what I did"


class TestWorldModelIsGroundedInWhatWasSeen:
    """Run 59313101: from "Earlier I thought: …" the world model
    invented a crossroads with white signposts — never perceived — and
    every later version built on it. The world model is now rebuilt
    from the registry of what was actually perceived (one entry per
    object, its latest percept), never from its own previous text."""

    def _second_world_prompt(self, gen, rt):
        rt.hierarchy.boundary = lambda *a, **k: True
        rt.tick()
        return [p for p in gen.prompts if p.endswith(_W)][-1]

    def test_the_previous_world_model_is_not_fed_back(self):
        gen = _ModelGen(world="I am at a crossroads with white signposts.")
        _, rt = _rt(gen)
        rt.sense("The river, close ahead.", "sight", key="river")
        rt.tick()
        rt.sense("A brown dog, close.", "sight", key="dog")
        p = self._second_world_prompt(gen, rt)
        assert "crossroads" not in p
        assert "The river, close ahead." in p and "A brown dog" in p

    def test_one_entry_per_object_its_latest_percept(self):
        gen, rt = _rt()
        rt.sense("A brown dog, far off ahead.", "sight", key="dog")
        rt.tick()
        rt.sense("A brown dog, close to the left.", "sight", key="dog")
        p = self._second_world_prompt(gen, rt)
        assert "close to the left" in p and "far off ahead" not in p

    def test_unkeyed_percepts_are_their_own_entries(self):
        gen, rt = _rt()
        rt.sense("Rain begins to fall.", "sound")
        rt.tick()
        p = self._second_world_prompt(gen, rt)
        assert "Rain begins to fall." in p

    def test_the_persona_line_is_not_in_model_prompts(self):
        # parroted back as the self model ("I am Brian, an always-on mind…")
        from neuroslm.cognition.runtime import MindConfig
        gen, rt = _rt()
        rt.sense("The river, close ahead.", "sight")
        rt.tick()
        persona = MindConfig().persona
        assert all(persona not in p for p in gen.prompts
                   if p.endswith((_W, _S)))

    def test_the_body_state_reaches_the_self_model(self):
        gen, rt = _rt()
        rt.nt = _FakeNT(DA=0.40, GABA=0.15)
        rt.sense("The river, close ahead.", "sight")
        rt.tick()
        self_p = [p for p in gen.prompts if p.endswith(_S)][-1]
        assert "DA" in self_p and "above" in self_p

    def test_models_keep_their_history(self):
        gen, rt = _rt()
        rt.sense("The river, close ahead.", "sight")
        rt.tick()
        gen.world = "I am at the mill door now."
        rt.sense("A mill door, close.", "sight")
        rt.hierarchy.boundary = lambda *a, **k: True
        rt.tick()
        hist = [t for _, t in rt.world_model.history]
        assert hist[-2:] == ["I am by the river near the old mill.",
                             "I am at the mill door now."]

    def test_no_update_without_an_event_boundary(self):
        gen, rt = _rt()
        rt.sense("The river, close ahead.", "sight")
        rt.tick()
        n = len(gen.prompts)
        rt.hierarchy.boundary = lambda *a, **k: False
        rt.sense("Another tree.", "sight")
        rt.tick()
        assert not any(p.endswith(_W)
                       for p in gen.prompts[n:])

    def test_a_rambling_update_is_trimmed_to_whole_sentences(self):
        gen = _ModelGen(world="I am by the river. " * 60)
        _, rt = _rt(gen, model_max_chars=60)
        rt.sense("The river, close ahead.", "sight")
        rt.tick()
        t = rt.world_model.text
        assert len(t) <= 60 and t.endswith(".")


class TestUpdatesAreContinuationsNotTasks:
    """Run 59311386: with bullet lists and "In a few sentences, what I
    now believe…" R1 treated the update as a writing exercise ("I need
    to write a story… beginning, middle and end", "**Title:**") and the
    story frame leaked into its thoughts via the context frame. The
    update is a first-person continuation: no lists, no instruction,
    ending on a stem the trunk simply completes."""

    def _prompts(self):
        gen, rt = _rt()
        rt.sense("The river, close ahead.", "sight")
        rt.sense("A brown dog, a few metres to the right.", "sight")
        rt.note_movement("head:0")
        rt.tick()
        w = [p for p in gen.prompts if p.endswith(_W)][-1]
        s = [p for p in gen.prompts if p.endswith(_S)][-1]
        return w, s

    def test_no_lists_and_no_instructions(self):
        for p in self._prompts():
            assert "\n- " not in p
            assert "In a few sentences" not in p and "what I now" not in p

    def test_the_evidence_reads_as_first_person_prose(self):
        w, s = self._prompts()
        assert "The river, close ahead.; A brown dog" not in w
        assert "The river, close ahead." in w and "A brown dog" in w
        assert "head:0" in s

    def test_the_model_is_the_stem_plus_its_completion(self):
        gen, rt = _rt()
        rt.sense("The river, close ahead.", "sight")
        rt.tick()
        assert rt.world_model.text == "I am by the river near the old mill."

    def test_only_the_first_paragraph_is_kept(self):
        gen = _ModelGen(world="I am in a park by a fountain.\n\nWait, so I "
                              "have to write a story about this.")
        _, rt = _rt(gen)
        rt.sense("A fountain, close.", "sight")
        rt.tick()
        assert rt.world_model.text == "I am in a park by a fountain."


class TestModelsFrameTheWorkspace:
    def test_both_models_are_in_the_next_thought_prompt(self):
        gen, rt = _rt()
        rt.sense("The river, close ahead.", "sight")
        rt.tick()
        rt.tick()
        p = [p for p in gen.prompts
             if not p.endswith((_W, _S))][-1]
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
        assert not any(p.endswith((_W, _S)) for p in gen.prompts)
