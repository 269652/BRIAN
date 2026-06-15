# -*- coding: utf-8 -*-
"""Contracts for the ``funnel`` DSL block.

A funnel wires N declared experts to a single trunk population via a
capacity-bottlenecked projection + a gating mechanism + an optional
distillation method. It's the "bowtie waist" of the
expert-ensemble → LM-trunk pathway::

    funnel ensemble_funnel {
        inputs:        [MathExpert, CodeExpert, LangExpert],
        target:        lm_trunk,
        d_bottleneck:  512,
        gate:          "softmax_router",
        method:        cfd_method,
    }

Compile-time validation
=======================

* ``inputs`` must be a non-empty list of references to declared
  ``expert`` blocks. Unknown name ⇒ :class:`NeuroMLError`.
* ``target`` must reference a declared ``population``. Unknown
  name ⇒ :class:`NeuroMLError` (catches typos before training).
* ``gate`` ∈ {``mean``, ``topk2``, ``softmax_router``, ``attention``}.
* ``method`` (optional) must reference a declared ``distillation``
  block. Unknown name ⇒ :class:`NeuroMLError`.

These contracts protect the wiring graph: a funnel that survives
``NeuroMLCompiler.compile`` has its inputs, target, and distillation
method all guaranteed to resolve at runtime.
"""
from __future__ import annotations

import pytest


# ─────────────────────────────────────────────────────────────────────
# Contract A — parsing
# ─────────────────────────────────────────────────────────────────────


class TestFunnelBlockParses:

    def test_minimum_funnel_block(self):
        from neuroslm.dsl.compiler import NeuroMLCompiler
        src = (
            "architecture toy { d_sem: 64 }\n"
            "population lm_trunk { count: 64 }\n"
            "expert MathExpert { model: \"gpt2\", role: \"math\" }\n"
            "funnel ensemble {\n"
            "    inputs:       [MathExpert],\n"
            "    target:       lm_trunk,\n"
            "    d_bottleneck: 256,\n"
            "    gate:         \"mean\"\n"
            "}\n"
        )
        prog = NeuroMLCompiler.compile(src)
        assert hasattr(prog, "funnels")
        f = prog.funnels[0]
        assert f.name == "ensemble"
        assert f.inputs == ["MathExpert"]
        assert f.target == "lm_trunk"
        assert f.d_bottleneck == 256
        assert f.gate == "mean"
        assert f.method_ref == "", "no method given ⇒ empty ref"

    def test_three_input_funnel_with_method(self):
        from neuroslm.dsl.compiler import NeuroMLCompiler
        src = (
            "architecture toy { d_sem: 64 }\n"
            "population lm_trunk { count: 64 }\n"
            "expert MathExpert { model: \"gpt2\",        role: \"math\" }\n"
            "expert CodeExpert { model: \"distilgpt2\",  role: \"code\" }\n"
            "expert LangExpert { model: \"gpt2-medium\", role: \"general\" }\n"
            "distillation cfd {\n"
            "    method: \"capacity_funneled\", temperature: 4.0, alpha: 0.7\n"
            "}\n"
            "funnel ensemble {\n"
            "    inputs:       [MathExpert, CodeExpert, LangExpert],\n"
            "    target:       lm_trunk,\n"
            "    d_bottleneck: 512,\n"
            "    gate:         \"softmax_router\",\n"
            "    method:       cfd\n"
            "}\n"
        )
        prog = NeuroMLCompiler.compile(src)
        f = prog.funnels[0]
        assert f.inputs == ["MathExpert", "CodeExpert", "LangExpert"]
        assert f.gate == "softmax_router"
        assert f.method_ref == "cfd"


# ─────────────────────────────────────────────────────────────────────
# Contract B — reference resolution
# ─────────────────────────────────────────────────────────────────────


class TestReferenceResolution:

    def test_unknown_input_expert_raises(self):
        from neuroslm.dsl.compiler import NeuroMLCompiler, NeuroMLError
        src = (
            "architecture toy { d_sem: 64 }\n"
            "population lm_trunk { count: 64 }\n"
            "expert MathExpert { model: \"gpt2\", role: \"math\" }\n"
            "funnel ensemble {\n"
            "    inputs: [MathExpert, UnknownExpert],\n"
            "    target: lm_trunk\n"
            "}\n"
        )
        with pytest.raises(NeuroMLError, match="UnknownExpert"):
            NeuroMLCompiler.compile(src)

    def test_unknown_target_population_raises(self):
        from neuroslm.dsl.compiler import NeuroMLCompiler, NeuroMLError
        src = (
            "architecture toy { d_sem: 64 }\n"
            "population p1 { count: 64 }\n"
            "expert MathExpert { model: \"gpt2\", role: \"math\" }\n"
            "funnel ensemble {\n"
            "    inputs: [MathExpert],\n"
            "    target: nonexistent_trunk\n"
            "}\n"
        )
        with pytest.raises(NeuroMLError, match="nonexistent_trunk"):
            NeuroMLCompiler.compile(src)

    def test_unknown_method_raises(self):
        from neuroslm.dsl.compiler import NeuroMLCompiler, NeuroMLError
        src = (
            "architecture toy { d_sem: 64 }\n"
            "population lm_trunk { count: 64 }\n"
            "expert MathExpert { model: \"gpt2\", role: \"math\" }\n"
            "funnel ensemble {\n"
            "    inputs: [MathExpert],\n"
            "    target: lm_trunk,\n"
            "    method: nonexistent_method\n"
            "}\n"
        )
        with pytest.raises(NeuroMLError, match="nonexistent_method"):
            NeuroMLCompiler.compile(src)


# ─────────────────────────────────────────────────────────────────────
# Contract C — gate enum validation
# ─────────────────────────────────────────────────────────────────────


class TestGateValidation:

    def test_unknown_gate_raises(self):
        from neuroslm.dsl.compiler import NeuroMLCompiler, NeuroMLError
        src = (
            "architecture toy { d_sem: 64 }\n"
            "population lm_trunk { count: 64 }\n"
            "expert E { model: \"gpt2\", role: \"math\" }\n"
            "funnel ensemble {\n"
            "    inputs: [E], target: lm_trunk, gate: \"telekinesis\"\n"
            "}\n"
        )
        with pytest.raises(NeuroMLError, match=r"gate|telekinesis"):
            NeuroMLCompiler.compile(src)
