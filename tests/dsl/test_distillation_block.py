# -*- coding: utf-8 -*-
"""Contracts for the ``distillation`` DSL block.

Lifts the CFD (Capacity-Funneled Distillation) hyperparameters from
flat ``cfd_*`` knobs inside ``multi_cortex { ... }`` into a named,
referenceable block::

    distillation cfd_method {
        method:             "capacity_funneled",
        temperature:        4.0,
        alpha:              0.7,
        bottleneck:         512,
        loss:               "kl_div",
        topk_start:         4,
        topk_end:           32,
        topk_anneal_steps:  10000,
        temperature_floor:  1.0,
    }

A ``funnel`` block then references the method by name
(``method: cfd_method``). This decouples *which* method is in use
from *how* it's parameterised, so a future arch can have multiple
distillation strategies declared and pick one per funnel.

Why a new block (vs a feature with equation)
--------------------------------------------

* ``feature`` requires a backing ``equation`` declaration; CFD has
  no clean closed-form formula string (it's a 3-stage pipeline).
* ``feature.params`` is a flat dict; CFD has structured nested
  validation (topk_start < topk_end, anneal_steps > 0, temperature > 0).
* CFD parameters appear in trainer log lines and metric names; a
  named block gives a stable handle (``cfd_method``) that
  surfaces in logs as the citation key.
"""
from __future__ import annotations

import pytest


# ─────────────────────────────────────────────────────────────────────
# Contract A — parsing
# ─────────────────────────────────────────────────────────────────────


class TestDistillationBlockParses:

    def test_capacity_funneled_block_parses(self):
        from neuroslm.dsl.compiler import NeuroMLCompiler
        src = (
            "architecture toy { d_sem: 64 }\n"
            "population p1 { count: 16 }\n"
            "distillation cfd {\n"
            "    method:            \"capacity_funneled\",\n"
            "    temperature:       4.0,\n"
            "    alpha:             0.7,\n"
            "    bottleneck:        512,\n"
            "    loss:              \"kl_div\",\n"
            "    topk_start:        4,\n"
            "    topk_end:          32,\n"
            "    topk_anneal_steps: 10000,\n"
            "    temperature_floor: 1.0\n"
            "}\n"
        )
        prog = NeuroMLCompiler.compile(src)
        assert hasattr(prog, "distillations"), (
            "ProgramIR must have a `distillations: List[DistillationIR]` "
            "field for funnel blocks to reference by name")
        assert len(prog.distillations) == 1
        d = prog.distillations[0]
        assert d.name == "cfd"
        assert d.method == "capacity_funneled"
        assert d.temperature == 4.0
        assert d.alpha == 0.7
        assert d.bottleneck == 512
        assert d.loss == "kl_div"
        assert d.topk_start == 4
        assert d.topk_end == 32
        assert d.topk_anneal_steps == 10000
        assert d.temperature_floor == 1.0

    def test_vanilla_kd_minimum_block(self):
        from neuroslm.dsl.compiler import NeuroMLCompiler
        src = (
            "architecture toy { d_sem: 64 }\n"
            "population p1 { count: 16 }\n"
            "distillation kd {\n"
            "    method:      \"vanilla_kd\",\n"
            "    temperature: 4.0,\n"
            "    alpha:       0.5\n"
            "}\n"
        )
        prog = NeuroMLCompiler.compile(src)
        d = prog.distillations[0]
        assert d.method == "vanilla_kd"
        # CFD-specific fields take their defaults when unused
        assert d.bottleneck == 0
        assert d.topk_start == 4


# ─────────────────────────────────────────────────────────────────────
# Contract B — method enum + numeric range validation
# ─────────────────────────────────────────────────────────────────────


class TestValidation:

    def test_unknown_method_raises(self):
        from neuroslm.dsl.compiler import NeuroMLCompiler, NeuroMLError
        src = (
            "architecture toy { d_sem: 64 }\n"
            "population p1 { count: 16 }\n"
            "distillation bad { method: \"telepathy\", temperature: 4.0, alpha: 0.5 }\n"
        )
        with pytest.raises(NeuroMLError, match=r"method|telepathy"):
            NeuroMLCompiler.compile(src)

    def test_temperature_must_be_positive(self):
        from neuroslm.dsl.compiler import NeuroMLCompiler, NeuroMLError
        src = (
            "architecture toy { d_sem: 64 }\n"
            "population p1 { count: 16 }\n"
            "distillation bad { method: \"vanilla_kd\", temperature: 0.0, alpha: 0.5 }\n"
        )
        with pytest.raises(NeuroMLError, match="temperature"):
            NeuroMLCompiler.compile(src)

    def test_alpha_must_be_in_unit_interval(self):
        from neuroslm.dsl.compiler import NeuroMLCompiler, NeuroMLError
        src = (
            "architecture toy { d_sem: 64 }\n"
            "population p1 { count: 16 }\n"
            "distillation bad { method: \"vanilla_kd\", temperature: 4.0, alpha: 1.5 }\n"
        )
        with pytest.raises(NeuroMLError, match="alpha"):
            NeuroMLCompiler.compile(src)

    def test_topk_start_must_be_le_topk_end(self):
        from neuroslm.dsl.compiler import NeuroMLCompiler, NeuroMLError
        src = (
            "architecture toy { d_sem: 64 }\n"
            "population p1 { count: 16 }\n"
            "distillation bad {\n"
            "    method:     \"capacity_funneled\",\n"
            "    temperature:4.0,\n"
            "    alpha:      0.5,\n"
            "    topk_start: 32,\n"
            "    topk_end:   4\n"
            "}\n"
        )
        with pytest.raises(NeuroMLError, match="topk"):
            NeuroMLCompiler.compile(src)
