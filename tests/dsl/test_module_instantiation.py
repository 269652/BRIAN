# -*- coding: utf-8 -*-
"""Contracts for the ``module <name> = <Lib> { ... }`` instantiation syntax.

This is the macro-style hygienic expansion that makes
``lib/LanguageCortex.neuro`` reusable: the arch author writes one
line that supplies template parameters, and the compiler inlines the
lib's body with ``%key%`` references substituted from the call site.

Grammar
=======

In the lib::

    export module LanguageCortex {
        params: { teacher_warmup_steps: 10000, d_bottleneck: 512, ... }
        body: '''
            distillation cfd { method: "capacity_funneled",
                               temperature: %cfd_temperature%, ... }
            funnel ensemble { inputs: %experts%, target: lm_trunk,
                              d_bottleneck: %d_bottleneck% }
            warmup tc { target: ensemble, action: "detach",
                        rules: [ { metric: "step", op: ">=",
                                   value: %teacher_warmup_steps% } ] }
        '''
    }

In the arch::

    import { LanguageCortex } from "@brian/LanguageCortex"

    module brain = LanguageCortex {
        teacher_warmup_steps: 10000,
        d_bottleneck:         512,
        cfd_temperature:      4.0,
    }

Compilation
===========

The compiler expands the ``module brain = LanguageCortex { ... }``
block into the lib's body with ``%key%`` substitution applied. The
expanded text is then re-parsed as if it had been inline. Each
expanded block is tagged with the instance name (``brain``) so
multiple instantiations of the same lib don't collide.
"""
from __future__ import annotations

import pytest


# ─────────────────────────────────────────────────────────────────────
# Contract A — parser produces a ModuleInstanceIR
# ─────────────────────────────────────────────────────────────────────


class TestModuleInstantiationParses:

    def test_basic_instantiation_yields_module_instance_ir(self):
        from neuroslm.dsl.compiler import NeuroMLCompiler
        src = (
            "architecture toy { d_sem: 64 }\n"
            "population p1 { count: 16 }\n"
            "module brain = LanguageCortex {\n"
            "    teacher_warmup_steps: 10000,\n"
            "    d_bottleneck:         512,\n"
            "    cfd_temperature:      4.0\n"
            "}\n"
        )
        prog = NeuroMLCompiler.compile(src)
        assert hasattr(prog, "module_instances")
        m = prog.module_instances[0]
        assert m.name == "brain"
        assert m.lib == "LanguageCortex"
        assert m.params["teacher_warmup_steps"] == 10000
        assert m.params["d_bottleneck"] == 512
        assert m.params["cfd_temperature"] == 4.0


# ─────────────────────────────────────────────────────────────────────
# Contract B — params support both numeric and list values
# ─────────────────────────────────────────────────────────────────────


class TestParamShapes:

    def test_list_param_preserved_as_list(self):
        from neuroslm.dsl.compiler import NeuroMLCompiler
        src = (
            "architecture toy { d_sem: 64 }\n"
            "population p1 { count: 16 }\n"
            "expert A { model: \"gpt2\",       role: \"math\" }\n"
            "expert B { model: \"distilgpt2\", role: \"code\" }\n"
            "module brain = LanguageCortex {\n"
            "    teacher_warmup_steps: 10000,\n"
            "    experts:              [A, B]\n"
            "}\n"
        )
        prog = NeuroMLCompiler.compile(src)
        m = prog.module_instances[0]
        assert m.params["experts"] == ["A", "B"], (
            "list params (e.g. `experts: [A, B]`) must round-trip as a "
            "Python list; the lib expander uses this for `%experts%`")
