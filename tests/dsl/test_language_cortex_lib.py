# -*- coding: utf-8 -*-
"""End-to-end smoke test for the bundled ``LanguageCortex`` lib module.

This test exercises the full path:

1. ``import { LanguageCortex } from "@brian/LanguageCortex"`` resolves
   to the file under ``architectures/lib/`` (PathResolver scope).
2. The ``module brain = LanguageCortex { ... }`` instantiation block
   is parsed and its template body is expanded with ``%key%``
   substitution.
3. The expanded body produces ``ExpertIR`` (from the arch's expert
   blocks), ``DistillationIR``, ``FunnelIR`` and ``WarmupIR`` rows on
   the final ProgramIR.

The test pins the full LanguageCortex public surface so future
refactors of the lib are forced to keep the same parameter contract.
"""
from __future__ import annotations

from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parent.parent.parent


def test_language_cortex_lib_file_exists():
    """The lib file must live where ``@lib/LanguageCortex`` resolves.

    PathResolver anchors ``@lib/`` at ``<repo>/lib/<rest>.neuro``
    (2026-06-15 layout — lib hoisted from ``architectures/lib/`` up
    to the repo root). We also keep a side-copy at
    ``.neuro/lib/LanguageCortex.neuro`` for workspace parity (the
    unfolded-DNA flow), but the canonical source is ``<repo>/lib/``.
    """
    canonical = REPO_ROOT / "lib" / "LanguageCortex.neuro"
    assert canonical.is_file(), (
        f"LanguageCortex lib must live at {canonical} "
        "(the @lib/ specifier resolves to <repo>/lib/)"
    )


def test_language_cortex_module_instantiation_produces_full_ir():
    """An arch.neuro that instantiates LanguageCortex with 3 experts +
    teacher_warmup_steps=10000 must produce 1 distillation + 1 funnel
    + 1 warmup IR row, all referencing the supplied experts."""
    from neuroslm.dsl.compiler import NeuroMLCompiler

    src = (
        "architecture lc_smoke { d_sem: 64 }\n"
        "population pfc { count: 64 }\n"
        "\n"
        "expert MathExpert { model: \"gpt2\",        role: \"math\"    }\n"
        "expert CodeExpert { model: \"distilgpt2\",  role: \"code\"    }\n"
        "expert LangExpert { model: \"gpt2-medium\", role: \"general\" }\n"
        "\n"
        "module brain = LanguageCortex {\n"
        "    teacher_warmup_steps: 10000,\n"
        "    d_bottleneck:         512,\n"
        "    cfd_temperature:      4.0,\n"
        "    cfd_alpha:            0.7,\n"
        "    experts:              [MathExpert, CodeExpert, LangExpert]\n"
        "}\n"
    )
    prog = NeuroMLCompiler.compile_with_lib(src)
    # 3 expert blocks
    assert len(prog.experts) == 3
    # 1 distillation block expanded from the lib
    assert len(prog.distillations) == 1
    d = prog.distillations[0]
    assert d.temperature == 4.0
    assert d.alpha == 0.7
    assert d.method == "capacity_funneled"
    # 1 funnel block with all 3 experts wired to the target.
    # The lib's default `target:` is `pfc` (the LM-trunk population
    # in the canonical bowtie). Pre-2026-06-15 the default was
    # `lm_trunk` which never appeared in any population roster --
    # see tests/test_arch_lm_trunk_wiring.py (Fix B) and the
    # 2026-06-15 NFG cross-alignment audit.
    assert len(prog.funnels) == 1
    f = prog.funnels[0]
    assert set(f.inputs) == {"MathExpert", "CodeExpert", "LangExpert"}
    assert f.target == "pfc"
    assert f.d_bottleneck == 512
    # 1 warmup block with the 10000-step rule
    assert len(prog.warmups) == 1
    w = prog.warmups[0]
    step_rule = next(r for r in w.rules if r.metric == "step")
    assert step_rule.value == 10000
    assert step_rule.op == ">="
    assert w.target == f.name, "warmup must target the funnel by name"


def test_language_cortex_with_default_temperature():
    """When the call site omits ``cfd_temperature``, the lib's default
    value is used. This is the whole point of ``params: { ... }`` in
    the lib — sensible defaults that the arch only overrides when it
    needs to."""
    from neuroslm.dsl.compiler import NeuroMLCompiler

    src = (
        "architecture lc_defaults { d_sem: 64 }\n"
        "population pfc { count: 64 }\n"
        "expert MathExpert { model: \"gpt2\", role: \"math\" }\n"
        "\n"
        "module brain = LanguageCortex {\n"
        "    teacher_warmup_steps: 10000,\n"
        "    experts:              [MathExpert]\n"
        "}\n"
    )
    prog = NeuroMLCompiler.compile_with_lib(src)
    d = prog.distillations[0]
    # The lib default for cfd_temperature is 4.0 (canonical CFD T0).
    assert d.temperature == 4.0
