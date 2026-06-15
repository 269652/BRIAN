# -*- coding: utf-8 -*-
"""Contracts for the ``expert`` DSL block.

Motivation
==========

The current expert roster lives inside ``multi_cortex { experts: [...] }``
— a list of dicts inside a string-keyed config block. That format
makes per-expert config a second-class citizen:

* Adding a per-expert field (``pool``, ``device``, ``dtype``,
  ``cache``, …) requires editing both the dict literal AND a string
  ``_parse_experts_list`` switch.
* Multiple-expert references in other DSL blocks (e.g.
  ``funnel { inputs: [MathExpert, CodeExpert] }``) have nothing to
  refer to — the experts are unnamed list rows.
* The arch.neuro author can't browse a single canonical list of
  declared experts via ``grep '^expert '``.

The new ``expert`` block makes the expert a first-class declaration,
mirroring how ``equation`` and ``feature`` are top-level keywords::

    expert MathExpert {
        model:  "Qwen/Qwen2.5-Math-7B-Instruct",
        role:   "math",
        d_out:  4096,
        frozen: true,
        pool:   "last_token",
    }

Grammar
=======

::

    expert <name> {
        model:      "<hf_id_or_alias>"      # required
        role:       "<routing_role>"        # required
        d_out:      <int>                   # optional, default 0 = auto
        frozen:     true | false            # optional, default true
        dtype:      "float32"|"float16"|"bfloat16"  # optional
        device:     "<torch_device>"        # optional, default trunk
        pool:       "last_token"|"mean"|"cls"  # optional, default "last_token"
        cache:      "<path>"                # optional, supports %key% and $ENV
        auth_token: $TOKEN                  # optional, env-only
    }

Compiled IR
===========

Each block becomes an :class:`ExpertIR` on
:attr:`ProgramIR.experts`. Consumers (the new ``funnel`` block and
the runtime expert loader) reference experts by name. Compile-time
validation:

* ``model`` is mandatory and non-empty.
* ``role`` is mandatory and non-empty.
* Duplicate ``expert <same_name>`` blocks raise :class:`NeuroMLError`.
* Duplicate ``role`` across experts in the same arch raises (mirrors
  ``_parse_experts_list``'s ``duplicate domain`` check — each routing
  role must be unique).
"""
from __future__ import annotations

from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parent.parent.parent


# ─────────────────────────────────────────────────────────────────────
# Contract A — parsing
# ─────────────────────────────────────────────────────────────────────


class TestExpertBlockParses:

    def test_minimum_block_yields_expert_ir(self):
        from neuroslm.dsl.compiler import NeuroMLCompiler
        src = (
            "architecture toy { d_sem: 64 }\n"
            "population p1 { count: 16 }\n"
            "expert MathExpert { model: \"Qwen/Qwen2.5-Math-7B\", "
            "role: \"math\" }\n"
        )
        prog = NeuroMLCompiler.compile(src)
        assert hasattr(prog, "experts"), (
            "ProgramIR must have a `experts: List[ExpertIR]` field "
            "so consumers (funnel, warmup) can resolve refs by name")
        names = [e.name for e in prog.experts]
        assert "MathExpert" in names, f"got experts {names!r}"
        e = next(e for e in prog.experts if e.name == "MathExpert")
        assert e.model == "Qwen/Qwen2.5-Math-7B"
        assert e.role == "math"
        # Defaults
        assert e.frozen is True, "frozen defaults to True (pretrained backbone)"
        assert e.pool == "last_token"
        assert e.d_out == 0, "d_out=0 means auto-detect from HF config"

    def test_full_block_round_trips_all_fields(self):
        from neuroslm.dsl.compiler import NeuroMLCompiler
        src = (
            "architecture toy { d_sem: 64 }\n"
            "population p1 { count: 16 }\n"
            "expert MathExpert {\n"
            "    model:  \"Qwen/Qwen2.5-Math-7B\",\n"
            "    role:   \"math\",\n"
            "    d_out:  4096,\n"
            "    frozen: true,\n"
            "    dtype:  \"bfloat16\",\n"
            "    device: \"cuda:0\",\n"
            "    pool:   \"mean\"\n"
            "}\n"
        )
        prog = NeuroMLCompiler.compile(src)
        e = next(e for e in prog.experts if e.name == "MathExpert")
        assert e.d_out == 4096
        assert e.dtype == "bfloat16"
        assert e.device == "cuda:0"
        assert e.pool == "mean"


# ─────────────────────────────────────────────────────────────────────
# Contract B — required-field validation
# ─────────────────────────────────────────────────────────────────────


class TestRequiredFields:

    def test_missing_model_field_raises(self):
        from neuroslm.dsl.compiler import NeuroMLCompiler, NeuroMLError
        src = (
            "architecture toy { d_sem: 64 }\n"
            "population p1 { count: 16 }\n"
            "expert NoModel { role: \"math\" }\n"
        )
        with pytest.raises(NeuroMLError, match="model"):
            NeuroMLCompiler.compile(src)

    def test_missing_role_field_raises(self):
        from neuroslm.dsl.compiler import NeuroMLCompiler, NeuroMLError
        src = (
            "architecture toy { d_sem: 64 }\n"
            "population p1 { count: 16 }\n"
            "expert NoRole { model: \"gpt2\" }\n"
        )
        with pytest.raises(NeuroMLError, match="role"):
            NeuroMLCompiler.compile(src)


# ─────────────────────────────────────────────────────────────────────
# Contract C — uniqueness
# ─────────────────────────────────────────────────────────────────────


class TestUniqueness:

    def test_duplicate_role_raises(self):
        from neuroslm.dsl.compiler import NeuroMLCompiler, NeuroMLError
        src = (
            "architecture toy { d_sem: 64 }\n"
            "population p1 { count: 16 }\n"
            "expert A { model: \"gpt2\", role: \"math\" }\n"
            "expert B { model: \"gpt2-medium\", role: \"math\" }\n"
        )
        with pytest.raises(NeuroMLError, match=r"duplicate.*role|role.*math"):
            NeuroMLCompiler.compile(src)


# ─────────────────────────────────────────────────────────────────────
# Contract D — multiple experts coexist
# ─────────────────────────────────────────────────────────────────────


class TestMultipleExperts:

    def test_three_experts_all_present(self):
        from neuroslm.dsl.compiler import NeuroMLCompiler
        src = (
            "architecture toy { d_sem: 64 }\n"
            "population p1 { count: 16 }\n"
            "expert MathExpert { model: \"gpt2\",        role: \"math\" }\n"
            "expert CodeExpert { model: \"distilgpt2\",  role: \"code\" }\n"
            "expert LangExpert { model: \"gpt2-medium\", role: \"general\" }\n"
        )
        prog = NeuroMLCompiler.compile(src)
        names = sorted(e.name for e in prog.experts)
        assert names == ["CodeExpert", "LangExpert", "MathExpert"]
