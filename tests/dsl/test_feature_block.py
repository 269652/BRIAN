# -*- coding: utf-8 -*-
"""TDD: ``feature`` DSL block — toggleable mechanisms with equation refs.

Motivation
----------
Per the user's 2026-06-12 design: we want to wire ablation-capable
mechanisms into an arch.neuro without inlining their math. Equations
live in ``lib/equations.neuro`` (already a supported pattern via
``export equation``). A ``feature`` block names a mechanism, points
at its equation by name, and carries an ``active`` toggle so a single
arch can be A/B'd by flipping one flag.

Grammar
-------
::

    feature <name> {
        equation: <equation_name>      # must reference an exported equation
        active:   true | false         # required
        params:   { k: v, ... }        # optional per-feature parameters
    }

Compiled IR
-----------
Each ``feature`` block becomes a :class:`FeatureIR` on the
:class:`ProgramIR`. Consumers (model builder, NFG renderer, harness)
read ``program.features`` and either wire the mechanism in (when
``active=True``) or skip it (when ``active=False``).

Why a new block vs. reusing ``equation``
----------------------------------------
- ``equation`` is a pure math defn; it has no notion of being on/off.
- A mechanism (e.g. Tonnetz RoPE) needs both the equation AND a
  runtime gate. Putting the gate on the equation would conflate
  "what the math is" with "do we use it here?".
- Ablation runs need the gate to be a single-line edit in the arch,
  not a re-edit of the math file.

Contracts pinned here
---------------------
A. ``feature foo { equation: bar; active: true }`` parses to a
   ``FeatureIR(name='foo', equation_ref='bar', active=True)``.
B. ``active: false`` parses as ``active=False`` (Python bool, not str).
C. Missing ``active`` field raises ``NeuroMLError`` (the gate is
   mandatory — the whole point of the block).
D. Missing ``equation`` field raises ``NeuroMLError``.
E. A reference to a non-existent equation is reported at compile time
   (catches typos before training; "fail fast" before vast.ai cost).
F. Optional ``params`` block parses as a dict.
"""
from __future__ import annotations

from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parent.parent.parent


# ─────────────────────────────────────────────────────────────────────
# Contract A — parsing the minimum valid block
# ─────────────────────────────────────────────────────────────────────


class TestFeatureBlockParses:

    def test_minimum_block_yields_feature_ir(self):
        from neuroslm.dsl.compiler import NeuroMLCompiler
        src = (
            "architecture toy { d_sem: 64 }\n"
            "population p1 { count: 16 }\n"
            "export equation eq_foo { params: [x], formula: \"y = x\" }\n"
            "feature feat_foo { equation: eq_foo, active: true }\n"
        )
        prog = NeuroMLCompiler.compile(src)
        assert hasattr(prog, "features"), (
            "ProgramIR must have a `features: List[FeatureIR]` field — "
            "the new block has no place to land otherwise.")
        names = [f.name for f in prog.features]
        assert "feat_foo" in names, f"got features {names!r}"
        f = next(f for f in prog.features if f.name == "feat_foo")
        assert f.equation_ref == "eq_foo", (
            f"equation_ref must hold the referenced equation name; "
            f"got {f.equation_ref!r}")
        assert f.active is True, (
            f"active must be a Python bool; got {f.active!r} of type "
            f"{type(f.active).__name__}")

    def test_inactive_block_parses_active_false(self):
        from neuroslm.dsl.compiler import NeuroMLCompiler
        src = (
            "architecture toy { d_sem: 64 }\n"
            "population p1 { count: 16 }\n"
            "export equation eq_foo { params: [x], formula: \"y = x\" }\n"
            "feature feat_foo { equation: eq_foo, active: false }\n"
        )
        prog = NeuroMLCompiler.compile(src)
        f = next(f for f in prog.features if f.name == "feat_foo")
        assert f.active is False, (
            f"active=false must parse to Python False, not 'false' "
            f"string or 0; got {f.active!r}")


# ─────────────────────────────────────────────────────────────────────
# Contract C / D — mandatory fields
# ─────────────────────────────────────────────────────────────────────


class TestRequiredFields:

    def test_missing_active_field_raises(self):
        from neuroslm.dsl.compiler import NeuroMLCompiler, NeuroMLError
        src = (
            "architecture toy { d_sem: 64 }\n"
            "population p1 { count: 16 }\n"
            "export equation eq_foo { params: [x], formula: \"y = x\" }\n"
            "feature feat_foo { equation: eq_foo }\n"   # no `active`
        )
        with pytest.raises(NeuroMLError, match="active"):
            NeuroMLCompiler.compile(src)

    def test_missing_equation_field_raises(self):
        from neuroslm.dsl.compiler import NeuroMLCompiler, NeuroMLError
        src = (
            "architecture toy { d_sem: 64 }\n"
            "population p1 { count: 16 }\n"
            "feature feat_foo { active: true }\n"   # no `equation`
        )
        with pytest.raises(NeuroMLError, match="equation"):
            NeuroMLCompiler.compile(src)


# ─────────────────────────────────────────────────────────────────────
# Contract E — unresolved equation ref
# ─────────────────────────────────────────────────────────────────────


class TestEquationRefMustResolve:

    def test_reference_to_undefined_equation_raises_at_compile(self):
        """Catches typos at compile time so a broken feature wiring
        doesn't make it to the vast.ai box."""
        from neuroslm.dsl.compiler import NeuroMLCompiler, NeuroMLError
        src = (
            "architecture toy { d_sem: 64 }\n"
            "population p1 { count: 16 }\n"
            # eq_foo is NOT defined anywhere
            "feature feat_foo { equation: eq_foo, active: true }\n"
        )
        with pytest.raises(NeuroMLError, match=r"eq_foo|undefined|unknown"):
            NeuroMLCompiler.compile(src)


# ─────────────────────────────────────────────────────────────────────
# Contract F — optional params dict
# ─────────────────────────────────────────────────────────────────────


class TestOptionalParamsDict:

    def test_params_block_parses_as_dict(self):
        from neuroslm.dsl.compiler import NeuroMLCompiler
        src = (
            "architecture toy { d_sem: 64 }\n"
            "population p1 { count: 16 }\n"
            "export equation eq_foo { params: [x], formula: \"y = x\" }\n"
            "feature feat_foo {\n"
            "    equation: eq_foo,\n"
            "    active: true,\n"
            "    params: { alpha: 0.5, k: 4 }\n"
            "}\n"
        )
        prog = NeuroMLCompiler.compile(src)
        f = next(f for f in prog.features if f.name == "feat_foo")
        assert isinstance(f.params, dict), (
            f"params must compile to a dict; got {type(f.params).__name__}")
        assert f.params.get("alpha") == 0.5 or f.params.get("alpha") == "0.5", (
            f"alpha=0.5 must round-trip; got {f.params.get('alpha')!r}")
        assert f.params.get("k") == 4 or f.params.get("k") == "4", (
            f"k=4 must round-trip; got {f.params.get('k')!r}")

    def test_params_block_omitted_yields_empty_dict(self):
        from neuroslm.dsl.compiler import NeuroMLCompiler
        src = (
            "architecture toy { d_sem: 64 }\n"
            "population p1 { count: 16 }\n"
            "export equation eq_foo { params: [x], formula: \"y = x\" }\n"
            "feature feat_foo { equation: eq_foo, active: true }\n"
        )
        prog = NeuroMLCompiler.compile(src)
        f = next(f for f in prog.features if f.name == "feat_foo")
        assert f.params == {}, (
            f"omitted params block must default to {{}}, not None; "
            f"got {f.params!r}")
