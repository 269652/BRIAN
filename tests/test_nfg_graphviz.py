# -*- coding: utf-8 -*-
"""TDD: Graphviz NFG emitter built on top of the Hypergraph IR.

The user complaint was that `brian compile nfg` produced overlapping,
hand-laid-out matplotlib plots. The fix:

  1. **Hypergraph IR is the source of truth** for the visualisation,
     not a separate parallel graph type. The lifter walks every
     resolved module file (not just arch.neuro) so populations
     declared in `modules/*.neuro` show up too.

  2. **Graphviz `dot` engine does the layout** (proper hierarchical
     layered DAG) instead of hand-coded coordinates.

  3. **Subgraph clusters by kind** (populations / neurotransmitters /
     formal specs / param scopes) give clean visual grouping.

  4. **Equations + properties on labels** so the math is visible on
     the diagram, not hidden in side panels.

These tests pin the contract of the new pipeline so the CLI rewrite
in `cmd_compile_nfg` stays honest.
"""
from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).parent.parent
RCC_ARCH = REPO_ROOT / "architectures" / "rcc_bowtie"

_HAS_DOT = shutil.which("dot") is not None
_HAS_GRAPHVIZ_PKG = True
try:
    import graphviz  # noqa: F401
except ImportError:
    _HAS_GRAPHVIZ_PKG = False


# ───────────────────────────────────────────────────────────────────────
# 1. Extended hypergraph lifter — walks resolved modules
# ───────────────────────────────────────────────────────────────────────

class TestLiftArchToHypergraph:
    """`lift_arch_to_hypergraph(arch_root)` must see populations / synapses /
    modulations declared in **imported module files**, not only the top-level
    `arch.neuro`.
    """

    def test_helper_exists(self):
        from neuroslm.compiler.hypergraph_ir import lift_arch_to_hypergraph  # noqa: F401

    def test_returns_hypergraph_ir(self):
        from neuroslm.compiler.hypergraph_ir import (
            lift_arch_to_hypergraph, HypergraphIR,
        )
        ir = lift_arch_to_hypergraph(RCC_ARCH)
        assert isinstance(ir, HypergraphIR)

    def test_sees_populations_from_modules(self):
        """rcc_bowtie has 33 populations declared across modules/*.neuro.
        The raw lifter on arch.neuro alone sees 0; the arch-walking lifter
        must see >= 20."""
        from neuroslm.compiler.hypergraph_ir import lift_arch_to_hypergraph
        ir = lift_arch_to_hypergraph(RCC_ARCH)
        pops = [n for n in ir.nodes if n.kind == "population"]
        assert len(pops) >= 20, f"only {len(pops)} populations found; expected >= 20"

    def test_sees_all_seven_neurotransmitters(self):
        """rcc_bowtie declares 7 NT systems in arch.neuro."""
        from neuroslm.compiler.hypergraph_ir import lift_arch_to_hypergraph
        ir = lift_arch_to_hypergraph(RCC_ARCH)
        nts = [n for n in ir.nodes if n.kind == "neurotransmitter"]
        assert len(nts) == 7, f"expected 7 NTs, got {len(nts)}"

    def test_sees_synapses_and_modulations(self):
        """rcc_bowtie has many synapse + modulation edges."""
        from neuroslm.compiler.hypergraph_ir import lift_arch_to_hypergraph
        ir = lift_arch_to_hypergraph(RCC_ARCH)
        syns = [e for e in ir.hyperedges if e.kind == "synapse"]
        mods = [e for e in ir.hyperedges if e.kind == "modulation"]
        assert len(syns) >= 10, f"expected >= 10 synapses, got {len(syns)}"
        assert len(mods) >= 10, f"expected >= 10 modulations, got {len(mods)}"

    def test_source_map_records_per_node_origin(self):
        """Provenance preserved: every node id has a span."""
        from neuroslm.compiler.hypergraph_ir import lift_arch_to_hypergraph
        ir = lift_arch_to_hypergraph(RCC_ARCH)
        for n in ir.nodes:
            assert n.id in ir.source_map.spans, f"node {n.id} missing span"

    def test_handles_string_path_argument(self):
        """Accepts `str` and `Path` interchangeably."""
        from neuroslm.compiler.hypergraph_ir import lift_arch_to_hypergraph
        ir_a = lift_arch_to_hypergraph(str(RCC_ARCH))
        ir_b = lift_arch_to_hypergraph(RCC_ARCH)
        assert len(ir_a.nodes) == len(ir_b.nodes)


# ───────────────────────────────────────────────────────────────────────
# 2. Graphviz emitter contracts
# ───────────────────────────────────────────────────────────────────────

@pytest.mark.skipif(not _HAS_GRAPHVIZ_PKG,
                    reason="graphviz python package required")
class TestEmitDotFromHypergraph:
    """`emit_dot_from_hypergraph(ir, ...) -> str` returns a syntactically
    valid DOT string driven entirely by the HypergraphIR."""

    def _ir(self):
        from neuroslm.compiler.hypergraph_ir import lift_arch_to_hypergraph
        return lift_arch_to_hypergraph(RCC_ARCH)

    def test_helper_exists(self):
        from neuroslm.compiler.nfg_graphviz import emit_dot_from_hypergraph  # noqa: F401

    def test_returns_nonempty_string(self):
        from neuroslm.compiler.nfg_graphviz import emit_dot_from_hypergraph
        dot = emit_dot_from_hypergraph(self._ir())
        assert isinstance(dot, str)
        assert len(dot) > 200, "DOT output suspiciously short"

    def test_dot_is_a_digraph(self):
        from neuroslm.compiler.nfg_graphviz import emit_dot_from_hypergraph
        dot = emit_dot_from_hypergraph(self._ir())
        assert dot.lstrip().startswith("digraph")

    def test_every_population_appears(self):
        from neuroslm.compiler.nfg_graphviz import emit_dot_from_hypergraph
        import re
        ir = self._ir()
        dot = emit_dot_from_hypergraph(ir)
        for n in ir.nodes:
            if n.kind == "population":
                # population must appear as a DOT node id (start of a
                # node-definition line): either quoted or as a bare word
                # before `[`. We only count node-definitions, not arbitrary
                # substring hits inside labels.
                pat = re.compile(
                    rf'(^|\n)\s*("?){re.escape(n.name)}\2\s*\[',
                    re.MULTILINE,
                )
                assert pat.search(dot), \
                    f"population {n.name} missing as a DOT node definition"

    def test_every_neurotransmitter_appears(self):
        from neuroslm.compiler.nfg_graphviz import emit_dot_from_hypergraph
        ir = self._ir()
        dot = emit_dot_from_hypergraph(ir)
        for n in ir.nodes:
            if n.kind == "neurotransmitter":
                assert n.name in dot, f"NT {n.name} missing from DOT output"

    def test_synapses_emit_edges(self):
        from neuroslm.compiler.nfg_graphviz import emit_dot_from_hypergraph
        ir = self._ir()
        dot = emit_dot_from_hypergraph(ir)
        # at least one synapse arrow should be visible: src -> dst
        assert "->" in dot
        syn_edges = [e for e in ir.hyperedges if e.kind == "synapse"]
        assert syn_edges, "fixture failed: no synapses in IR"
        sample = syn_edges[0]
        src, dst = sample.members[0], sample.members[1]
        assert f"{src}" in dot and f"{dst}" in dot

    def test_clusters_by_kind(self):
        """Populations should be grouped in `subgraph cluster_*` blocks."""
        from neuroslm.compiler.nfg_graphviz import emit_dot_from_hypergraph
        dot = emit_dot_from_hypergraph(self._ir())
        assert "subgraph cluster_" in dot, \
            "expected subgraph clustering for visual grouping"

    def test_uses_dot_engine_directive_by_default(self):
        """The emitter should request hierarchical (layered) layout via dot."""
        from neuroslm.compiler.nfg_graphviz import emit_dot_from_hypergraph
        dot = emit_dot_from_hypergraph(self._ir())
        # Either explicit rankdir or layout=... directive
        assert "rankdir" in dot or "layout=" in dot

    def test_back_compat_engine_param_neato(self):
        """`engine='neato'` switches to force-directed layout."""
        from neuroslm.compiler.nfg_graphviz import emit_dot_from_hypergraph
        dot = emit_dot_from_hypergraph(self._ir(), engine="neato")
        assert isinstance(dot, str) and len(dot) > 100


# ───────────────────────────────────────────────────────────────────────
# 3. Rendering (PNG/SVG) — requires `dot` binary on PATH
# ───────────────────────────────────────────────────────────────────────

@pytest.mark.skipif(not (_HAS_GRAPHVIZ_PKG and _HAS_DOT),
                    reason="graphviz package + dot binary required")
class TestRenderHypergraphToPng:
    """`render_hypergraph(ir, out_path, format='png')` writes a real image."""

    def test_helper_exists(self):
        from neuroslm.compiler.nfg_graphviz import render_hypergraph  # noqa: F401

    def test_writes_png(self, tmp_path):
        from neuroslm.compiler.hypergraph_ir import lift_arch_to_hypergraph
        from neuroslm.compiler.nfg_graphviz import render_hypergraph
        ir = lift_arch_to_hypergraph(RCC_ARCH)
        out = tmp_path / "nfg.png"
        render_hypergraph(ir, str(out), format="png")
        assert out.exists()
        assert out.stat().st_size > 1000, "PNG implausibly small"

    def test_writes_svg(self, tmp_path):
        from neuroslm.compiler.hypergraph_ir import lift_arch_to_hypergraph
        from neuroslm.compiler.nfg_graphviz import render_hypergraph
        ir = lift_arch_to_hypergraph(RCC_ARCH)
        out = tmp_path / "nfg.svg"
        render_hypergraph(ir, str(out), format="svg")
        assert out.exists()
        # SVG is XML text
        head = out.read_text(encoding="utf-8")[:200]
        assert "<svg" in head or "<?xml" in head

    def test_writes_dot(self, tmp_path):
        from neuroslm.compiler.hypergraph_ir import lift_arch_to_hypergraph
        from neuroslm.compiler.nfg_graphviz import render_hypergraph
        ir = lift_arch_to_hypergraph(RCC_ARCH)
        out = tmp_path / "nfg.dot"
        render_hypergraph(ir, str(out), format="dot")
        assert out.exists()
        text = out.read_text(encoding="utf-8")
        assert "digraph" in text


# ───────────────────────────────────────────────────────────────────────
# 4. CLI integration — `brian compile nfg` uses the new pipeline
# ───────────────────────────────────────────────────────────────────────

@pytest.mark.skipif(not (_HAS_GRAPHVIZ_PKG and _HAS_DOT),
                    reason="graphviz package + dot binary required")
class TestCliUsesGraphvizByDefault:
    """`brian compile nfg <arch>` should default to the new Graphviz pipeline.
    Old matplotlib path remains reachable via `--legacy`."""

    def test_default_emits_png_via_graphviz(self, tmp_path):
        import sys
        import argparse
        from neuroslm.cli import cmd_compile_nfg
        out_png = tmp_path / "nfg.png"
        args = argparse.Namespace(
            arch=str(RCC_ARCH),
            out=None,
            png=str(out_png),
            semantic=False,
            legacy=False,
        )
        rc = cmd_compile_nfg(args)
        assert rc == 0
        assert out_png.exists()

    def test_legacy_flag_falls_back_to_matplotlib(self, tmp_path):
        """When --legacy is set, the old `neuroslm.dsl.nfg` path is used."""
        import argparse
        from neuroslm.cli import cmd_compile_nfg
        out_py = tmp_path / "nfg.py"
        out_png = tmp_path / "nfg_legacy.png"
        args = argparse.Namespace(
            arch=str(RCC_ARCH),
            out=str(out_py),
            png=str(out_png),
            semantic=False,
            legacy=True,
        )
        rc = cmd_compile_nfg(args)
        assert rc == 0
        # legacy path always writes the .py too
        assert out_py.exists()
