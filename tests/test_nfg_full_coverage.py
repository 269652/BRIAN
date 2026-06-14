# -*- coding: utf-8 -*-
"""TDD: complete NFG coverage — multi-cortex experts, param_scopes,
anatomical clustering.

Three gaps in the current Graphviz NFG pipeline (reported by the user
2026-06-09):

  1. The 4 GPT-2 cortex experts declared inside the
     ``training { multi_cortex { ... } }`` block of ``arch.neuro`` are
     NOT lifted into the hypergraph — only top-level `population`
     declarations are picked up.

  2. The KL distillation aux loss (Slot A) and NT-mediated α gating
     (Slot C) connecting cortex_experts → lm_trunk are likewise invisible
     in the diagram.

  3. ``param_scope trunk { ... }`` and ``param_scope bio { ... }`` group
     populations into anatomical / functional regions but the renderer
     dumps every population into one giant ``cluster_populations`` so
     basal ganglia, memory system, neuromodulator nuclei all blur
     together.

These tests pin the contract for the fix: the lifter must see cortex
experts as ``cortex_expert`` HyperNodes, must emit ``distillation`` +
``inhibition`` HyperEdges from each expert to the LM trunk, and must
populate every population's ``attrs["param_scope"]`` so the renderer can
draw clusters by scope/region.
"""
from __future__ import annotations

import shutil
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
# 1. Lifter must surface multi-cortex experts + Slot A/C edges
# ───────────────────────────────────────────────────────────────────────

class TestLifterSurfacesMultiCortex:
    """The rcc_bowtie arch declares
        training {
          multi_cortex { enabled: true, n_cortices: 4, domains: [...],
                         distillation_enabled: true,
                         inhibition_enabled:   true, ... }
        }
    The lifter must emit one ``cortex_expert`` node per declared domain
    plus ``distillation`` / ``inhibition`` hyperedges to the LM trunk so
    Slot A + Slot C show up in the diagram.
    """

    def _ir(self):
        from neuroslm.compiler.hypergraph_ir import lift_arch_to_hypergraph
        return lift_arch_to_hypergraph(RCC_ARCH)

    def _expected_domains(self):
        """The roster of experts the arch.neuro currently declares.

        Source of truth: parse the arch DSL and read the
        ``multi_cortex.experts`` block (post-H21 roster path) or fall
        back to the legacy ``domains`` list. Data-driven so this test
        does not break every time the roster is edited — it only
        breaks when the *lifter* gets out of sync with the *parser*,
        which is the actual contract being asserted.
        """
        from neuroslm.dsl.training_config import (
            _extract_block, parse_training_config,
        )
        body = (RCC_ARCH / "arch.neuro").read_text(encoding="utf-8")
        # parse_training_config expects the contents INSIDE the
        # `training { ... }` braces, not the whole arch file.
        training_body = _extract_block(body, "training")
        assert training_body is not None, "arch.neuro has no training {} block"
        cfg = parse_training_config(training_body)
        mc = getattr(cfg, "multi_cortex", None)
        assert mc is not None, "arch.neuro has no multi_cortex block"
        experts = list(getattr(mc, "experts", []) or [])
        if experts:
            return [str(e.domain) for e in experts]
        # Legacy path: domains list with no per-expert spec.
        return list(getattr(mc, "domains", []))

    def test_emits_lm_trunk_node(self):
        """A synthetic `lm_trunk` node anchors distillation + inhibition
        edges in the diagram — distinct from any DSL population."""
        ir = self._ir()
        trunk = [n for n in ir.nodes if n.kind == "lm_trunk"]
        assert len(trunk) == 1, f"expected exactly 1 lm_trunk node, got {len(trunk)}"

    def test_emits_cortex_expert_nodes(self):
        ir = self._ir()
        experts = [n for n in ir.nodes if n.kind == "cortex_expert"]
        expected_domains = self._expected_domains()
        assert len(experts) == len(expected_domains), (
            f"lifter must emit one cortex_expert node per arch.neuro "
            f"roster entry — expected {len(expected_domains)} "
            f"({expected_domains}), got {len(experts)} "
            f"({[n.name for n in experts]})"
        )
        names = {n.name for n in experts}
        expected_names = {f"cortex_{d}" for d in expected_domains}
        assert names == expected_names, (
            f"expert node names {names} do not match arch.neuro roster "
            f"{expected_names}"
        )

    def test_cortex_experts_record_weights_provider(self):
        ir = self._ir()
        experts = [n for n in ir.nodes if n.kind == "cortex_expert"]
        # Per-expert weights tag must be non-empty. Each entry comes from
        # either the per-expert `id:` field (new MoE roster path) or the
        # block-level `weights:` field (legacy single-weights path).
        for n in experts:
            w = n.attrs.get("weights")
            assert w not in (None, "", "None"), (
                f"expert {n.name} has no weights provider tag "
                f"(attrs={dict(n.attrs)})"
            )
            assert n.attrs.get("freeze_weights") in {"true", True}, \
                f"expert {n.name} missing freeze_weights tag"
    def test_emits_distillation_edges_to_trunk(self):
        """One `distillation` hyperedge per cortex expert, all targeting
        the synthetic `lm_trunk` node."""
        ir = self._ir()
        expected = len(self._expected_domains())
        dist_edges = [e for e in ir.hyperedges if e.kind == "distillation"]
        assert len(dist_edges) == expected, \
            f"expected {expected} distillation edges, got {len(dist_edges)}"
        for e in dist_edges:
            assert len(e.members) == 2
            assert e.members[1] == "lm_trunk", \
                f"distillation edge {e.id} should target lm_trunk, got {e.members[1]}"
            assert e.members[0].startswith("cortex_"), \
                f"distillation edge {e.id} should originate at a cortex expert"

    def test_distillation_edges_carry_lambda_max(self):
        ir = self._ir()
        dist_edges = [e for e in ir.hyperedges if e.kind == "distillation"]
        for e in dist_edges:
            assert "lambda_max" in e.attrs, \
                f"distillation edge {e.id} missing λ_max attr"
            assert "temperature" in e.attrs, \
                f"distillation edge {e.id} missing temperature attr"

    def test_emits_inhibition_edges_trunk_to_experts(self):
        """One `inhibition` (Slot C / NT-gated α) hyperedge per expert,
        from `lm_trunk` BACK to each cortex expert (the trunk releases
        the inhibitory signal when it outperforms)."""
        ir = self._ir()
        expected = len(self._expected_domains())
        inh_edges = [e for e in ir.hyperedges if e.kind == "inhibition"]
        assert len(inh_edges) == expected, \
            f"expected {expected} inhibition edges, got {len(inh_edges)}"
        for e in inh_edges:
            assert e.members[0] == "lm_trunk", \
                f"inhibition edge {e.id} should originate at lm_trunk"
            assert e.members[1].startswith("cortex_"), \
                f"inhibition edge {e.id} should target a cortex expert"


# ───────────────────────────────────────────────────────────────────────
# 2. Lifter must tag every population with its param_scope (anatomy)
# ───────────────────────────────────────────────────────────────────────

class TestParamScopeAnnotation:
    """`param_scope trunk { populations: [...] }` and
    `param_scope bio   { populations: [..., gradient: "detached..."] }`
    must annotate every listed population's HyperNode with
    ``attrs["param_scope"]`` so the renderer can cluster them."""

    def _ir(self):
        from neuroslm.compiler.hypergraph_ir import lift_arch_to_hypergraph
        return lift_arch_to_hypergraph(RCC_ARCH)

    def test_bg_tagged_as_bio(self):
        ir = self._ir()
        bg = next((n for n in ir.nodes
                   if n.kind == "population" and n.name == "bg"), None)
        assert bg is not None, "basal ganglia population missing"
        assert bg.attrs.get("param_scope") == "bio", \
            f"bg should be in param_scope=bio, got {bg.attrs.get('param_scope')}"

    def test_hippo_tagged_as_bio(self):
        ir = self._ir()
        hippo = next((n for n in ir.nodes
                      if n.kind == "population" and n.name == "hippo"), None)
        assert hippo is not None
        assert hippo.attrs.get("param_scope") == "bio"

    def test_pfc_tagged_as_trunk(self):
        ir = self._ir()
        pfc = next((n for n in ir.nodes
                    if n.kind == "population" and n.name == "pfc"), None)
        assert pfc is not None
        assert pfc.attrs.get("param_scope") == "trunk"

    def test_every_population_has_a_scope(self):
        """Every population in the IR must be assigned to either trunk
        or bio (no orphans)."""
        ir = self._ir()
        for n in ir.nodes:
            if n.kind != "population":
                continue
            scope = n.attrs.get("param_scope")
            assert scope in {"trunk", "bio"}, \
                f"population {n.name} has unexpected scope {scope!r}"


# ───────────────────────────────────────────────────────────────────────
# 3. Renderer must group by anatomical/functional region
# ───────────────────────────────────────────────────────────────────────

@pytest.mark.skipif(not _HAS_GRAPHVIZ_PKG,
                    reason="graphviz python package required")
class TestRendererAnatomicalClustering:
    """Replace the one giant `cluster_populations` with per-region
    clusters so basal ganglia, memory system, nuclei etc. each get
    visual real-estate."""

    def _ir(self):
        from neuroslm.compiler.hypergraph_ir import lift_arch_to_hypergraph
        return lift_arch_to_hypergraph(RCC_ARCH)

    def test_dot_includes_multi_cortex_cluster(self):
        from neuroslm.compiler.nfg_graphviz import emit_dot_from_hypergraph
        dot = emit_dot_from_hypergraph(self._ir())
        assert "cluster_multi_cortex" in dot, \
            "expected `cluster_multi_cortex` subgraph for GPT-2 experts"

    def test_dot_includes_basal_ganglia_cluster(self):
        from neuroslm.compiler.nfg_graphviz import emit_dot_from_hypergraph
        dot = emit_dot_from_hypergraph(self._ir())
        assert "cluster_basal_ganglia" in dot, \
            "expected `cluster_basal_ganglia` subgraph"

    def test_dot_includes_memory_cluster(self):
        from neuroslm.compiler.nfg_graphviz import emit_dot_from_hypergraph
        dot = emit_dot_from_hypergraph(self._ir())
        assert "cluster_memory" in dot, \
            "expected `cluster_memory` subgraph (hippo + entorhinal + cerebellum)"

    def test_dot_includes_neuromodulator_nuclei_cluster(self):
        from neuroslm.compiler.nfg_graphviz import emit_dot_from_hypergraph
        dot = emit_dot_from_hypergraph(self._ir())
        assert "cluster_nuclei" in dot, \
            "expected `cluster_nuclei` subgraph (vta, locus_coeruleus, raphe...)"

    def test_dot_shows_distillation_and_inhibition_edges(self):
        from neuroslm.compiler.nfg_graphviz import emit_dot_from_hypergraph
        dot = emit_dot_from_hypergraph(self._ir())
        # both kinds of cortex↔trunk edges should appear in the rendered DOT
        assert "lm_trunk" in dot, "synthetic lm_trunk node not rendered"
        # at least one edge labelled with KL/distill or inhibition
        assert ("distill" in dot.lower() or "kl" in dot.lower()), \
            "distillation edges not visible in DOT output"
        assert "inhibit" in dot.lower() or "nt-gat" in dot.lower(), \
            "inhibition edges not visible in DOT output"


# ───────────────────────────────────────────────────────────────────────
# 4. End-to-end: PNG render exists and is non-trivial
# ───────────────────────────────────────────────────────────────────────

@pytest.mark.skipif(not (_HAS_GRAPHVIZ_PKG and _HAS_DOT),
                    reason="graphviz package + dot binary required")
class TestPngRenderHasNewElements:
    def test_png_renders_with_all_new_nodes(self, tmp_path):
        from neuroslm.compiler.hypergraph_ir import lift_arch_to_hypergraph
        from neuroslm.compiler.nfg_graphviz import render_hypergraph
        ir = lift_arch_to_hypergraph(RCC_ARCH)
        out = tmp_path / "nfg.png"
        render_hypergraph(ir, str(out), format="png")
        assert out.exists()
        # ~600 KB for the full rcc_bowtie arch; raise the floor a bit so
        # accidental omission of the multi-cortex cluster gets caught.
        assert out.stat().st_size > 50_000, \
            f"PNG implausibly small ({out.stat().st_size} bytes)"
