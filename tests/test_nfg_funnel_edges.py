"""Contracts for NFG visibility of declarative ``funnel`` IR rows.

Fix C (2026-06-15 audit follow-up): the ``module brain = LanguageCortex``
block produces a ``FunnelIR`` that wires the expert ensemble into the
LM-trunk population (``target: pfc``). That's the actual MoE→LM
gradient path. But the NFG renderer only emitted edges for ``synapse``
and ``modulation`` rows, so the funnel projection was invisible on the
graph — the brain module looked unconnected to anything.

This file pins the new ``kind="funnel"`` edges:

  * ``compile_nfg(arch_root)`` must emit one NFGEdge per
    (input_expert, target) pair on every ``FunnelIR`` row.
  * The new edges are tagged ``kind="funnel"`` so the renderer can
    draw them with a distinct visual grammar (dashed gradient edge)
    without confusing them with forward synapses or NT modulations.
  * The funnel edges live next to the synapse + modulation edges
    on ``NeuralFlowGraph.edges`` so no consumer needs a new
    iteration surface.
"""
from __future__ import annotations

from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parent.parent
MASTER_ARCH = REPO_ROOT / "architectures" / "master"


# ──────────────────────────────────────────────────────────────────────
# Helper: a minimal arch.neuro with one LanguageCortex instantiation
# ──────────────────────────────────────────────────────────────────────


def _minimal_arch_with_funnel(tmp_path: Path) -> Path:
    """Write a tiny but-compilable arch into ``tmp_path`` whose IR
    has exactly one FunnelIR row: ``[E1, E2] -> pfc``."""
    arch_dir = tmp_path / "fixc_arch"
    arch_dir.mkdir()
    (arch_dir / "arch.neuro").write_text(
        "architecture fixc { d_sem: 32 }\n"
        "\n"
        "population pfc { count: 32 }\n"
        "population thalamus { count: 32 }\n"
        "\n"
        "expert E1 { model: \"gpt2\",       role: \"general\" }\n"
        "expert E2 { model: \"distilgpt2\", role: \"code\"    }\n"
        "\n"
        "distillation cfd_loss {\n"
        "    method:      \"capacity_funneled\",\n"
        "    temperature: 4.0,\n"
        "    alpha:       0.7,\n"
        "    bottleneck:  64,\n"
        "    loss:        \"kl_div\"\n"
        "}\n"
        "\n"
        "funnel teacher_ensemble {\n"
        "    inputs:       [E1, E2],\n"
        "    target:       pfc,\n"
        "    d_bottleneck: 64,\n"
        "    gate:         \"softmax_router\",\n"
        "    method:       cfd_loss\n"
        "}\n",
        encoding="utf-8",
    )
    return arch_dir


# ──────────────────────────────────────────────────────────────────────
# Fix C — compile_nfg must emit funnel edges
# ──────────────────────────────────────────────────────────────────────


class TestFixCNFGEmitsFunnelEdges:
    """``compile_nfg`` must surface ``FunnelIR`` rows as graph edges so
    the rendered NFG matches the actual gradient topology."""

    def test_compile_nfg_emits_one_edge_per_funnel_input(self, tmp_path: Path):
        """One edge per (input, target) pair — so a 2-expert funnel
        produces 2 edges (E1->pfc, E2->pfc)."""
        from neuroslm.dsl.nfg import compile_nfg

        arch = _minimal_arch_with_funnel(tmp_path)
        g = compile_nfg(arch)

        funnel_edges = [e for e in g.edges if e.kind == "funnel"]
        # Endpoints sorted so the assert is order-independent.
        pairs = sorted((e.src, e.tgt) for e in funnel_edges)
        assert pairs == [("E1", "pfc"), ("E2", "pfc")], (
            f"expected exactly 2 funnel edges (one per input), got "
            f"{pairs!r}"
        )

    def test_funnel_edges_are_kind_funnel(self, tmp_path: Path):
        """The new edge tag MUST be ``kind=\"funnel\"`` (not
        ``synapse`` or ``modulation``) so the renderer can pick them
        out and draw them with a dashed distillation-gradient
        grammar."""
        from neuroslm.dsl.nfg import compile_nfg

        arch = _minimal_arch_with_funnel(tmp_path)
        g = compile_nfg(arch)

        # NO funnel edge should be misclassified as a synapse
        synapse_pairs = {(e.src, e.tgt) for e in g.edges if e.kind == "synapse"}
        assert ("E1", "pfc") not in synapse_pairs, (
            "funnel edge E1->pfc was emitted with kind='synapse'; "
            "must be kind='funnel' so the renderer can distinguish "
            "distillation gradient from forward signal"
        )
        funnel_kinds = {e.kind for e in g.edges if (e.src, e.tgt) in {("E1", "pfc"), ("E2", "pfc")}}
        assert funnel_kinds == {"funnel"}, (
            f"funnel edges have wrong kind tag: {funnel_kinds!r}; "
            f"expected {{'funnel'}}"
        )

    def test_funnel_edges_appear_in_stats(self, tmp_path: Path):
        """``g.stats()`` must report the new edge count so dashboards
        + tests can spot when a funnel is missing."""
        from neuroslm.dsl.nfg import compile_nfg

        arch = _minimal_arch_with_funnel(tmp_path)
        g = compile_nfg(arch)

        stats = g.stats()
        assert "n_funnels" in stats, (
            "stats() must report `n_funnels` so missing-funnel "
            "regressions are visible at a glance"
        )
        assert stats["n_funnels"] == 2, (
            f"expected n_funnels==2 (one edge per input), got "
            f"{stats['n_funnels']}"
        )

    def test_no_funnel_in_arch_means_no_funnel_edges(self, tmp_path: Path):
        """Negative control: an arch with NO funnel block must NOT
        produce any funnel edges."""
        from neuroslm.dsl.nfg import compile_nfg

        arch_dir = tmp_path / "no_funnel_arch"
        arch_dir.mkdir()
        (arch_dir / "arch.neuro").write_text(
            "architecture nf { d_sem: 32 }\n"
            "population pfc { count: 32 }\n"
            "population thalamus { count: 32 }\n"
            "synapse thalamus -> pfc { weight: 0.5, neurotransmitter: \"glutamate\" }\n",
            encoding="utf-8",
        )
        g = compile_nfg(arch_dir)
        funnel_edges = [e for e in g.edges if e.kind == "funnel"]
        assert funnel_edges == [], (
            f"arch has no funnel block but renderer emitted "
            f"{len(funnel_edges)} phantom funnel edges: {funnel_edges!r}"
        )


# ──────────────────────────────────────────────────────────────────────
# End-to-end on the canonical master arch
# ──────────────────────────────────────────────────────────────────────


class TestMasterArchFunnelEdges:
    """The actual ``architectures/master/`` arch declares
    ``module brain = LanguageCortex { target: pfc, experts: [...] }``
    which expands into a funnel with 3 inputs (cortex_code,
    cortex_general, cortex_reasoning). Pin the resulting NFG funnel
    edges so the brain module's projection is visible end-to-end."""

    @pytest.fixture(scope="class")
    def graph(self):
        from neuroslm.dsl.nfg import compile_nfg
        return compile_nfg(MASTER_ARCH)

    def test_master_arch_has_funnel_edges(self, graph):
        funnel_edges = [e for e in graph.edges if e.kind == "funnel"]
        assert len(funnel_edges) >= 1, (
            "architectures/master/ declares `module brain = "
            "LanguageCortex { ... }` which expands into a funnel; "
            "expected at least one funnel edge on the rendered NFG "
            "but got zero — the brain module's MoE→LM projection is "
            "invisible on the graph"
        )

    def test_master_arch_funnel_target_is_pfc(self, graph):
        """Every funnel edge in the master arch must point at ``pfc``
        (the population that plays the LM-trunk role in the bowtie)."""
        funnel_edges = [e for e in graph.edges if e.kind == "funnel"]
        targets = {e.tgt for e in funnel_edges}
        assert targets == {"pfc"}, (
            f"master arch funnel targets {sorted(targets)} != "
            f"{{'pfc'}} — the LanguageCortex template's `target: pfc` "
            f"override must reach the funnel IR row"
        )
