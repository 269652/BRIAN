# -*- coding: utf-8 -*-
"""Phase N10/N11 — hypershape compiler + graph-theoretic analysis.

N10: lower the DSL model to a typed multigraph (HyperShape) — nodes are
ops/params, edges are shaped tensor flows, with a parallel adjoint
(gradient) graph and labeled subsystem regions.

N11: analyze that graph — Fiedler connectivity, spectral gap, degree
centrality, articulation points (computation bottlenecks), and a
Φ-bipartition (integration via normalized min-cut). This is the
machinery the docs/dsl_nn_language.md §N10-N11 roadmap calls for, to
optimize intelligence density / Φ / EI by inspecting the model's
mathematical structure.
"""
import pytest

from neuroslm.dsl.nn_lang import build_language_model
from neuroslm.dsl import hypershape as H


@pytest.fixture
def small_lm():
    return build_language_model(vocab=64, d_model=32, depth=3,
                                n_heads=4, max_ctx=64)


# ── N10: compilation ───────────────────────────────────────────────────

class TestCompileHypershape:
    def test_produces_nodes_and_edges(self, small_lm):
        hs = H.compile_hypershape(small_lm)
        assert len(hs.nodes) > 0
        assert len(hs.edges) > 0

    def test_has_embed_and_head(self, small_lm):
        hs = H.compile_hypershape(small_lm)
        kinds = {n.kind for n in hs.nodes}
        assert "embedding" in kinds
        assert "head" in kinds

    def test_one_region_per_block(self, small_lm):
        hs = H.compile_hypershape(small_lm)
        # depth=3 → three block regions
        block_regions = {n.region for n in hs.nodes if n.region.startswith("block")}
        assert len(block_regions) == 3

    def test_forward_path_connected(self, small_lm):
        # There must be a forward path embed → … → head.
        hs = H.compile_hypershape(small_lm)
        assert hs.has_path("embed", "head")

    def test_adjoint_reverses_edges(self, small_lm):
        hs = H.compile_hypershape(small_lm)
        adj = hs.adjoint()
        # Adjoint (gradient) graph must connect head back to embed.
        assert adj.has_path("head", "embed")


# ── N11: graph analysis ────────────────────────────────────────────────

class TestGraphAnalysis:
    def test_fiedler_positive(self, small_lm):
        hs = H.compile_hypershape(small_lm)
        assert H.fiedler_value(hs) > 0

    def test_spectral_gap_nonneg(self, small_lm):
        hs = H.compile_hypershape(small_lm)
        assert H.spectral_gap(hs) >= 0

    def test_degree_centrality_sums(self, small_lm):
        hs = H.compile_hypershape(small_lm)
        cent = H.degree_centrality(hs)
        assert set(cent.keys()) == {n.id for n in hs.nodes}
        assert all(v >= 0 for v in cent.values())

    def test_articulation_points_include_bottleneck(self, small_lm):
        # In a chain-like transformer, intermediate nodes are cut vertices
        # (removing them disconnects the graph) — the computation
        # bottlenecks the hypershape analysis should surface.
        hs = H.compile_hypershape(small_lm)
        arts = H.articulation_points(hs)
        assert len(arts) > 0

    def test_phi_bipartition_returns_score_and_parts(self, small_lm):
        hs = H.compile_hypershape(small_lm)
        score, part_a, part_b = H.phi_bipartition(hs)
        assert score >= 0
        assert len(part_a) > 0 and len(part_b) > 0
        assert set(part_a).isdisjoint(part_b)


# ── Inspectable summary ────────────────────────────────────────────────

class TestSummary:
    def test_summary_dict(self, small_lm):
        hs = H.compile_hypershape(small_lm)
        s = H.analyze(hs)
        for key in ("n_nodes", "n_edges", "fiedler", "spectral_gap",
                    "n_articulation_points", "phi_bipartition"):
            assert key in s


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
