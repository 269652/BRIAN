# -*- coding: utf-8 -*-
"""TDD: L6 NFG heat overlay.

The NFG compiler accepts an optional heatmap dict (or .json file path)
and tints node / edge fills by normalized heat. Cold paths render dim,
hot paths render saturated red. The base diagram (clusters, kind
colors) is preserved when no heat is supplied.

Contract:
    emit_dot_from_hypergraph(ir, *, heat=None, ...)         # DOT string
    render_hypergraph(ir, out, *, heat=None, ...)           # rendered file
    render_arch(arch_root, out, *, heat=None, ...)          # one-shot

``heat`` can be:
  - a ``dict[str, float]`` (raw or pre-normalized) mapping element id
    to heat value
  - a ``TrainingHeatmap`` (uses ``.normalized()``)
  - a ``str | Path`` pointing at a ``<arch>.heatmap.json`` produced by
    :class:`HeatmapPublisher`
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest


SAMPLE_DSL = (
    "architecture a { d_sem: 256 }\n"
    "neurotransmitter dopamine { base_concentration: 0.5 }\n"
    'population cortex { count: 512, dynamics: "rate_code" }\n'
    'population striatum { count: 256, dynamics: "rate_code" }\n'
    "synapse cortex -> striatum { weight: 0.5 }\n"
    "modulation dopamine -> striatum { gain: 1.2 }\n"
)


@pytest.fixture
def sample_ir():
    from neuroslm.compiler.hypergraph_ir import lift_dsl_to_hypergraph
    return lift_dsl_to_hypergraph(SAMPLE_DSL)


@pytest.fixture
def sample_heat_dict():
    return {
        "population:cortex":              1.0,    # hottest node
        "population:striatum":            0.05,   # cold node
        "synapse:cortex->striatum":       0.95,   # hot edge
        "modulation:dopamine->striatum":  0.02,   # cold edge
    }


# ── normalisation helper ──────────────────────────────────────────


class TestHeatNormalization:
    def test_dict_normalized_to_unit_max(self):
        from neuroslm.compiler.heat_overlay import normalize_heat
        normed = normalize_heat({"a": 4.0, "b": 2.0, "c": 1.0})
        assert normed["a"] == pytest.approx(1.0)
        assert normed["b"] == pytest.approx(0.5)
        assert normed["c"] == pytest.approx(0.25)

    def test_already_normalized_dict_passes_through(self):
        from neuroslm.compiler.heat_overlay import normalize_heat
        normed = normalize_heat({"a": 0.7, "b": 0.3, "c": 1.0})
        assert normed == {"a": 0.7, "b": 0.3, "c": 1.0}

    def test_empty_dict_returns_empty(self):
        from neuroslm.compiler.heat_overlay import normalize_heat
        assert normalize_heat({}) == {}


class TestHeatColormap:
    def test_zero_heat_maps_to_dim_white(self):
        from neuroslm.compiler.heat_overlay import heat_to_fillcolor
        c = heat_to_fillcolor(0.0)
        assert c.startswith("#")
        # Should be a near-white / very dim color
        rgb = tuple(int(c[i:i+2], 16) for i in (1, 3, 5))
        assert min(rgb) > 230               # all channels near 255

    def test_one_heat_maps_to_saturated_red(self):
        from neuroslm.compiler.heat_overlay import heat_to_fillcolor
        c = heat_to_fillcolor(1.0)
        r, g, b = (int(c[i:i+2], 16) for i in (1, 3, 5))
        assert r > 200 and g < 100 and b < 100

    def test_monotone_in_heat(self):
        from neuroslm.compiler.heat_overlay import heat_to_fillcolor
        # Hotter -> larger red-vs-green saturation gap (white = R=G=B,
        # full red = R>>G,B). Our ramp moves WHITE -> BRICK by reducing
        # G and B faster than R, so R-G increases monotonically.
        gaps = []
        for h in (0.0, 0.5, 1.0):
            c = heat_to_fillcolor(h)
            r, g, _b = (int(c[i:i+2], 16) for i in (1, 3, 5))
            gaps.append(r - g)
        assert gaps[0] < gaps[1] < gaps[2]


# ── DOT emission with heat overlay ─────────────────────────────────


class TestEmitDotWithHeat:
    def test_emits_without_heat_unchanged(self, sample_ir):
        """No heat kwarg -> output matches the un-overlaid DOT."""
        from neuroslm.compiler.nfg_graphviz import emit_dot_from_hypergraph
        plain = emit_dot_from_hypergraph(sample_ir)
        with_no_heat = emit_dot_from_hypergraph(sample_ir, heat=None)
        assert plain == with_no_heat

    def test_heat_dict_changes_population_fillcolor(self, sample_ir, sample_heat_dict):
        from neuroslm.compiler.nfg_graphviz import emit_dot_from_hypergraph
        from neuroslm.compiler.heat_overlay import heat_to_fillcolor
        dot = emit_dot_from_hypergraph(sample_ir, heat=sample_heat_dict)
        # Cortex (heat=1.0) should have the saturated-red fill.
        hot_fill = heat_to_fillcolor(1.0)
        assert hot_fill.lower() in dot.lower(), \
            f"hot fillcolor {hot_fill} not found in DOT:\n{dot[:1500]}"

    def test_heat_dict_changes_edge_color(self, sample_ir, sample_heat_dict):
        """A hot synapse should render in a hot color rather than the
        default NT color (or at least carry a hot penwidth)."""
        from neuroslm.compiler.nfg_graphviz import emit_dot_from_hypergraph
        from neuroslm.compiler.heat_overlay import heat_to_fillcolor
        dot = emit_dot_from_hypergraph(sample_ir, heat=sample_heat_dict)
        hot_color = heat_to_fillcolor(0.95)        # cortex->striatum heat
        # The hot edge color (or a near-equivalent saturated red) appears.
        assert hot_color.lower() in dot.lower() \
            or "#ff" in dot.lower() or "#e" in dot.lower()

    def test_unknown_element_ids_are_ignored(self, sample_ir):
        """Heat entries that don't match any IR element are silently ignored."""
        from neuroslm.compiler.nfg_graphviz import emit_dot_from_hypergraph
        # Must not raise.
        dot = emit_dot_from_hypergraph(
            sample_ir,
            heat={"population:nonexistent": 1.0, "synapse:also->missing": 0.5},
        )
        assert "cortex" in dot              # base diagram still rendered


class TestHeatFromTrainingHeatmap:
    def test_accepts_training_heatmap_instance(self, sample_ir):
        from neuroslm.compiler.nfg_graphviz import emit_dot_from_hypergraph
        from neuroslm.evolution import TrainingHeatmap
        hm = TrainingHeatmap()
        hm.update(
            {"population:cortex": 1.0, "population:striatum": 0.1},
            kinds={"population:cortex": "node", "population:striatum": "node"},
        )
        dot = emit_dot_from_hypergraph(sample_ir, heat=hm)
        # Some non-default fill should appear (cortex is now hotter).
        assert "fillcolor=\"#" in dot


class TestHeatFromJsonFile:
    def test_loads_heat_from_json_path(self, sample_ir, sample_heat_dict, tmp_path):
        from neuroslm.compiler.nfg_graphviz import emit_dot_from_hypergraph
        heat_path = tmp_path / "test.heatmap.json"
        # Mirror the on-disk format TrainingHeatmap writes.
        heat_path.write_text(json.dumps({
            "entries": {
                eid: {"heat": v, "kind": "node", "ema_decay": 0.95,
                      "last_step": 1000}
                for eid, v in sample_heat_dict.items()
            }
        }), encoding="utf-8")
        dot = emit_dot_from_hypergraph(sample_ir, heat=str(heat_path))
        # The hot fill made it in.
        from neuroslm.compiler.heat_overlay import heat_to_fillcolor
        assert heat_to_fillcolor(1.0).lower() in dot.lower()


# ── CLI integration ──────────────────────────────────────────────


class TestCliHeatFlag:
    def test_cli_compile_nfg_with_heat_writes_png(
            self, sample_heat_dict, tmp_path, monkeypatch):
        """`brian compile nfg --heat <path>` produces a (DOT) artifact."""
        # We render to DOT format so the test passes without graphviz `dot`
        # binary being installed.
        from neuroslm.cli import _build_parser
        # Write a tiny arch
        arch_root = tmp_path / "tiny"
        arch_root.mkdir()
        (arch_root / "arch.neuro").write_text(SAMPLE_DSL, encoding="utf-8")
        # Write a heatmap
        heat_path = tmp_path / "heat.json"
        heat_path.write_text(
            json.dumps({"entries": {
                eid: {"heat": v, "kind": "node", "ema_decay": 0.95,
                      "last_step": 1000}
                for eid, v in sample_heat_dict.items()
            }}),
            encoding="utf-8",
        )
        # Output
        out_dot = tmp_path / "out.dot"

        # We don't actually invoke the full CLI; we check the parser
        # accepts the flag and that an internal renderer picks it up.
        parser = _build_parser()
        args = parser.parse_args([
            "compile", "nfg", str(arch_root),
            "--out", str(out_dot),
            "--format", "dot",
            "--heat", str(heat_path),
        ])
        assert args.heat == str(heat_path)
        # Run the dispatched function.
        rc = args.func(args)
        assert rc == 0
        assert out_dot.exists()
        from neuroslm.compiler.heat_overlay import heat_to_fillcolor
        text = out_dot.read_text(encoding="utf-8")
        assert heat_to_fillcolor(1.0).lower() in text.lower()
