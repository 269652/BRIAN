# -*- coding: utf-8 -*-
"""Graphviz-backed Neural Flow Graph emitter.

The hypergraph IR (:mod:`neuroslm.compiler.hypergraph_ir`) is the source
of truth for everything declared in the architecture: populations,
neurotransmitters, synapses, modulations. This module turns that IR into
a Graphviz ``Digraph`` with proper hierarchical layout (``dot``),
subgraph clusters by node kind, and labels carrying the salient
properties (count / dt / equation / weight / NT / gain / effect).

Two public helpers:

  :func:`emit_dot_from_hypergraph`
      Pure string output (no I/O, no external binary required) — useful
      for tests, snapshotting, and the ``brian compile nfg --out X.dot``
      path.

  :func:`render_hypergraph`
      Writes ``.dot`` / ``.png`` / ``.svg`` etc. to disk. PNG/SVG require
      the Graphviz ``dot`` binary on ``$PATH``.

The visual grammar (chosen so the diagram stays readable on architectures
with 30+ populations, 7 NT systems, 40+ edges):

  populations          → box, light blue, layered by topological position
  neurotransmitters    → ellipse, light yellow, gathered in a cluster
  architecture decl    → folder shape, light grey, top of diagram
  synapses             → solid black arrows, label = nt + weight
  modulations          → dashed coloured arrows (per source NT), label =
                         effect + gain
"""
from __future__ import annotations

from typing import Any, Dict, Iterable, List, Optional

from neuroslm.compiler.hypergraph_ir import HypergraphIR, HyperNode, HyperEdge


# Visual constants — single source of truth so renderer changes are local.

_NT_COLOURS = {
    "dopamine":          "#d62728",  # red
    "norepinephrine":    "#ff7f0e",  # orange
    "serotonin":         "#9467bd",  # purple
    "acetylcholine":     "#2ca02c",  # green
    "endocannabinoid":   "#8c564b",  # brown
    "glutamate":         "#1f77b4",  # blue
    "gaba":              "#7f7f7f",  # grey
}
_DEFAULT_MOD_COLOUR = "#444444"

_KIND_STYLES = {
    "population":       dict(shape="box",     style="filled,rounded",
                             fillcolor="#cfe8ff", color="#1f4d75"),
    "neurotransmitter": dict(shape="ellipse", style="filled",
                             fillcolor="#fff6c2", color="#7a6b00"),
    "architecture":     dict(shape="folder",  style="filled",
                             fillcolor="#eeeeee", color="#333333"),
    "cortex_expert":    dict(shape="box3d",   style="filled",
                             fillcolor="#f5e8ff", color="#7a2b9c"),
    "lm_trunk":         dict(shape="component", style="filled",
                             fillcolor="#ffd6f0", color="#7a2b9c"),
}


# ── Anatomical / functional clustering ─────────────────────────────────
#
# rcc_bowtie declares ~33 populations across two ``param_scope`` blocks
# (``trunk`` and ``bio``). Dumping every population into one giant
# ``cluster_populations`` makes the diagram unreadable — basal ganglia,
# memory system and neuromodulator nuclei all blur together. The map
# below routes each named population into an anatomical / functional
# region cluster instead. Populations not in any explicit region fall
# back to ``other_trunk`` / ``other_bio`` based on their ``param_scope``.

_ANATOMICAL_REGIONS: Dict[str, set] = {
    "multi_cortex":  {"cortex_math", "cortex_code",
                      "cortex_chat", "cortex_general"},
    "basal_ganglia": {"bg", "vta", "nucleus_accumbens", "substantia_nigra"},
    "memory":        {"hippo", "entorhinal", "cerebellum"},
    "nuclei":        {"locus_coeruleus", "raphe_nuclei", "nucleus_basalis"},
    "limbic":        {"amygdala", "insula"},
    "sensory_motor": {"sensory", "association", "thalamus", "motor"},
    "executive":     {"pfc", "acc", "dmn", "gws",
                      "forward_m", "evaluator",
                      "thought_transformer", "claustrum",
                      "reasoning_cortex"},
    "self_world":    {"world", "self_m", "qualia", "neural_geometry"},
}

_REGION_STYLES: Dict[str, dict] = {
    "multi_cortex":  dict(color="#7a2b9c", fillcolor="#f5e8ff",
                          label="multi-cortex (GPT-2 experts)"),
    "basal_ganglia": dict(color="#cc4400", fillcolor="#fff0e0",
                          label="basal ganglia"),
    "memory":        dict(color="#0a6e3d", fillcolor="#e7fbf0",
                          label="memory system (hippo · entorhinal · cerebellum)"),
    "nuclei":        dict(color="#7a6b00", fillcolor="#fffbe6",
                          label="neuromodulator nuclei"),
    "limbic":        dict(color="#b00060", fillcolor="#ffe8f3",
                          label="limbic"),
    "sensory_motor": dict(color="#1f4d75", fillcolor="#e8f1fb",
                          label="sensory · motor"),
    "executive":     dict(color="#244466", fillcolor="#eef3f9",
                          label="executive trunk"),
    "self_world":    dict(color="#444444", fillcolor="#f4f4f4",
                          label="self · world models"),
    "other_trunk":   dict(color="#666666", fillcolor="#f7f7f7",
                          label="other trunk"),
    "other_bio":     dict(color="#aa5500", fillcolor="#fff5ec",
                          label="other bio"),
}

# Render order — multi_cortex is rendered separately (it holds the
# cortex_expert kind, not population), the rest of the list controls the
# order in which population subclusters appear in the DOT output.
_REGION_RENDER_ORDER: List[str] = [
    "sensory_motor", "executive", "self_world", "memory",
    "limbic", "basal_ganglia", "nuclei", "other_trunk", "other_bio",
]


def _classify_population(pop: HyperNode) -> str:
    """Map a population HyperNode to its anatomical cluster name."""
    for region, names in _ANATOMICAL_REGIONS.items():
        if pop.name in names:
            return region
    scope = (pop.attrs or {}).get("param_scope")
    if scope == "bio":
        return "other_bio"
    return "other_trunk"


def _abbrev(value: object, limit: int = 24) -> str:
    """Truncate long property values so labels stay readable."""
    s = str(value).strip()
    if len(s) <= limit:
        return s
    return s[: limit - 1] + "…"


def _escape_label(text: str) -> str:
    """Escape characters that have special meaning inside a Graphviz label."""
    return (text
            .replace("\\", "\\\\")
            .replace('"', '\\"')
            .replace("\n", "\\l"))


def _population_label(node: HyperNode) -> str:
    """Multiline label: name + (count) + dynamics/equation snippet."""
    lines = [node.name]
    attrs = node.attrs or {}
    count = attrs.get("count")
    if count is not None:
        lines.append(f"n={_abbrev(count, 10)}")
    # Prefer an explicit equation/ode/dynamics over the generic "dynamics"
    # field when present — math first, label second.
    for key in ("equation", "ode", "dynamics"):
        if key in attrs and attrs[key]:
            lines.append(f"{key}: {_abbrev(attrs[key], 32)}")
            break
    timescale = attrs.get("timescale")
    if timescale is not None:
        lines.append(f"τ={_abbrev(timescale, 10)}")
    return _escape_label("\n".join(lines))


def _nt_label(node: HyperNode) -> str:
    lines = [node.name]
    a = node.attrs or {}
    for key in ("base_concentration", "release_rate",
                "reuptake_rate", "diffusion_rate"):
        if key in a:
            short = key.split("_")[0]  # base / release / reuptake / diffusion
            lines.append(f"{short}={_abbrev(a[key], 10)}")
    return _escape_label("\n".join(lines))


def _arch_label(node: HyperNode) -> str:
    lines = [f"architecture: {node.name}"]
    a = node.attrs or {}
    for key in ("d_sem", "dt", "preset", "vocab_size", "n_layers"):
        if key in a:
            lines.append(f"{key}={_abbrev(a[key], 14)}")
    return _escape_label("\n".join(lines))


def _expert_label(node: HyperNode) -> str:
    """Cortex expert label: name + domain + GPT-2 backbone + freeze flag."""
    lines = [node.name]
    a = node.attrs or {}
    domain = a.get("domain")
    if domain:
        lines.append(f"domain: {domain}")
    weights = a.get("weights")
    if weights:
        lines.append(f"weights: {_abbrev(weights, 14)}")
    if a.get("freeze_weights", "").lower() in ("true", "1"):
        lines.append("(frozen backbone)")
    count = a.get("count")
    if count is not None:
        lines.append(f"n={_abbrev(count, 10)}")
    return _escape_label("\n".join(lines))


def _trunk_label(node: HyperNode) -> str:
    """LM trunk label: anchor node for the GPT-2 ensemble's KL+α edges."""
    a = node.attrs or {}
    lines = ["lm_trunk", "(language-model trunk)"]
    n_cx = a.get("n_cortices")
    if n_cx:
        lines.append(f"fuses {n_cx} cortices")
    return _escape_label("\n".join(lines))


def _edge_label(edge: HyperEdge) -> str:
    """Compact label: kind-specific salient attrs only."""
    a = edge.attrs or {}
    parts: list[str] = []
    if edge.kind == "synapse":
        if "neurotransmitter" in a:
            parts.append(str(a["neurotransmitter"]))
        if "weight" in a:
            parts.append(f"w={_abbrev(a['weight'], 8)}")
    elif edge.kind == "modulation":
        if "effect" in a:
            parts.append(str(a["effect"]))
        if "gain" in a:
            parts.append(f"g={_abbrev(a['gain'], 8)}")
    return _escape_label(" ".join(parts))


def _nt_colour(name: str) -> str:
    """Look up a deterministic colour for a neurotransmitter name."""
    key = (name or "").lower()
    return _NT_COLOURS.get(key, _DEFAULT_MOD_COLOUR)


def _graph_attrs(engine: str) -> dict:
    """Top-level graph attributes — engine-specific tweaks."""
    base = dict(
        labelloc="t",
        fontsize="22",
        fontname="Helvetica",
        bgcolor="white",
        pad="0.5",
    )
    if engine == "dot":
        base.update(rankdir="LR", nodesep="0.35", ranksep="0.6",
                    overlap="false", splines="spline")
    elif engine == "neato":
        base.update(overlap="prism", splines="curved", sep="+12")
    elif engine in {"sfdp", "fdp"}:
        base.update(overlap="prism", splines="true", K="1.4")
    return base


def emit_dot_from_hypergraph(
    ir: HypergraphIR,
    *,
    engine: str = "dot",
    title: Optional[str] = None,
    heat: Optional[Any] = None,
) -> str:
    """Build a DOT-string view of the hypergraph.

    Args:
        ir:      the hypergraph (typically from
                 :func:`neuroslm.compiler.hypergraph_ir.lift_arch_to_hypergraph`)
        engine:  layout engine name baked into the output — ``"dot"`` for
                 layered hierarchical (default), ``"neato"`` / ``"sfdp"`` /
                 ``"fdp"`` for force-directed.
        title:   optional graph title displayed at the top.
        heat:    optional heatmap overlay — accepts a ``dict[str, float]``,
                 a :class:`TrainingHeatmap` instance, or a path to the
                 JSON file produced by :class:`HeatmapPublisher`. Element
                 ids that match a population / synapse / modulation are
                 retinted by their normalized heat. See
                 :mod:`neuroslm.compiler.heat_overlay`.

    Returns:
        A complete DOT source string ready for ``dot -Tpng`` or
        :class:`graphviz.Source`.
    """
    try:
        import graphviz  # local — keep import optional
    except ImportError as exc:  # pragma: no cover — covered by skipif
        raise ImportError(
            "the 'graphviz' Python package is required: pip install graphviz"
        ) from exc

    # Resolve heat source -> flat normalized dict (or {} on None).
    from neuroslm.compiler.heat_overlay import (
        load_heat_source, heat_to_fillcolor,
    )
    heat_map: Dict[str, float] = load_heat_source(heat)

    g = graphviz.Digraph("nfg", engine=engine)
    g.attr(**_graph_attrs(engine))
    if title:
        g.attr(label=_escape_label(title))

    # 1. Architecture cluster (single node, top)
    arch_nodes = [n for n in ir.nodes if n.kind == "architecture"]
    if arch_nodes:
        with g.subgraph(name="cluster_architecture") as c:
            c.attr(label="architecture", style="rounded,dashed",
                   color="#888888", fontsize="14", bgcolor="#fafafa")
            for n in arch_nodes:
                style = _KIND_STYLES["architecture"]
                c.node(n.name, label=_arch_label(n), **style)

    # 2. Populations grouped by anatomical / functional region
    pop_nodes = [n for n in ir.nodes if n.kind == "population"]
    by_region: Dict[str, List[HyperNode]] = {}
    for n in pop_nodes:
        by_region.setdefault(_classify_population(n), []).append(n)

    for region in _REGION_RENDER_ORDER:
        nodes_in_region = by_region.get(region, [])
        if not nodes_in_region:
            continue
        style = _REGION_STYLES.get(region, _REGION_STYLES["other_trunk"])
        with g.subgraph(name=f"cluster_{region}") as c:
            c.attr(label=style["label"], style="rounded,dashed",
                   color=style["color"], fontsize="13",
                   bgcolor=style["fillcolor"])
            for n in nodes_in_region:
                # Tint each population's fill by its region so they stay
                # visually anchored to the cluster even when the layout
                # engine routes edges across cluster boundaries.
                pop_style = dict(_KIND_STYLES["population"])
                pop_style["color"]     = style["color"]
                pop_style["fillcolor"] = style["fillcolor"]
                # Heat-overlay: hot populations get a warm thermal fill.
                eid = f"population:{n.name}"
                if eid in heat_map and heat_map[eid] > 0.0:
                    pop_style["fillcolor"] = heat_to_fillcolor(heat_map[eid])
                c.node(n.name, label=_population_label(n), **pop_style)

    # 2b. Multi-cortex cluster — cortex_expert nodes + lm_trunk anchor.
    #     This is the visual home of the GPT-2 ensemble plus the
    #     distillation / inhibition edges; rendered as a separate
    #     cluster so the diagram makes the bowtie topology obvious
    #     (cortex experts on one side, trunk in the middle).
    expert_nodes = [n for n in ir.nodes if n.kind == "cortex_expert"]
    trunk_nodes  = [n for n in ir.nodes if n.kind == "lm_trunk"]
    if expert_nodes or trunk_nodes:
        style = _REGION_STYLES["multi_cortex"]
        with g.subgraph(name="cluster_multi_cortex") as c:
            c.attr(label=style["label"], style="rounded,solid",
                   color=style["color"], fontsize="13",
                   bgcolor=style["fillcolor"])
            for n in trunk_nodes:
                trunk_style = dict(_KIND_STYLES["lm_trunk"])
                c.node(n.name, label=_trunk_label(n), **trunk_style)
            for n in expert_nodes:
                expert_style = dict(_KIND_STYLES["cortex_expert"])
                c.node(n.name, label=_expert_label(n), **expert_style)

    # 3. Neurotransmitter cluster
    nt_nodes = [n for n in ir.nodes if n.kind == "neurotransmitter"]
    if nt_nodes:
        with g.subgraph(name="cluster_neurotransmitters") as c:
            c.attr(label="neurotransmitters", style="rounded,dashed",
                   color="#7a6b00", fontsize="14", bgcolor="#fffcec")
            for n in nt_nodes:
                style = dict(_KIND_STYLES["neurotransmitter"])
                # Tint the NT node by its semantic colour so the modulator
                # arrows it emits stay visually anchored.
                colour = _nt_colour(n.name)
                style["color"] = colour
                g_node = c
                g_node.node(n.name, label=_nt_label(n), **style)

    # 4. Synapse edges (solid)
    for edge in ir.hyperedges:
        if edge.kind != "synapse":
            continue
        if len(edge.members) < 2:
            continue
        src, dst = edge.members[0], edge.members[1]
        a = edge.attrs or {}
        nt_name = (a.get("neurotransmitter") or "").lower()
        colour = _nt_colour(nt_name) if nt_name else "#222222"
        # Derive arrow weight from the synaptic weight when present so
        # strong projections visually dominate.
        try:
            w = float(a.get("weight", 1.0))
        except (TypeError, ValueError):
            w = 1.0
        penwidth = f"{max(0.6, min(4.0, abs(w) * 1.6)):.2f}"
        # Heat-overlay: hot synapses repaint with the thermal color
        # AND get a thicker pen so they pop visually.
        eid = f"synapse:{src}->{dst}"
        if eid in heat_map and heat_map[eid] > 0.0:
            h = heat_map[eid]
            colour = heat_to_fillcolor(h)
            penwidth = f"{max(1.4, 1.4 + 3.0 * h):.2f}"
        g.edge(src, dst,
               label=_edge_label(edge),
               color=colour,
               fontcolor=colour,
               fontsize="9",
               penwidth=penwidth,
               arrowsize="0.7")

    # 5. Modulation edges (dashed, NT-coloured)
    for edge in ir.hyperedges:
        if edge.kind != "modulation":
            continue
        if len(edge.members) < 2:
            continue
        src, dst = edge.members[0], edge.members[1]
        colour = _nt_colour(src)
        penwidth = "1.0"
        # Heat-overlay: hot modulations also repaint thermal.
        eid = f"modulation:{src}->{dst}"
        if eid in heat_map and heat_map[eid] > 0.0:
            h = heat_map[eid]
            colour = heat_to_fillcolor(h)
            penwidth = f"{max(1.2, 1.2 + 2.5 * h):.2f}"
        g.edge(src, dst,
               label=_edge_label(edge),
               color=colour,
               fontcolor=colour,
               fontsize="9",
               style="dashed",
               arrowhead="vee",
               penwidth=penwidth,
               arrowsize="0.7")

    # 6. Distillation edges (Slot A: KL distillation, expert -> trunk).
    #    Drawn bold purple so they pop out of the population/synapse
    #    mass — these are aux-loss edges, not neural projections.
    for edge in ir.hyperedges:
        if edge.kind != "distillation":
            continue
        if len(edge.members) < 2:
            continue
        src, dst = edge.members[0], edge.members[1]
        a = edge.attrs or {}
        parts = ["KL distill"]
        if "lambda_max" in a:
            parts.append(f"λ_max={a['lambda_max']}")
        if "temperature" in a:
            parts.append(f"T={a['temperature']}")
        g.edge(src, dst,
               label=_escape_label("\n".join(parts)),
               color="#7a2b9c",
               fontcolor="#7a2b9c",
               fontsize="9",
               style="bold",
               penwidth="2.0",
               arrowsize="0.9",
               arrowhead="normal")

    # 7. Inhibition edges (Slot C: NT-gated α inhibition, trunk -> expert).
    #    Dashed orange with a `tee` arrowhead to read as "blocking".
    for edge in ir.hyperedges:
        if edge.kind != "inhibition":
            continue
        if len(edge.members) < 2:
            continue
        src, dst = edge.members[0], edge.members[1]
        a = edge.attrs or {}
        parts = ["NT-gated α inhibit"]
        if "ema_alpha" in a:
            parts.append(f"α_ema={a['ema_alpha']}")
        g.edge(src, dst,
               label=_escape_label("\n".join(parts)),
               color="#cc4400",
               fontcolor="#cc4400",
               fontsize="9",
               style="dashed",
               penwidth="1.4",
               arrowsize="0.7",
               arrowhead="tee")

    return g.source


def render_hypergraph(
    ir: HypergraphIR,
    out_path: str,
    *,
    format: str = "png",
    engine: str = "dot",
    title: Optional[str] = None,
    heat: Optional[Any] = None,
) -> str:
    """Render the hypergraph to disk.

    Args:
        ir:        the hypergraph
        out_path:  destination path; suffix is overridden by *format* unless
                   *format* is ``"dot"`` (in which case the DOT text is
                   written verbatim — no external binary needed).
        format:    ``"png"`` / ``"svg"`` / ``"pdf"`` / ``"dot"``
        engine:    Graphviz layout engine (``"dot"`` default, also
                   ``"neato"`` / ``"sfdp"`` / ``"fdp"`` / ``"circo"``)
        title:     optional title for the rendered diagram.
        heat:      optional heatmap overlay (see :func:`emit_dot_from_hypergraph`).

    Returns:
        Absolute path to the file actually written.
    """
    from pathlib import Path as _Path

    dot_src = emit_dot_from_hypergraph(
        ir, engine=engine, title=title, heat=heat)
    out = _Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)

    if format == "dot":
        out.write_text(dot_src, encoding="utf-8")
        return str(out.resolve())

    # Defer to graphviz Source for non-text formats (uses subprocess `dot`).
    try:
        import graphviz
    except ImportError as exc:  # pragma: no cover
        raise ImportError(
            "the 'graphviz' Python package is required: pip install graphviz"
        ) from exc

    src = graphviz.Source(dot_src, engine=engine)
    # graphviz.Source.render() writes <out>.<format> — strip the suffix from
    # *out_path* so the final file lands exactly where the caller asked.
    stem = out.with_suffix("")
    rendered = src.render(filename=str(stem), format=format, cleanup=True)
    return str(_Path(rendered).resolve())


def render_arch(
    arch_root,
    out_path: str,
    *,
    format: str = "png",
    engine: str = "dot",
    title: Optional[str] = None,
    heat: Optional[Any] = None,
) -> str:
    """Convenience: lift an arch folder and render in one call."""
    from neuroslm.compiler.hypergraph_ir import lift_arch_to_hypergraph
    ir = lift_arch_to_hypergraph(arch_root)
    if title is None:
        try:
            from pathlib import Path as _Path
            title = f"NFG · {_Path(arch_root).name}"
        except Exception:
            title = None
    return render_hypergraph(ir, out_path, format=format,
                             engine=engine, title=title, heat=heat)
