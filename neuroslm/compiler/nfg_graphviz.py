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
    """Look up a deterministic colour for a neurotransmitter name.

    The DSL stores NT names as quoted strings
    (``neurotransmitter: "glutamate"``) and the raw attr value carries
    those quotes through to the hypergraph IR. Without stripping them
    here, every glutamate synapse in the master arch silently falls
    through to ``_DEFAULT_MOD_COLOUR`` (grey) instead of rendering as
    the proper blue ``#1f77b4`` — which is exactly the visual bug
    surfaced by the NFG audit on ``nfg_rcc_bowtie.png``.
    """
    key = (name or "").lower().strip().strip('"').strip("'")
    return _NT_COLOURS.get(key, _DEFAULT_MOD_COLOUR)


def _graph_attrs(engine: str, dpi: int = 96) -> dict:
    """Top-level graph attributes — engine-specific tweaks.

    Parameters
    ----------
    engine
        Layout engine — ``"dot"`` / ``"neato"`` / ``"sfdp"`` / ``"fdp"``.
    dpi
        PNG rasterization DPI. Graphviz default is 96; bumping to 150+
        produces sharper text/lines when zooming. SVG / PDF ignore this
        attribute (they're vector formats). Pass via ``[nfg].dpi`` in
        brian.toml or the ``BRIAN_NFG_DPI`` env var.
    """
    base = dict(
        labelloc="t",
        fontsize="22",
        fontname="Helvetica",
        bgcolor="white",
        pad="0.5",
        dpi=str(int(dpi)),
    )
    if engine == "dot":
        # Vanilla dot — let it pick its natural canvas. The back-edge
        # cycle-breaker (see _compute_back_edges + the synapse loop in
        # emit_dot_from_hypergraph) is what makes dot work at all on
        # this cyclic graph: forward synapses carry constraint=true and
        # contribute to ranking, DFS-detected back-edges get
        # constraint=false so they render without confusing dot's
        # rank algorithm.
        #
        # newrank=true uses dot's per-node-rank algorithm (vs the old
        # cluster-rank default), which behaves better on graphs with
        # mixed constrained / unconstrained edges.
        base.update(rankdir="TB", nodesep="0.4", ranksep="0.8",
                    overlap="false", splines="spline",
                    newrank="true", concentrate="true")
    elif engine == "neato":
        base.update(overlap="prism", splines="curved", sep="+12")
    elif engine in {"sfdp", "fdp"}:
        base.update(overlap="prism", splines="curved", K="1.4", sep="+10",
                    size="16,20", ratio="compress")
    return base


def _compute_back_edges(ir: HypergraphIR) -> set:
    """Identify back-edges in the synapse graph via iterative 3-colour DFS.

    A back-edge ``(u, v)`` is one such that ``v`` is currently on the DFS
    stack — i.e. removing it breaks a cycle. The renderer marks these
    ``constraint="false"`` so they appear in the image but don't confuse
    dot's hierarchical rank algorithm.

    All other synapse edges become rank-contributing (``constraint="true"``),
    which means dot computes the vertical strata directly from the
    architecture's real connectivity rather than from a hand-coded anatomy
    table. New modules slot into the right layer automatically based on
    how far they sit from the synaptic sinks.

    Determinism
    -----------
    Result is stable given the IR's hyperedge order (which itself is
    stable across compiles since arch.neuro is parsed deterministically).
    Two-cycle ties (``A↔B``) resolve to whichever direction DFS visits
    first — typically the order the synapses were declared in arch.neuro.

    Cost
    ----
    O(V+E) — single DFS pass. For 31 populations × 61 synapses runs in
    microseconds. Iterative implementation avoids recursion-limit risk
    on deep architectures.
    """
    # Build adjacency list over populations only. Modulation, distillation
    # and inhibition edges are already rank-noise (constraint=false in
    # their emission loops), so they don't need cycle-breaking.
    adj: Dict[str, List[str]] = {}
    for edge in ir.hyperedges:
        if edge.kind != "synapse":
            continue
        if len(edge.members) < 2:
            continue
        src, dst = edge.members[0], edge.members[1]
        adj.setdefault(src, []).append(dst)
        adj.setdefault(dst, [])  # ensure dst is reachable from the state map

    WHITE, GRAY, BLACK = 0, 1, 2
    state: Dict[str, int] = {n: WHITE for n in adj}
    back_edges: set = set()

    # Iterative DFS using an explicit (node, child-iterator) stack.
    for start in list(state.keys()):
        if state[start] != WHITE:
            continue
        state[start] = GRAY
        stack: List = [(start, iter(adj[start]))]
        while stack:
            u, it = stack[-1]
            try:
                v = next(it)
            except StopIteration:
                state[u] = BLACK
                stack.pop()
                continue
            s = state.get(v, WHITE)
            if s == GRAY:
                back_edges.add((u, v))
            elif s == WHITE:
                state[v] = GRAY
                stack.append((v, iter(adj.get(v, []))))
            # s == BLACK: cross/forward edge to a finished subtree, ignore.
    return back_edges


# ── Cluster row-wrapping ───────────────────────────────────────────────
#
# Clusters with too many populations (e.g. "executive" has 9 modules:
# evaluator / forward_m / reasoning_cortex / pfc / acc / dmn / gws /
# claustrum / thought_transformer) make the canvas painfully wide when
# laid out as a single horizontal row. We wrap into multiple rank=same
# rows once the cluster crosses ``_WRAP_THRESHOLD`` nodes — invisible
# constraint edges between row anchors force vertical stacking inside
# the cluster, so the cluster grows DOWNWARD instead of stretching
# rightward off the canvas.
#
# Threshold = 5 keeps clusters comfortably wide while preventing the
# "one cluster takes 60 % of the canvas width" problem. To change
# globally, edit this constant; to expose per-cluster, promote to
# ``[nfg].wrap_threshold`` in brian.toml and thread through.
_WRAP_THRESHOLD = 5


def _emit_cluster_rows(c, items: List, prefix: str,
                       wrap: int = _WRAP_THRESHOLD) -> None:
    """Emit ``items`` into the cluster subgraph ``c``, wrapping rows
    once ``len(items)`` exceeds ``wrap``.

    Parameters
    ----------
    c
        The cluster's :class:`graphviz.Digraph` subgraph context.
    items
        List of ``(name, label, style_dict)`` tuples. ``name`` is the
        node id, ``label`` the rendered label string, ``style_dict``
        the kwargs passed to ``c.node(...)``.
    prefix
        Unique string used to name the nested anonymous row subgraphs
        (e.g. ``"executive"`` → ``"_row_executive_0"``). Must be unique
        per cluster to avoid name collisions.
    wrap
        Maximum nodes per row. Defaults to :data:`_WRAP_THRESHOLD`.

    Behaviour
    ---------
    * ``len(items) <= wrap`` → single ``rank=same`` row inside the
      cluster (unchanged from the pre-wrap behaviour).
    * ``len(items) >  wrap`` → ⌈len/wrap⌉ rank-bands stacked
      vertically; invisible high-weight edges between consecutive
      rows' first nodes constrain the ordering so dot lays them top
      to bottom inside the cluster bounding box.
    """
    if len(items) <= wrap:
        # Fast path — preserves the original single-row behaviour
        # exactly (no nested subgraph, no invisible glue edges).
        c.attr(rank="same")
        for name, label, style in items:
            c.node(name, label=label, **style)
        return

    # Wrap path — chunk into rows of `wrap` and emit each as its own
    # anonymous rank=same subgraph nested inside the cluster.
    chunks = [items[i:i + wrap] for i in range(0, len(items), wrap)]
    for ci, chunk in enumerate(chunks):
        with c.subgraph(name=f"_row_{prefix}_{ci}") as row:
            row.attr(rank="same")
            for name, label, style in chunk:
                row.node(name, label=label, **style)

    # Invisible glue between row anchors. Without this, dot is free
    # to put both rows on the same actual rank (defeating the wrap),
    # or to flip them. weight=100 makes the constraint near-rigid;
    # style=invis keeps the line out of the final image; minlen=1
    # gives one rank's worth of vertical separation between rows.
    for ci in range(len(chunks) - 1):
        c.edge(chunks[ci][0][0],            # name = items[ci][0][0]
               chunks[ci + 1][0][0],
               style="invis",
               constraint="true",
               weight="100",
               minlen="1")


def emit_dot_from_hypergraph(
    ir: HypergraphIR,
    *,
    engine: str = "dot",
    title: Optional[str] = None,
    heat: Optional[Any] = None,
    dpi: int = 96,
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
        dpi:     PNG rasterization DPI baked into the DOT source. Default
                 96 (graphviz default); bump to 150+ for sharper PNG.
                 Ignored by SVG / PDF outputs.

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
    g.attr(**_graph_attrs(engine, dpi=dpi))
    if title:
        g.attr(label=_escape_label(title))

    # 1. Architecture cluster (single node, top)
    arch_nodes = [n for n in ir.nodes if n.kind == "architecture"]
    if arch_nodes:
        with g.subgraph(name="cluster_architecture") as c:
            c.attr(label="architecture", style="rounded,dashed",
                   color="#888888", fontsize="14", bgcolor="#fafafa")
            # rank=same forces all nodes in this cluster onto a single
            # horizontal row (the cluster becomes a wide horizontal
            # band). Combined with rankdir=TB at the graph level, the
            # clusters STACK top-to-bottom as horizontal strata →
            # landscape canvas. Without rank=same, dot lays each
            # cluster's nodes onto distinct ranks (turning every
            # cluster into a vertical column → portrait canvas).
            c.attr(rank="same")
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
            # Build (name, label, style_dict) items for the helper.
            # Tint each population's fill by its region so they stay
            # visually anchored to the cluster even when the layout
            # engine routes edges across cluster boundaries.
            region_items: List = []
            for n in nodes_in_region:
                pop_style = dict(_KIND_STYLES["population"])
                pop_style["color"]     = style["color"]
                pop_style["fillcolor"] = style["fillcolor"]
                # Heat-overlay: hot populations get a warm thermal fill.
                eid = f"population:{n.name}"
                if eid in heat_map and heat_map[eid] > 0.0:
                    pop_style["fillcolor"] = heat_to_fillcolor(heat_map[eid])
                region_items.append(
                    (n.name, _population_label(n), pop_style))
            # Wraps to multiple rows when len > _WRAP_THRESHOLD.
            # "executive" (9 modules) wraps to 5+4 = 2 rows; smaller
            # clusters stay as a single horizontal band.
            _emit_cluster_rows(c, region_items, prefix=region)

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
            # Build (name, label, style_dict) items — trunk first so
            # it anchors the row, then the cortex experts. Goes
            # through _emit_cluster_rows so behaviour is uniform
            # with the anatomical region clusters above.
            cortex_items: List = []
            for n in trunk_nodes:
                trunk_style = dict(_KIND_STYLES["lm_trunk"])
                cortex_items.append(
                    (n.name, _trunk_label(n), trunk_style))
            for n in expert_nodes:
                expert_style = dict(_KIND_STYLES["cortex_expert"])
                cortex_items.append(
                    (n.name, _expert_label(n), expert_style))
            _emit_cluster_rows(c, cortex_items, prefix="multi_cortex")

    # 3. Neurotransmitter cluster
    nt_nodes = [n for n in ir.nodes if n.kind == "neurotransmitter"]
    if nt_nodes:
        with g.subgraph(name="cluster_neurotransmitters") as c:
            c.attr(label="neurotransmitters", style="rounded,dashed",
                   color="#7a6b00", fontsize="14", bgcolor="#fffcec")
            # Build (name, label, style_dict) items — each NT keeps
            # its semantic colour so the modulator arrows it emits
            # stay visually anchored. 7 NTs > _WRAP_THRESHOLD (5),
            # so the cluster wraps to 5+2 across 2 rows.
            nt_items: List = []
            for n in nt_nodes:
                nt_style = dict(_KIND_STYLES["neurotransmitter"])
                nt_style["color"] = _nt_colour(n.name)
                nt_items.append((n.name, _nt_label(n), nt_style))
            _emit_cluster_rows(c, nt_items, prefix="neurotransmitters")

    # 3b. Cycle-breaking — let dot rank from the real synapse DAG.
    #
    #   We compute the back-edges (those that close cycles) of the
    #   synapse subgraph via DFS and mark *only* those constraint=false
    #   when emitting them below. All other synapse edges become
    #   rank-contributing, which means dot derives the vertical strata
    #   directly from the architecture's real connectivity.
    #
    #   No hand-coded anatomy table — the layout adapts to any arch.
    #   For brian.master today this produces a sensory→thalamus→cortex
    #   →executive→motor flow because that's what the DAG implies after
    #   removing the qualia↔dmn / NT-feedback back-edges.
    back_edges: set = _compute_back_edges(ir)

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
        # Forward synapses contribute to vertical ranking; back-edges
        # (those that close a cycle, detected in step 3b) get
        # constraint=false so they render but don't break dot's
        # acyclic rank algorithm.
        is_back = (src, dst) in back_edges
        g.edge(src, dst,
               label=_edge_label(edge),
               color=colour,
               fontcolor=colour,
               fontsize="9",
               penwidth=penwidth,
               arrowsize="0.7",
               constraint="false" if is_back else "true")

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
               arrowsize="0.7",
               constraint="false")

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
               arrowhead="normal",
               constraint="false")

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
               arrowhead="tee",
               constraint="false")

    return g.source


def render_hypergraph(
    ir: HypergraphIR,
    out_path: str,
    *,
    format: str = "png",
    engine: str = "dot",
    title: Optional[str] = None,
    heat: Optional[Any] = None,
    dpi: int = 96,
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
        dpi:       PNG rasterization DPI (default 96). Ignored by SVG/PDF.

    Returns:
        Absolute path to the file actually written.
    """
    from pathlib import Path as _Path

    dot_src = emit_dot_from_hypergraph(
        ir, engine=engine, title=title, heat=heat, dpi=dpi)
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
    dpi: int = 96,
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
                             engine=engine, title=title, heat=heat, dpi=dpi)
