# -*- coding: utf-8 -*-
"""Neural Flow Graph — full architectural blueprint (IR + PNG).

The NFG is the *visual ground-truth* of the architecture: anything declared
in arch.neuro (populations, synapses, modulations, NT kinetics, param
scopes, training config, MAT-gated mechanisms, pass marks, formal specs,
sheaves) is faithfully shown on the rendered diagram, so the full
mathematical + ML pipeline is deducible from the graph alone.

Panels:
    [main]     anatomical brain map (populations + synapses + modulations)
    [meta]     architecture name, d_sem, dt, total counts, param scopes
    [train]    optimizer, LR, weight decay, batch, seq_len, steps, clipping
    [mech]     each MAT-gated mechanism with its phase_gate(MAT) curve
    [nt]       7-NT kinetics table (base, release, reuptake, diffusion)
    [pass]     pass_marks rules
    [formal]   formal_spec + sheaf declarations
    [legend]   colours + edge styles
"""
from __future__ import annotations
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple


# ── 0a. Locked rendering spec ─────────────────────────────────────────
# NFGRenderSpec captures every visual grammar constant that was converged
# on over the v35-v37 iteration cycle. Passing a spec to render_nfg()
# guarantees a stable, reproducible "house render" for any preset that
# inherits from rcc_bowtie.
#
# To create a variant: dataclasses.replace(RCC_BOWTIE_SPEC, mod_top_k=3)
# To freeze a new preset:  save as a module-level constant below.

@dataclass(frozen=True)
class NFGRenderSpec:
    # ── Envelope grammar ─────────────────────────────────────────────
    envelope_pad:         float = 0.78    # horizontal padding around member nodes
    envelope_pad_v_frac:  float = 0.68    # vertical pad = pad * this fraction
    envelope_rounding:    float = 0.55    # corner rounding (data units, uniform)
    envelope_border_lw:   float = 1.3     # border linewidth (uniform)
    envelope_dash:        tuple = (0, (5, 3))  # dash pattern (uniform)
    envelope_fill_alpha:  float = 0.22    # fill alpha (uniform)
    envelope_label_fs:    float = 8.0     # label fontsize (uniform)
    # ── Modulation rendering ─────────────────────────────────────────
    mod_top_k:            int   = 2       # full-prominence edges per target
    mod_primary_alpha:    float = 0.58    # branch alpha for primary arcs
    mod_demoted_alpha:    float = 0.22    # stub alpha for demoted arcs
    mod_demoted_reach:    float = 0.18    # stub length as fraction of edge length
    mod_rail_frac:        float = 0.40    # rail = this fraction toward centroid
    # ── Synapse rendering ─────────────────────────────────────────────
    syn_forward_lw_scale: float = 2.4     # lw = max(1.1, weight * this)
    syn_feedback_lw_scale:float = 2.0
    syn_hub_min_indeg:    int   = 3       # port-order hub threshold
    syn_port_spread:      float = 0.60    # port arc spread: [-spread/2, +spread/2]
    # ── Edge label suppression in dense trunk zone ───────────────────
    # Edge weight labels are suppressed when BOTH endpoints are within
    # trunk_label_zone_x of each other AND their midpoint y is within
    # trunk_label_zone_y of 0 (the spine y).  This declutters
    # thalamus/GWS/PFC without removing labels on peripheral edges.
    trunk_label_zone_x:   float = 4.0    # half-width of suppression zone (x)
    trunk_label_zone_y:   float = 1.2    # half-height of suppression zone (y)
    # ── Spine band ───────────────────────────────────────────────────
    spine_band_alpha:     float = 0.55
    spine_band_color:     str   = "#fef9e7"
    # ── Node sizes ───────────────────────────────────────────────────
    node_base_size:       int   = 1800
    node_fan_scale:       int   = 220    # size += fan_scale * min(fan,10)
    node_alpha:           float = 0.90
    # ── Figure ───────────────────────────────────────────────────────
    figsize:              tuple = (26, 16)


# The canonical locked spec for the rcc_bowtie preset family (v37 baseline).
RCC_BOWTIE_SPEC = NFGRenderSpec()


# ── 0b. IR data classes (lightweight projection of compiler.ProgramIR) ─

@dataclass
class NFGNode:
    name: str
    kind: str                      # "pop" | "nt"
    op: str = ""
    equation: Optional[str] = None
    properties: Dict = field(default_factory=dict)


@dataclass
class NFGEdge:
    src: str
    tgt: str
    kind: str                      # "synapse" | "modulation"
    weight: float = 1.0
    nt: Optional[str] = None
    effect: Optional[str] = None
    equation: Optional[str] = None


@dataclass
class NeuralFlowGraph:
    arch_name: str
    nodes: List[NFGNode] = field(default_factory=list)
    edges: List[NFGEdge] = field(default_factory=list)
    # Architecture-level metadata pulled from `architecture { ... }` block.
    architecture_meta: Dict = field(default_factory=dict)
    # NT system full kinetics (so the diagram can render the table even
    # when a NT isn't currently modulating anything).
    nt_systems: List[Dict] = field(default_factory=list)
    # Param scope membership (declarative trunk vs bio split).
    param_scopes: List[Dict] = field(default_factory=list)
    # Formal constraint systems (sheaves + formal_specs).
    formal_specs: List[Dict] = field(default_factory=list)
    sheaves: List[Dict] = field(default_factory=list)

    def stats(self) -> Dict[str, int]:
        return {
            "n_populations": sum(1 for n in self.nodes if n.kind == "pop"),
            "n_neurotransmitters": len(self.nt_systems),
            "n_synapses": sum(1 for e in self.edges if e.kind == "synapse"),
            "n_modulations": sum(1 for e in self.edges if e.kind == "modulation"),
        }


# ── 1. Compile arch → NFG ─────────────────────────────────────────────

def _classify_op(equation: Optional[str], ode: Optional[str]) -> str:
    if ode:
        return "ode"
    if not equation:
        return "linear"
    e = equation.lower()
    if "softmax" in e and "relu" in e:
        return "softmax_relu"
    if "softmax" in e:
        return "softmax"
    if "sigmoid" in e and ("relu" in e or "max" in e):
        return "gated"
    if "tanh" in e:
        return "tanh"
    if "relu" in e or "max(0" in e:
        return "relu"
    return "linear"


def compile_nfg(arch_root) -> NeuralFlowGraph:
    """Build a Neural Flow Graph from an architecture folder."""
    from .multifile import compile_folder
    from .param_scopes import load_param_scopes_from_arch
    ir = compile_folder(Path(arch_root))

    name = Path(arch_root).name or "?"
    g = NeuralFlowGraph(arch_name=name)

    # Populations
    for pop in ir.populations:
        op = _classify_op(getattr(pop, "equation", None),
                          getattr(pop, "ode", None))
        g.nodes.append(NFGNode(
            name=pop.name, kind="pop", op=op,
            equation=pop.equation or pop.ode,
            properties={"count": pop.count,
                        "dynamics": pop.dynamics,
                        "timescale": getattr(pop, "timescale", None),
                        "output_dim": getattr(pop, "output_dim", None)},
        ))

    # NT systems — kept regardless of whether they currently modulate
    nt_used = {m.source_nt for m in ir.modulations}
    for nt in ir.neurotransmitter_systems:
        # Append as IR-derived dict so the table renderer sees ALL fields,
        # not just the small subset embedded as graph nodes.
        g.nt_systems.append({
            "name": nt.name,
            "base":     nt.base_concentration,
            "release":  nt.release_rate,
            "reuptake": nt.reuptake_rate,
            "diffusion": nt.diffusion_rate,
            "used": nt.name in nt_used,
        })
        if nt.name in nt_used:
            g.nodes.append(NFGNode(
                name=nt.name, kind="nt", op="nt_dynamics",
                properties={"base": nt.base_concentration,
                            "release": nt.release_rate,
                            "reuptake": nt.reuptake_rate,
                            "diffusion": nt.diffusion_rate},
            ))

    # Synapses
    for syn in ir.synapses:
        g.edges.append(NFGEdge(
            src=syn.source, tgt=syn.target, kind="synapse",
            weight=float(syn.weight) if syn.weight is not None else 1.0,
            nt=syn.neurotransmitter,
            equation=getattr(syn, "equation", None),
        ))

    # Modulations — NT → population
    for mod in ir.modulations:
        g.edges.append(NFGEdge(
            src=mod.source_nt, tgt=mod.target_population, kind="modulation",
            weight=float(mod.gain) if mod.gain is not None else 1.0,
            nt=mod.source_nt, effect=mod.effect,
            equation=getattr(mod, "equation", None),
        ))

    # Architecture-level metadata (d_sem, dt, ...)
    g.architecture_meta = dict(getattr(ir, "architecture", {}) or {})
    # Resolver also stores it under arch.architecture — fall back via folder.
    try:
        from .multifile import Resolver
        program = Resolver(Path(arch_root)).resolve()
        if not g.architecture_meta and program.architecture:
            g.architecture_meta = dict(program.architecture)
    except Exception:
        pass

    # Param scopes
    try:
        for sc in load_param_scopes_from_arch(arch_root):
            g.param_scopes.append({
                "name": sc.name,
                "populations": list(sc.populations),
                "gradient": sc.gradient,
            })
    except Exception:
        pass

    # Formal specs + sheaves
    for fs in getattr(ir, "formal_specs", None) or []:
        g.formal_specs.append({
            "name": fs.name,
            "spec_type": getattr(fs, "spec_type", "generic"),
            "properties": dict(getattr(fs, "properties", {}) or {}),
        })
    for sh in getattr(ir, "sheaf_specs", None) or []:
        g.sheaves.append({
            "name": sh.name,
            "contradiction_threshold": getattr(sh, "contradiction_threshold", None),
            "mechanism": getattr(sh, "mechanism", None),
            "action": getattr(sh, "action", None),
        })

    return g


# ── 2. Layered layout fallback ────────────────────────────────────────

def _layered_positions(g: NeuralFlowGraph) -> Dict[str, Tuple[float, float]]:
    succ: Dict[str, List[str]] = {}
    pred: Dict[str, List[str]] = {}
    pop_names = [n.name for n in g.nodes if n.kind == "pop"]
    for e in g.edges:
        if e.kind != "synapse":
            continue
        succ.setdefault(e.src, []).append(e.tgt)
        pred.setdefault(e.tgt, []).append(e.src)
    sources = [n for n in pop_names if n not in pred]
    depth: Dict[str, int] = {n: 0 for n in sources}
    frontier = list(sources)
    while frontier:
        nf = []
        for u in frontier:
            for v in succ.get(u, []):
                d = depth.get(u, 0) + 1
                if v not in depth or depth[v] < d:
                    depth[v] = d
                    nf.append(v)
        frontier = nf
    max_d = max(depth.values()) if depth else 0
    for n in pop_names:
        depth.setdefault(n, max_d + 1)
    by_depth: Dict[int, List[str]] = {}
    for n, d in depth.items():
        by_depth.setdefault(d, []).append(n)
    pos: Dict[str, Tuple[float, float]] = {}
    for d in sorted(by_depth):
        col = sorted(by_depth[d])
        for i, n in enumerate(col):
            x = (i - (len(col) - 1) / 2) * 1.5
            y = -d * 1.2
            pos[n] = (x, y)
    nts = [n.name for n in g.nodes if n.kind == "nt"]
    for i, nt in enumerate(nts):
        x = max([p[0] for p in pos.values()], default=0) + 3.5
        y = -(i / max(1, len(nts) - 1)) * (len(by_depth) - 1) * 1.2 if len(nts) > 1 else 0
        pos[nt] = (x, y)
    return pos


# ── 3. Render to PNG ──────────────────────────────────────────────────

_REGION_COLORS = {
    "input":      "#3498db",
    "thalamic":   "#9b59b6",
    "cortex":     "#f39c12",
    "memory":     "#2ecc71",
    "subcort":    "#16a085",
    "world":      "#7f8c8d",
    "output":     "#e74c3c",
    "nuclei":     "#fdb6c8",
}
_REGION_OF = {
    "sensory": "input", "association": "input",
    "thalamus": "thalamic",
    "pfc": "cortex", "acc": "cortex", "dmn": "cortex",
    "gws": "cortex", "claustrum": "cortex", "neural_geometry": "cortex",
    "qualia": "cortex", "thought_transformer": "cortex",
    "math_cortex": "cortex", "reasoning_cortex": "cortex",
    "language_cortex": "cortex",
    "hippo": "memory", "entorhinal": "memory", "cerebellum": "memory",
    "amygdala": "subcort", "insula": "subcort", "bg": "subcort",
    "forward_m": "subcort", "evaluator": "subcort",
    "world": "world", "self_m": "world",
    "motor": "output",
    "vta": "nuclei", "nucleus_accumbens": "nuclei",
    "locus_coeruleus": "nuclei", "raphe_nuclei": "nuclei",
    "nucleus_basalis": "nuclei", "substantia_nigra": "nuclei",
}

_RESERVED_SLOTS = {
    "sensory":    (-7.0,  1.5), "association":(-5.5,  1.5),
    "thalamus":   (-3.5,  0.0), "gws":        (-1.0,  0.0),
    "pfc":        ( 1.5,  0.0), "bg":         ( 4.0,  0.0),
    "motor":      ( 6.5,  0.0),
    "acc":        ( 1.5,  2.0), "dmn":        (-1.0,  2.5),
    "claustrum":  ( 0.0,  2.0), "thought_transformer": (1.0, 2.5),
    "hippo":      ( 0.5, -2.0), "entorhinal": (-0.5, -2.0),
    "cerebellum": ( 3.0, -2.5),
    "world":      (-3.5,  2.5), "self_m":     (-3.5, -2.5),
    "qualia":     (-1.0, -2.5), "neural_geometry": (-2.0, 2.5),
    "amygdala":   (-5.5, -2.0), "insula":     (-7.0, -2.0),
    "forward_m":  ( 4.0, -2.0), "evaluator":  ( 5.5, -2.0),
    "math_cortex":      ( 4.0,  2.5),
    "reasoning_cortex": ( 3.0,  2.0),
    "language_cortex":  ( 5.5,  2.5),
}

_REGION_CENTROIDS = {
    "input":    (-6.5,  1.0), "thalamic": (-3.5,  0.0),
    "cortex":   ( 0.0,  2.0), "memory":   ( 0.0, -2.0),
    "subcort":  ( 4.0, -1.5), "world":    (-3.5,  0.0),
    "output":   ( 6.5,  0.0), "nuclei":   ( 7.5,  2.5),
}

_NUCLEI_RING = {
    # Layout pass 3: place each nucleus near its DOMINANT influence domain
    # rather than parked on a right-edge column.
    #   VTA / SN     → near BG / PFC          (right-of-waist, top)
    #   nucleus_accumbens → BG cluster        (right-of-waist, lower)
    #   locus_coeruleus   → thalamus/sensory  (top-left, arousal axis)
    #   raphe_nuclei      → DMN / self-model  (above-left, affective loop)
    #   nucleus_basalis   → PFC / cortex      (above PFC, attention/gating)
    "vta":                ( 3.5,  3.3),     # near BG/PFC top
    "substantia_nigra":   ( 5.0,  3.3),     # next to VTA
    "nucleus_accumbens":  ( 5.0,  1.2),     # near BG
    "locus_coeruleus":    (-4.5,  3.3),     # near thalamus, top-left
    "raphe_nuclei":       (-2.5,  3.0),     # near DMN
    "nucleus_basalis":    ( 2.0,  3.3),     # above PFC
}

_NT_SLOTS = {
    "dopamine":         ( 2.5,  3.5),
    "norepinephrine":   (-2.0,  3.5),
    "serotonin":        ( 0.0,  4.0),
    "acetylcholine":    ( 4.5,  3.5),
    "endocannabinoid":  (-6.0,  3.5),
    "glutamate":        ( 0.0, -4.0),
    "gaba":             (-3.5, -3.5),
}

_NT_ABBREV = {
    "dopamine": "DA", "norepinephrine": "NE", "serotonin": "5HT",
    "acetylcholine": "ACh", "endocannabinoid": "eCB",
    "glutamate": "Glu", "gaba": "GABA",
}

# ── Canonical slot templates for known preset families ─────────────────
# When g.arch_name contains the key string the matching template REPLACES
# _RESERVED_SLOTS so recurring presets always render with the same stable
# house layout regardless of graph-force relaxation.
_PRESET_TEMPLATES: Dict[str, Dict[str, Tuple[float, float]]] = {
    "rcc_bowtie": {
        # ── main spine (y=0) — wider spacing to reduce trunk crowding ─
        "sensory":     (-8.0,  0.0), "association":  (-6.2,  0.0),
        "thalamus":    (-4.2,  0.0), "gws":          (-1.4,  0.0),
        "pfc":         ( 1.6,  0.0), "bg":           ( 4.4,  0.0),
        "motor":       ( 7.2,  0.0),
        # ── upper cortical/cognitive loop ─────────────────────────────
        "acc":              ( 1.6,  2.4), "dmn":           (-1.4,  2.9),
        "claustrum":        ( 0.2,  2.4),
        "thought_transformer": ( 0.8,  3.0),
        "reasoning_cortex": ( 3.2,  2.4), "language_cortex": ( 6.0,  3.0),
        "math_cortex":      ( 4.6,  3.0),
        # ── memory cluster ────────────────────────────────────────────
        "hippo":       ( 0.5, -2.4), "entorhinal":   (-0.7, -2.4),
        "cerebellum":  ( 3.4, -3.0),
        # ── self / world / predictive-control cluster ─────────────────
        "world":       (-4.2,  2.8), "self_m":       (-4.2, -3.0),
        "qualia":      (-1.4, -2.9), "neural_geometry": (-2.4,  2.9),
        "forward_m":   ( 4.4, -2.4), "evaluator":    ( 6.0, -2.4),
        # ── subcortical / interoceptive ───────────────────────────────
        "amygdala":    (-6.2, -2.4), "insula":       (-8.0, -2.4),
    },
}

# ── Subsystem envelope definitions ────────────────────────────────────
# Each tuple: (label, [member_pop_names], fill_color, border_color)
# Used by _draw_subsystem_envelopes() to draw faint grouped regions.
_SUBSYSTEM_ENVELOPES: List[Tuple] = [
    ("memory",
     ["hippo", "entorhinal"],
     "#d5f5e3", "#27ae60"),
    ("self-model",
     ["self_m", "qualia", "world", "neural_geometry"],
     "#d6eaf8", "#2980b9"),
    ("predictive ctrl",
     ["forward_m", "evaluator", "cerebellum"],
     "#fdf2f8", "#8e44ad"),
    ("cortical loop",
     ["acc", "dmn", "claustrum", "thought_transformer",
      "language_cortex", "math_cortex", "reasoning_cortex"],
     "#fef9e7", "#f39c12"),
    ("interoceptive",
     ["amygdala", "insula"],
     "#fce4ec", "#c0392b"),
]


def _derive_spine(g: "NeuralFlowGraph") -> List[str]:
    """Derive the input→...→output spine from the synapse graph itself.

    Strategy: find the longest simple directed path through the weighted
    synapse DAG, breaking ties by total weight. Source candidates are
    populations with in-degree 0 (or smallest in-degree); sinks are
    out-degree 0 (or smallest out-degree).

    Falls back to the rcc_bowtie default spine when the architecture
    has no clear longest path (e.g. fully-disconnected graphs).
    """
    pops = [n.name for n in g.nodes if n.kind == "pop"]
    # Feedback-arc set via BFS spanning tree: start from probable inputs
    # (lowest in-degree), BFS forward, label any edge that goes to an
    # already-discovered node as a "back-edge" only when removing it
    # keeps the rest of the graph reachable. Avoids the over-tagging
    # problem of "any edge participating in any cycle".
    adj_full: Dict[str, set] = {}
    raw_in: Dict[str, int] = {p: 0 for p in pops}
    for e in g.edges:
        if e.kind == "synapse":
            adj_full.setdefault(e.src, set()).add(e.tgt)
            raw_in[e.tgt] = raw_in.get(e.tgt, 0) + 1
    # Seed BFS from populations with the smallest in-degree first
    bfs_order = sorted(pops, key=lambda p: raw_in.get(p, 0))
    discovered_at: Dict[str, int] = {}
    counter = [0]
    visited: set = set()
    for seed in bfs_order:
        if seed in visited:
            continue
        front = [seed]
        while front:
            nxt = []
            for u in front:
                if u in visited:
                    continue
                visited.add(u)
                discovered_at[u] = counter[0]
                counter[0] += 1
                for v in adj_full.get(u, ()):
                    if v not in visited:
                        nxt.append(v)
            front = nxt
    # An edge u→v is a back-edge iff discovered_at[v] < discovered_at[u]
    # (v was discovered before u in the BFS order — it's a return-jump).
    cycle_edges = {(e.src, e.tgt) for e in g.edges
                   if e.kind == "synapse"
                   and e.src in discovered_at and e.tgt in discovered_at
                   and discovered_at[e.tgt] <= discovered_at[e.src]}
    # Reentry-tail detection: any edge whose target has out_deg=0 AND
    # in_deg=1 in the forward graph is a "reentry tail" (e.g. motor→sensory)
    # — we want sensory as a LOGICAL INPUT, not as the spine's sink.
    pre_in: Dict[str, int] = {p: 0 for p in pops}
    pre_out: Dict[str, int] = {p: 0 for p in pops}
    for e in g.edges:
        if e.kind != "synapse" or (e.src, e.tgt) in cycle_edges:
            continue
        pre_in[e.tgt] = pre_in.get(e.tgt, 0) + 1
        pre_out[e.src] = pre_out.get(e.src, 0) + 1
    reentry_tail = {(e.src, e.tgt) for e in g.edges
                    if e.kind == "synapse"
                    and pre_out.get(e.tgt, 0) == 0
                    and pre_in.get(e.tgt, 0) == 1}

    succ: Dict[str, List[Tuple[str, float]]] = {}
    in_deg: Dict[str, int] = {p: 0 for p in pops}
    out_deg: Dict[str, int] = {p: 0 for p in pops}
    for e in g.edges:
        if e.kind != "synapse":
            continue
        if (e.src, e.tgt) in cycle_edges:
            continue   # skip back-edges
        if (e.src, e.tgt) in reentry_tail:
            continue   # skip reentry tails (motor→sensory etc.)
        succ.setdefault(e.src, []).append((e.tgt, e.weight))
        in_deg[e.tgt] = in_deg.get(e.tgt, 0) + 1
        out_deg[e.src] = out_deg.get(e.src, 0) + 1
    sources = [p for p in pops if in_deg.get(p, 0) == 0]
    if not sources:
        sources = sorted(pops, key=lambda p: in_deg.get(p, 0))[:2]
    best_path: List[str] = []
    best_score: float = -1
    for src in sources:
        # BFS-style longest-path search with cycle guard (graph is small)
        stack = [(src, [src], 0.0)]
        while stack:
            node, path, score = stack.pop()
            extended = False
            for tgt, w in succ.get(node, []):
                if tgt in path:
                    continue
                extended = True
                stack.append((tgt, path + [tgt], score + w))
            if not extended:
                if len(path) > len(best_path) or (
                        len(path) == len(best_path) and score > best_score):
                    best_path = path
                    best_score = score
    if len(best_path) >= 3:
        return best_path
    # Fallback when derivation fails
    return [p for p in ["sensory", "thalamus", "gws", "pfc", "bg", "motor"]
            if p in pops]


def _derive_nucleus_targets(g: "NeuralFlowGraph",
                             pos: Dict[str, Tuple[float, float]]
                            ) -> Dict[str, Tuple[float, float]]:
    """Place each modulatory nucleus near the centroid of its NT's targets.

    Uses the NT system already declared in arch.neuro:
      - `nucleus_produces` name pattern (e.g. vta → dopamine; raphe → serotonin)
      - the modulation edges (NT → population) determine the target set
      - the nucleus gets placed near the centroid of those populations
    """
    # Name-pattern based nucleus → NT mapping (well-known from neuroscience)
    nucleus_nt = {
        "vta":               "dopamine",
        "substantia_nigra":  "dopamine",
        "nucleus_accumbens": "dopamine",
        "locus_coeruleus":   "norepinephrine",
        "raphe_nuclei":      "serotonin",
        "nucleus_basalis":   "acetylcholine",
    }
    # Group nuclei by NT first so we can fan multiple nuclei sharing an NT
    by_nt: Dict[str, List[str]] = {}
    for nuc_name, nt_name in nucleus_nt.items():
        if any(n.name == nuc_name for n in g.nodes):
            by_nt.setdefault(nt_name, []).append(nuc_name)
    out: Dict[str, Tuple[float, float]] = {}
    import math as _m
    for nt_name, nuclei in by_nt.items():
        targets = [e.tgt for e in g.edges
                   if e.kind == "modulation" and e.src == nt_name
                   and e.tgt in pos]
        if not targets:
            continue
        cx = sum(pos[t][0] for t in targets) / len(targets)
        cy = sum(pos[t][1] for t in targets) / len(targets)
        # Fan multiple nuclei around the centroid so they don't overlap
        k = len(nuclei)
        radius = 1.4 if k > 1 else 0.0
        for i, nuc in enumerate(sorted(nuclei)):
            ang = _m.pi * (0.5 - (i - (k - 1) / 2) * 0.18)
            out[nuc] = (cx + radius * _m.cos(ang),
                         cy + 1.8 + radius * _m.sin(ang))
    return out


def _derive_regions(g: "NeuralFlowGraph", spine: List[str]
                    ) -> Dict[str, str]:
    """Bucket each population into a coarse role bucket derived from its
    position on the spine (input/integration/control/action) and its
    synaptic connectivity to spine nodes.

    Returns {pop_name → bucket} where bucket ∈ {"input", "integration",
    "control", "action", "memory", "subcort", "world", "nuclei"}.
    """
    nucleus_names = {"vta", "substantia_nigra", "nucleus_accumbens",
                      "locus_coeruleus", "raphe_nuclei", "nucleus_basalis"}
    spine_role: Dict[str, str] = {}
    if spine:
        n = len(spine)
        for i, name in enumerate(spine):
            frac = i / max(1, n - 1)
            if frac < 0.25:   spine_role[name] = "input"
            elif frac < 0.55: spine_role[name] = "integration"
            elif frac < 0.85: spine_role[name] = "control"
            else:             spine_role[name] = "action"
    out: Dict[str, str] = {}
    for n in g.nodes:
        if n.kind != "pop":
            continue
        if n.name in nucleus_names:
            out[n.name] = "nuclei"; continue
        if n.name in spine_role:
            out[n.name] = spine_role[n.name]; continue
        # Otherwise: classify by name patterns when DSL gives no hint
        out[n.name] = _REGION_OF.get(n.name, "world")
    return out


def _neuroanatomical_layout(g: "NeuralFlowGraph") -> Dict[str, Tuple[float, float]]:
    """DSL-derived layout — backbone-first, attachment-semantics, orbit.

    The spine is computed from the synapse graph (longest weighted path).
    Modulatory nucleus positions are derived from modulation-edge targets
    in the DSL. Region buckets are inferred from spine position +
    connectivity. Hardcoded `_RESERVED_SLOTS` / `_NUCLEI_RING` tables are
    ONLY used as fallbacks for populations/nuclei the derivation can't
    place (e.g. fully-disconnected architectures).
    """
    import math
    import random
    rng = random.Random(42)
    pos: Dict[str, Tuple[float, float]] = {}

    # Canonical preset template: use preset-family slots when the arch name
    # matches a known pattern — gives stable "house layout" per preset family.
    preset_slots: Dict[str, Tuple[float, float]] = {}
    arch_lower = g.arch_name.lower()
    for pat, slots in _PRESET_TEMPLATES.items():
        if pat in arch_lower:
            preset_slots = slots
            break
    _effective_slots = {**_RESERVED_SLOTS, **preset_slots}

    # 1. DSL-derived spine — lay it out left→right on y=0
    spine = _derive_spine(g)
    if spine:
        n = len(spine)
        x_step = 12.0 / max(1, n - 1)
        for i, name in enumerate(spine):
            pos[name] = (-6.0 + i * x_step, 0.0)
    # 2. Fallback pin for any spine population we couldn't derive
    for n in g.nodes:
        if n.name in _effective_slots and n.name not in pos:
            pos[n.name] = _effective_slots[n.name]
    # 3. DSL-derived nucleus placement (uses modulation edges)
    nuc_pos = _derive_nucleus_targets(g, pos)
    pos.update(nuc_pos)
    # Any nucleus still unplaced → hardcoded ring fallback
    for n in g.nodes:
        if n.name in _NUCLEI_RING and n.name not in pos:
            pos[n.name] = _NUCLEI_RING[n.name]
    # 3. Orphan pops — orbit around region centroid (peripheral ring)
    region_orphans: Dict[str, List[str]] = {}
    for n in g.nodes:
        if n.kind != "pop" or n.name in pos:
            continue
        region = _REGION_OF.get(n.name, "world")
        region_orphans.setdefault(region, []).append(n.name)
    for region, names in region_orphans.items():
        cx, cy = _REGION_CENTROIDS.get(region, (0.0, 0.0))
        # Distribute on a small circle (radius scales with count)
        k = len(names)
        radius = 0.9 + 0.15 * k
        for i, name in enumerate(sorted(names)):
            ang = 2 * math.pi * i / max(1, k)
            jx = radius * math.cos(ang) + 0.2 * rng.uniform(-1, 1)
            jy = radius * math.sin(ang) + 0.2 * rng.uniform(-1, 1)
            pos[name] = (cx + jx, cy + jy)
    for n in g.nodes:
        if n.kind != "nt" or n.name in pos:
            continue
        if n.name in _NT_SLOTS:
            pos[n.name] = _NT_SLOTS[n.name]
        else:
            angle_idx = len([k for k in pos if k.startswith("__nt_extra")])
            ang = 2 * math.pi * angle_idx / 8
            pos[n.name] = (9.0 * math.cos(ang), 9.0 * math.sin(ang))
    # De-overlap pass, but PIN the reserved spine — nuclei/orphans can
    # nudge, the bowtie backbone cannot.
    PINNED = set(_effective_slots.keys()) | set(_NT_SLOTS.keys())

    # Anchor pass: pull each non-pinned population toward the weighted
    # centroid of its direct synapse neighbours. This snaps peripheral
    # nodes (memory, interoceptive, self-model) closer to their cluster
    # without disturbing the pinned spine.
    ANCHOR_ALPHA = 0.22   # blend fraction (0=no pull, 1=fully at centroid)
    for _ in range(3):
        for n in g.nodes:
            if n.kind != "pop" or n.name not in pos or n.name in PINNED:
                continue
            nbr_wx, nbr_wy, total_w = 0.0, 0.0, 0.0
            for e in g.edges:
                if e.kind != "synapse":
                    continue
                w = max(e.weight, 0.1)
                if e.src == n.name and e.tgt in pos:
                    nbr_wx += pos[e.tgt][0] * w
                    nbr_wy += pos[e.tgt][1] * w
                    total_w += w
                elif e.tgt == n.name and e.src in pos:
                    nbr_wx += pos[e.src][0] * w
                    nbr_wy += pos[e.src][1] * w
                    total_w += w
            if total_w < 1e-6:
                continue
            cx, cy = nbr_wx / total_w, nbr_wy / total_w
            px, py = pos[n.name]
            pos[n.name] = (px + ANCHOR_ALPHA * (cx - px),
                           py + ANCHOR_ALPHA * (cy - py))

    MIN_SEP = 1.0
    names = list(pos.keys())
    for _ in range(3):
        for i, a in enumerate(names):
            for b in names[i + 1:]:
                ax, ay = pos[a]; bx, by = pos[b]
                dx, dy = bx - ax, by - ay
                d = (dx * dx + dy * dy) ** 0.5
                if d < MIN_SEP and d > 1e-3:
                    push = (MIN_SEP - d) / 2.0
                    ux, uy = dx / d, dy / d
                    if a not in PINNED:
                        pos[a] = (ax - ux * push, ay - uy * push)
                    if b not in PINNED:
                        pos[b] = (bx + ux * push, by + uy * push)
    return pos


def _inferred_arch_root(g: "NeuralFlowGraph") -> str:
    from pathlib import Path as _P
    here = _P(__file__).resolve().parent.parent.parent
    candidate = here / "architectures" / g.arch_name
    if candidate.is_dir():
        return str(candidate)
    return str(here / "architectures" / "rcc_bowtie")


# ── 3a. Helpers for rendering side panels ─────────────────────────────

def _fmt_si(x) -> str:
    try:
        v = float(x)
    except Exception:
        return str(x)
    if v >= 1e9:  return f"{v/1e9:.2f}G"
    if v >= 1e6:  return f"{v/1e6:.2f}M"
    if v >= 1e3:  return f"{v/1e3:.1f}k"
    if v >= 1:    return f"{v:.0f}"
    if v == 0:    return "0"
    return f"{v:.3g}"


def _phase_gate_curve(center: float, width: float, n: int = 50):
    """Return (xs, ys) for the phase_gate(MAT) curve."""
    import math
    xs = [i / (n - 1) for i in range(n)]
    ys = [0.5 * (1.0 + math.tanh((x - center) / max(1e-6, width))) for x in xs]
    return xs, ys


def _draw_subsystem_envelopes(ax, pos: Dict[str, Tuple[float, float]],
                               pad: float = 0.78,
                               spec: Optional["NFGRenderSpec"] = None) -> None:
    """Draw uniform dashed rounded-rect envelopes around subsystem clusters.

    Visual grammar rules are driven by NFGRenderSpec so every envelope is
    guaranteed identical styling. Shown only when ≥2 members have positions.
    """
    import matplotlib.patches as mpatches
    rs = spec if spec is not None else RCC_BOWTIE_SPEC
    _pad      = rs.envelope_pad
    _pad_v    = rs.envelope_pad * rs.envelope_pad_v_frac
    ROUNDING  = rs.envelope_rounding
    BORDER_LW = rs.envelope_border_lw
    DASH      = rs.envelope_dash
    FILL_A    = rs.envelope_fill_alpha
    LABEL_FS  = rs.envelope_label_fs

    for label, members, fc, ec in _SUBSYSTEM_ENVELOPES:
        pts = [pos[m] for m in members if m in pos]
        if len(pts) < 2:
            continue
        xs = [p[0] for p in pts]
        ys = [p[1] for p in pts]
        xmin, xmax = min(xs) - _pad, max(xs) + _pad
        ymin, ymax = min(ys) - _pad_v, max(ys) + _pad_v
        ax.add_patch(mpatches.FancyBboxPatch(
            (xmin, ymin), xmax - xmin, ymax - ymin,
            boxstyle=f"round,pad=0.05,rounding_size={ROUNDING}",
            linewidth=BORDER_LW, linestyle=DASH,
            edgecolor=ec, facecolor=fc,
            alpha=FILL_A, zorder=0))
        ax.text(xmin + 0.18, ymax - 0.08,
                label,
                ha="left", va="top",
                fontsize=LABEL_FS, fontweight="bold", color=ec,
                alpha=0.95, zorder=2,
                bbox=dict(boxstyle="round,pad=0.15", fc="white",
                          ec=ec, lw=0.8, alpha=0.85))


def _draw_main_graph(ax, g: "NeuralFlowGraph",
                     show_weights: bool, show_equations: bool,
                     spec: Optional["NFGRenderSpec"] = None):
    """Render the brain region graph onto the supplied Axes."""
    import matplotlib.patches as mpatches
    from matplotlib.patches import FancyArrowPatch

    rs = spec if spec is not None else RCC_BOWTIE_SPEC

    pos = _neuroanatomical_layout(g)
    if not pos:
        pos = _layered_positions(g)
    for n in g.nodes:
        pos.setdefault(n.name, (0.0, 0.0))

    # Spine highlight: draw a translucent band beneath the DSL-derived
    # primary pathway so the reader sees input → integration → control →
    # action at a glance.
    spine_nodes = _derive_spine(g)
    spine_pts = [pos[n] for n in spine_nodes if n in pos]
    if len(spine_pts) >= 2:
        import matplotlib.patches as _mp
        xs = [p[0] for p in spine_pts]
        ys = [p[1] for p in spine_pts]
        xmin, xmax = min(xs) - 0.6, max(xs) + 0.6
        ymin, ymax = min(ys) - 0.9, max(ys) + 0.9
        ax.add_patch(_mp.FancyBboxPatch(
            (xmin, ymin), xmax - xmin, ymax - ymin,
            boxstyle="round,pad=0.10,rounding_size=0.5",
            linewidth=0, facecolor=rs.spine_band_color,
            alpha=rs.spine_band_alpha, zorder=0))
        ax.text((xmin + xmax) / 2, ymax + 0.05,
                "input → integration → control → action",
                ha="center", va="bottom",
                fontsize=7.5, color="#7d6608", style="italic",
                zorder=1)

    # Subsystem envelopes — drawn before nodes so they sit in the background
    _draw_subsystem_envelopes(ax, pos, spec=rs)

    fan: Dict[str, int] = {}
    for e in g.edges:
        if e.kind == "synapse":
            fan[e.src] = fan.get(e.src, 0) + 1
            fan[e.tgt] = fan.get(e.tgt, 0) + 1

    # Cycle / feedback detection
    adj: Dict[str, set] = {}
    for e in g.edges:
        if e.kind == "synapse":
            adj.setdefault(e.src, set()).add(e.tgt)

    def _reaches(s, t, max_depth=12):
        if s == t: return True
        seen, front, depth = {s}, [s], 0
        while front and depth < max_depth:
            nxt = []
            for x in front:
                for y in adj.get(x, ()):
                    if y == t: return True
                    if y not in seen:
                        seen.add(y); nxt.append(y)
            front = nxt; depth += 1
        return False

    cycle_edges = set()
    for e in g.edges:
        if e.kind == "synapse" and _reaches(e.tgt, e.src):
            cycle_edges.add((e.src, e.tgt))

    # Populations
    pop_nodes = [n for n in g.nodes if n.kind == "pop"]
    # Find trunk vs bio for border colour
    trunk_set: set = set()
    bio_set: set = set()
    for sc in g.param_scopes:
        if sc.get("gradient", "normal") == "detached_from_main_loss":
            bio_set.update(sc.get("populations", []))
        else:
            trunk_set.update(sc.get("populations", []))

    for node in pop_nodes:
        x, y = pos[node.name]
        region = _REGION_OF.get(node.name, "world")
        color = _REGION_COLORS[region]
        f = fan.get(node.name, 0)
        size = rs.node_base_size + rs.node_fan_scale * min(f, 10)
        edge_c = "#2c3e50"
        edge_lw = 1.8
        if node.name in bio_set:
            edge_c, edge_lw = "#c0392b", 2.6   # bio → red border (detached)
        elif node.name in trunk_set:
            edge_c, edge_lw = "#1f618d", 2.2   # trunk → deep blue border
        ax.scatter([x], [y], s=size, c=color, edgecolors=edge_c,
                   linewidths=edge_lw, zorder=3, alpha=rs.node_alpha)
        ax.text(x, y, node.name, ha="center", va="center", fontsize=8.5,
                fontweight="bold", color="white", zorder=4)
        # Annotation pill: op + count/dim
        count = node.properties.get("count")
        out_d = node.properties.get("output_dim")
        bits = [node.op]
        if count is not None:
            bits.append(f"N={_fmt_si(count)}")
        if out_d:
            bits.append(f"d={out_d}")
        label = " · ".join(bits)
        ax.annotate(label, xy=(x, y), xytext=(0, -22),
                    textcoords="offset points",
                    ha="center", va="top", fontsize=6.0,
                    style="italic", color="#2c3e50", zorder=4,
                    bbox=dict(boxstyle="round,pad=0.18", fc="white",
                              ec=color, lw=1.0, alpha=0.9))

    # NT diamonds
    for node in [n for n in g.nodes if n.kind == "nt"]:
        x, y = pos[node.name]
        abbrev = _NT_ABBREV.get(node.name, node.name[:3])
        ax.scatter([x], [y], s=1200, c="#fff3a0", marker="D",
                   edgecolors="#b7950b", linewidths=1.5, zorder=3, alpha=0.92)
        ax.text(x, y, abbrev, ha="center", va="center", fontsize=8,
                fontweight="bold", color="#5d4501", zorder=4)

    # Port ordering for trunk hubs (thalamus, GWS, PFC, and any hub with
    # in-degree >= 3). Edges are sorted so that sources to the LEFT of the
    # hub enter from the left arc and sources to the RIGHT enter from the
    # right arc. This eliminates most left-right crossings at trunk nodes.
    in_deg: Dict[str, int] = {}
    for e in g.edges:
        if e.kind == "synapse":
            in_deg[e.tgt] = in_deg.get(e.tgt, 0) + 1
    hub_set = {n for n, d in in_deg.items() if d >= rs.syn_hub_min_indeg}
    # Compute per-hub ordering: sort sources by their x-position so the
    # leftmost source gets the most-negative curvature and the rightmost
    # gets the most-positive — edges "spread out" symmetrically rather than
    # all bunching into the same arc.
    hub_ports: Dict[Tuple[str, str], int] = {}
    for hub in hub_set:
        if hub not in pos:
            continue
        hx = pos[hub][0]
        sources = sorted(
            {e.src for e in g.edges if e.kind == "synapse" and e.tgt == hub},
            key=lambda s: pos.get(s, (0.0, 0.0))[0] - hx)
        for i, src in enumerate(sources):
            hub_ports[(src, hub)] = i

    # Synapses — primary pathway distinguished from feedback
    for e in g.edges:
        if e.kind != "synapse":
            continue
        if e.src not in pos or e.tgt not in pos:
            continue
        x0, y0 = pos[e.src]; x1, y1 = pos[e.tgt]
        is_cycle = (e.src, e.tgt) in cycle_edges
        if is_cycle:
            # Feedback loop — distinctive purple, still distinguishable
            lw = max(1.3, e.weight * 2.0)
            color = "#8e44ad"
            rad = 0.35 if (hash(e.src + e.tgt) & 1) else -0.35
            alpha = 0.75
        else:
            # Primary pathway — bolder + more opaque so spine reads first
            lw = max(1.1, e.weight * 2.4)
            color = "#1c2833"
            # Entry-port ordering for hub targets: rotate the curve so
            # each source enters from a distinct angle.
            if e.tgt in hub_set and (e.src, e.tgt) in hub_ports:
                n_in = max(1, in_deg[e.tgt])
                port = hub_ports[(e.src, e.tgt)]
                half = rs.syn_port_spread / 2
                rad = -half + rs.syn_port_spread * port / max(1, n_in - 1)
            else:
                rad = 0.10 if (hash(e.src + e.tgt) & 1) else -0.10
            alpha = 0.92
        arrow = FancyArrowPatch(
            (x0, y0), (x1, y1),
            arrowstyle="-|>", mutation_scale=14 if is_cycle else 13,
            color=color, lw=lw, alpha=alpha,
            connectionstyle=f"arc3,rad={rad}",
            shrinkA=22, shrinkB=22,
            zorder=4 if (not is_cycle) else 3)
        ax.add_patch(arrow)
        # Edge label: weight + NT abbrev.
        # Suppressed when both endpoints are inside the dense trunk zone
        # (midpoint within trunk_label_zone_x of center AND near y=0)
        # to reduce label competition around thalamus/GWS/PFC.
        if show_weights:
            mx, my = (x0 + x1) / 2, (y0 + y1) / 2
            in_trunk_zone = (
                abs(mx) < rs.trunk_label_zone_x
                and abs(my) < rs.trunk_label_zone_y
            )
            if not in_trunk_zone:
                label_bits = [f"w={e.weight:.2f}"]
                if e.nt:
                    label_bits.append(_NT_ABBREV.get(e.nt, e.nt))
                txt = " ".join(label_bits)
                ax.text(mx, my + (0.18 if rad >= 0 else -0.18),
                        txt, ha="center", va="center", fontsize=5.5,
                        color=color, zorder=4,
                        bbox=dict(boxstyle="round,pad=0.10", fc="white",
                                  ec="none", alpha=0.85))

    # Modulations — selective prominence rendering.
    #
    # Rule: for each target population, only the top-K incoming modulation
    # edges (ranked by |weight|) are drawn with full prominence (solid stem
    # + curved branch). The rest are drawn as a single faint short stub
    # directly from the NT diamond toward the target — visually present but
    # not competing for attention.
    #
    # This keeps the diagram semantically complete while avoiding the
    # "red canopy" look that comes from every long-range arc rising to the
    # same ceiling height simultaneously.
    MOD_TOP_K = rs.mod_top_k

    nt_outgoing: Dict[str, list] = {}
    for e in g.edges:
        if e.kind == "modulation" and e.src in pos and e.tgt in pos:
            nt_outgoing.setdefault(e.src, []).append(e)

    # Rank by weight per target to decide which edges are "primary"
    tgt_edges: Dict[str, list] = {}
    for edges in nt_outgoing.values():
        for e in edges:
            tgt_edges.setdefault(e.tgt, []).append(e)
    primary_edges: set = set()
    for tgt, edges in tgt_edges.items():
        top = sorted(edges, key=lambda e: abs(e.weight), reverse=True)[:MOD_TOP_K]
        for e in top:
            primary_edges.add(id(e))

    for nt_name, edges in nt_outgoing.items():
        nx, ny = pos[nt_name]
        # Rail: offset toward centroid of PRIMARY targets only (reduces
        # the rail being pulled toward a distant outlier).
        primary = [e for e in edges if id(e) in primary_edges]
        ref_edges = primary if primary else edges
        tx = sum(pos[e.tgt][0] for e in ref_edges) / len(ref_edges)
        ty = sum(pos[e.tgt][1] for e in ref_edges) / len(ref_edges)
        dx, dy = tx - nx, ty - ny
        d = (dx * dx + dy * dy) ** 0.5 or 1.0
        # Rail: 40% of the way toward primary-target centroid
        rail = (nx + rs.mod_rail_frac * dx / d, ny + rs.mod_rail_frac * dy / d)

        for e in edges:
            x1, y1 = pos[e.tgt]
            color = "#c0392b" if e.effect == "multiplicative" else "#2874a6"
            is_primary = id(e) in primary_edges

            if is_primary:
                # Full prominence: shared stem → curved branch
                stem = FancyArrowPatch(
                    (nx, ny), rail,
                    arrowstyle="-",
                    color=color, lw=0.9, alpha=0.50,
                    linestyle=(0, (3, 2)),
                    connectionstyle="arc3,rad=0",
                    shrinkA=14, shrinkB=0, zorder=1)
                ax.add_patch(stem)
                rad = 0.15 if (hash(e.src + e.tgt) & 1) else -0.15
                branch = FancyArrowPatch(
                    rail, (x1, y1),
                    arrowstyle="-|>", mutation_scale=10,
                    color=color, lw=0.9, alpha=rs.mod_primary_alpha,
                    linestyle=(0, (3, 2)),
                    connectionstyle=f"arc3,rad={rad}",
                    shrinkA=0, shrinkB=22, zorder=1)
                ax.add_patch(branch)
            else:
                # Demoted: single faint short stub pointing toward target
                mx = nx + rs.mod_demoted_reach * (x1 - nx) / max(d, 0.1)
                my = ny + rs.mod_demoted_reach * (y1 - ny) / max(d, 0.1)
                stub = FancyArrowPatch(
                    (nx, ny), (mx, my),
                    arrowstyle="-|>", mutation_scale=7,
                    color=color, lw=0.55, alpha=rs.mod_demoted_alpha,
                    linestyle=(0, (2, 3)),
                    connectionstyle="arc3,rad=0",
                    shrinkA=14, shrinkB=0, zorder=1)
                ax.add_patch(stub)

    # Legends inside the main axis (compact)
    region_handles = [mpatches.Patch(color=col, label=name)
                      for name, col in _REGION_COLORS.items()]
    edge_handles = [
        mpatches.Patch(color="#2c3e50", label="synapse (forward)"),
        mpatches.Patch(color="#8e44ad",
                        label=f"synapse (feedback ×{len(cycle_edges)})"),
        mpatches.Patch(color="#c0392b", label="modulation (multiplicative)"),
        mpatches.Patch(color="#2874a6", label="modulation (additive)"),
        mpatches.Patch(color="#fff3a0", label="neurotransmitter"),
    ]
    scope_handles = [
        mpatches.Patch(facecolor="white", edgecolor="#1f618d", lw=2,
                       label="trunk scope (LM grad)"),
        mpatches.Patch(facecolor="white", edgecolor="#c0392b", lw=2,
                       label="bio scope (detached)"),
    ]
    leg1 = ax.legend(handles=region_handles, loc="upper left", fontsize=7,
                     title="regions", framealpha=0.92, title_fontsize=8)
    ax.add_artist(leg1)
    leg2 = ax.legend(handles=edge_handles, loc="lower left", fontsize=7,
                     title="edges", framealpha=0.92, title_fontsize=8)
    ax.add_artist(leg2)
    ax.legend(handles=scope_handles, loc="lower right", fontsize=7,
              title="param_scope", framealpha=0.92, title_fontsize=8)
    ax.set_axis_off()
    return cycle_edges


def _panel_text(ax, title: str, lines: List[str], *,
                 font="monospace", color="#2c3e50",
                 boxcolor="#ecf0f1", titlecolor="#2c3e50"):
    """Render a titled text panel onto an axis (axis-off)."""
    ax.set_axis_off()
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    ax.add_patch(__import__("matplotlib").patches.FancyBboxPatch(
        (0.0, 0.0), 1.0, 1.0,
        boxstyle="round,pad=0.012,rounding_size=0.02",
        linewidth=1.0, edgecolor="#34495e", facecolor=boxcolor,
        alpha=0.95, transform=ax.transAxes))
    ax.text(0.5, 0.96, title, ha="center", va="top",
            fontsize=9, fontweight="bold", color=titlecolor,
            transform=ax.transAxes)
    body = "\n".join(lines)
    ax.text(0.04, 0.90, body, ha="left", va="top",
            fontsize=7.0, family=font, color=color,
            transform=ax.transAxes)


def _draw_mechanisms_panel(ax, tc):
    """Render each MAT-gated mechanism with its phase_gate(MAT) curve."""
    ax.set_axis_off()
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    ax.add_patch(__import__("matplotlib").patches.FancyBboxPatch(
        (0.0, 0.0), 1.0, 1.0,
        boxstyle="round,pad=0.012,rounding_size=0.02",
        linewidth=1.0, edgecolor="#34495e", facecolor="#ecf0f1",
        alpha=0.95, transform=ax.transAxes))
    ax.text(0.5, 0.97, "trunk mechanisms · MAT-phase-gated",
            ha="center", va="top", fontsize=9, fontweight="bold",
            color="#2c3e50", transform=ax.transAxes)
    ax.text(0.5, 0.92,
            "g(MAT) = ½·(1 + tanh((MAT − center)/width))",
            ha="center", va="top", fontsize=6.5,
            family="monospace", color="#7f8c8d",
            transform=ax.transAxes)

    mech_rows = []
    m = tc.mechanisms
    if m.dropout:
        s, gate = m.dropout
        mech_rows.append(("dropout",      f"s={s:.2f}",                  gate, "#7f8c8d"))
    if m.pct_trunk:
        s, gate = m.pct_trunk
        mech_rows.append(("PCT trunk",    f"s={s:.2f} α·∑Δ→top-down",     gate, "#2980b9"))
    if m.tonnetz:
        p, bw, gate = m.tonnetz
        mech_rows.append(("Tonnetz",      f"p={p} bw={bw} torus-mask",    gate, "#16a085"))
    if m.nemori:
        f0, gate = m.nemori
        mech_rows.append(("NEMORI gate",  f"floor={f0:.2f} (skip low-surprise)", gate, "#c0392b"))
    if m.bema:
        rw, gate = m.bema
        mech_rows.append(("BEMA optim.",  f"rollback={rw} steps on PPL rise",   gate, "#8e44ad"))

    if not mech_rows:
        ax.text(0.5, 0.50, "(no MAT-gated mechanisms declared)",
                ha="center", va="center", fontsize=8,
                color="#7f8c8d", transform=ax.transAxes)
        return

    n = len(mech_rows)
    row_h = 0.78 / n
    for i, (label, formula, gate, color) in enumerate(mech_rows):
        y0 = 0.88 - (i + 1) * row_h
        # Mini phase_gate curve plot inset
        xs, ys = _phase_gate_curve(gate.center, gate.width, n=40)
        # axis-space inset: plot the curve from x=0.55..0.96, y=y0+0.02..y0+row_h-0.04
        plot_x0, plot_x1 = 0.55, 0.96
        plot_y0, plot_y1 = y0 + 0.012, y0 + row_h - 0.020
        # Background
        ax.add_patch(__import__("matplotlib").patches.Rectangle(
            (plot_x0, plot_y0), plot_x1 - plot_x0, plot_y1 - plot_y0,
            transform=ax.transAxes,
            fc="white", ec="#bdc3c7", lw=0.6, alpha=0.95, zorder=2))
        # Curve
        for j in range(len(xs) - 1):
            xa = plot_x0 + xs[j] * (plot_x1 - plot_x0)
            xb = plot_x0 + xs[j + 1] * (plot_x1 - plot_x0)
            ya = plot_y0 + ys[j] * (plot_y1 - plot_y0)
            yb = plot_y0 + ys[j + 1] * (plot_y1 - plot_y0)
            ax.plot([xa, xb], [ya, yb], color=color, lw=1.4,
                    transform=ax.transAxes, zorder=3)
        # Center marker
        cx = plot_x0 + gate.center * (plot_x1 - plot_x0)
        ax.plot([cx, cx], [plot_y0, plot_y1],
                linestyle=":", color=color, lw=0.7,
                transform=ax.transAxes, zorder=3)
        # Mechanism label (left)
        ax.text(0.04, y0 + row_h / 2 + 0.012, label,
                ha="left", va="center", fontsize=7.5, fontweight="bold",
                color=color, transform=ax.transAxes)
        ax.text(0.04, y0 + row_h / 2 - 0.015, formula,
                ha="left", va="center", fontsize=6.2, family="monospace",
                color="#34495e", transform=ax.transAxes)
        # Gate params under the curve
        ax.text((plot_x0 + plot_x1) / 2, plot_y0 - 0.005,
                f"c={gate.center:.2f} w={gate.width:.2f}",
                ha="center", va="top", fontsize=5.6, family="monospace",
                color="#7f8c8d", transform=ax.transAxes)


def _draw_nt_table(ax, nt_systems: List[Dict]):
    ax.set_axis_off()
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    ax.add_patch(__import__("matplotlib").patches.FancyBboxPatch(
        (0.0, 0.0), 1.0, 1.0,
        boxstyle="round,pad=0.012,rounding_size=0.02",
        linewidth=1.0, edgecolor="#b7950b", facecolor="#fffbeb",
        alpha=0.95, transform=ax.transAxes))
    ax.text(0.5, 0.96, f"neurotransmitter kinetics · 7 systems",
            ha="center", va="top", fontsize=9, fontweight="bold",
            color="#5d4501", transform=ax.transAxes)
    header = f"{'name':<11} {'abbr':<5} {'base':>5} {'rel':>5} {'reup':>5} {'diff':>5}"
    ax.text(0.04, 0.88, header, ha="left", va="top",
            fontsize=6.5, family="monospace", fontweight="bold",
            color="#5d4501", transform=ax.transAxes)
    y = 0.83
    for s in nt_systems:
        abbr = _NT_ABBREV.get(s["name"], s["name"][:3])
        used_tag = "" if s.get("used") else "·"
        row = (f"{s['name']:<11} {abbr:<5} "
               f"{s.get('base',0) or 0:>5.2f} "
               f"{s.get('release',0) or 0:>5.2f} "
               f"{s.get('reuptake',0) or 0:>5.2f} "
               f"{s.get('diffusion',0) or 0:>5.3f} {used_tag}")
        ax.text(0.04, y, row, ha="left", va="top",
                fontsize=6.3, family="monospace",
                color="#5d4501", transform=ax.transAxes)
        y -= 0.085
    ax.text(0.5, 0.04,
            "dc/dt = release · drive − reuptake · c + diffusion · ∇²c",
            ha="center", va="bottom", fontsize=6.0, family="monospace",
            color="#b7950b", transform=ax.transAxes)


# ── 3b. Public render entry point ─────────────────────────────────────

def render_nfg(g: NeuralFlowGraph, output_path: str,
               figsize: Tuple[int, int] = (26, 16),
               layout: str = "neuroanatomical",
               show_weights: bool = True,
               show_equations: bool = False,
               spec: Optional[NFGRenderSpec] = None,
               **_ignored) -> None:
    """Render the NFG to PNG with the full sidebar of architecture metadata.

    Args:
        figsize:        figure size (defaults to 26x16 — large enough for sidebar)
        layout:         only "neuroanatomical" (the others are kept for
                        debugging via `_draw_main_graph` directly)
        show_weights:   annotate each synapse with its weight + NT abbrev
        show_equations: show the per-synapse `y = ...` equation on the edge
                        label (default off — usually identical y = w·(x@W))
        spec:           NFGRenderSpec controlling every visual grammar constant.
                        Defaults to RCC_BOWTIE_SPEC (the v37 locked baseline).
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.gridspec import GridSpec

    rs = spec if spec is not None else RCC_BOWTIE_SPEC

    # Try to load the training config + architecture for the sidebar
    arch_root = _inferred_arch_root(g)
    tc = None
    try:
        from neuroslm.dsl.training_config import load_training_config_from_arch
        tc = load_training_config_from_arch(arch_root)
    except Exception:
        tc = None

    fig = plt.figure(figsize=rs.figsize, facecolor="#fafbfc")
    # 3 columns: main graph (wide) | meta+train+mech | nt+pass+formal
    gs = GridSpec(
        nrows=8, ncols=3,
        width_ratios=[3.6, 1.0, 1.0],
        height_ratios=[0.55, 0.55, 0.85, 0.85, 0.85, 0.85, 0.85, 0.85],
        hspace=0.18, wspace=0.10,
        left=0.02, right=0.98, top=0.95, bottom=0.03,
    )

    # ── Title bar (spans all columns) ──
    title_ax = fig.add_subplot(gs[0, :])
    title_ax.set_axis_off()
    s = g.stats()
    cycle_count_placeholder = ""    # filled later if we want
    title_ax.text(0.5, 0.8,
        f"Neural Flow Graph — {g.arch_name}",
        ha="center", va="center", fontsize=16, fontweight="bold",
        color="#2c3e50", transform=title_ax.transAxes)
    # arch meta is stored as {name, properties:{d_sem, dt, ...}} — flatten
    arch_props = (g.architecture_meta or {}).get("properties", {}) or {}
    arch_bits = []
    for k in ("d_sem", "dt", "d_model", "n_layers", "n_heads"):
        v = arch_props.get(k) if arch_props.get(k) is not None else (g.architecture_meta or {}).get(k)
        if v not in (None, ""):
            arch_bits.append(f"{k}={v}")
    if tc and tc.preset:
        arch_bits.append(f"preset={tc.preset}")
    arch_str = " · ".join(arch_bits)
    title_ax.text(0.5, 0.20,
        f"{s['n_populations']} populations · {s['n_synapses']} synapses · "
        f"{s['n_modulations']} modulations · {s['n_neurotransmitters']} NT systems"
        + (f"   |   {arch_str}" if arch_str else ""),
        ha="center", va="center", fontsize=10, color="#7f8c8d",
        transform=title_ax.transAxes)

    # ── Main brain map (rows 1..7, col 0) ──
    main_ax = fig.add_subplot(gs[1:8, 0])
    main_ax.set_facecolor("#fafbfc")
    _draw_main_graph(main_ax, g,
                     show_weights=show_weights,
                     show_equations=show_equations,
                     spec=rs)

    # ── Sidebar column 1 (meta / training / mechanisms) ──
    # [row 1]: meta panel
    meta_ax = fig.add_subplot(gs[1, 1])
    meta_lines = []
    arch_name = (g.architecture_meta or {}).get("name") or g.arch_name
    meta_lines.append(f"name    = {arch_name}")
    for k in ("d_sem", "dt", "d_model", "n_layers", "n_heads"):
        v = arch_props.get(k) if arch_props.get(k) is not None else (g.architecture_meta or {}).get(k)
        if v is not None:
            meta_lines.append(f"{k:<8}= {v}")
    # Total counts
    total_units = sum((n.properties.get("count") or 0) for n in g.nodes if n.kind == "pop")
    meta_lines.append(f"Σunits   = {_fmt_si(total_units)}")
    # Param scopes
    for sc in g.param_scopes:
        meta_lines.append(f"scope {sc['name']:<5}: {len(sc['populations'])} pops "
                          f"({'detached' if sc['gradient']=='detached_from_main_loss' else 'normal'})")
    _panel_text(meta_ax, "architecture meta", meta_lines or ["(no metadata)"],
                boxcolor="#eaf2f8", titlecolor="#1f618d")

    # [row 2]: training panel
    train_ax = fig.add_subplot(gs[2, 1])
    train_lines = []
    if tc:
        train_lines = [
            f"optim    = {tc.optimizer}",
            f"lr       = {tc.learning_rate:.1e}  wd={tc.weight_decay:.2f}",
            f"batch    = {tc.batch_size}  ctx={tc.seq_len}",
            f"tok/step = {_fmt_si(tc.batch_size * tc.seq_len)}",
            f"steps    = {_fmt_si(tc.steps)}  warmup={tc.warmup_steps}",
            f"min_lr   = lr × {tc.min_lr_ratio}",
            f"grad_clip= {tc.grad_clip}  smooth={tc.label_smoothing}",
            (f"clip     = per_sample ×{tc.loss_clipping.factor:.1f}"
             if tc.loss_clipping.enabled else "clip     = off"),
            (f"quant    = int{tc.quantization.bits}" if tc.quantization.enabled
             else "quant    = off"),
        ]
    _panel_text(train_ax, "training pipeline",
                train_lines or ["(no training block)"],
                boxcolor="#e8f8f5", titlecolor="#117864")

    # [rows 3..6]: mechanisms panel (tall)
    mech_ax = fig.add_subplot(gs[3:6, 1])
    if tc:
        _draw_mechanisms_panel(mech_ax, tc)
    else:
        _panel_text(mech_ax, "trunk mechanisms",
                    ["(no training config)"])

    # [rows 6..7]: pass marks panel
    pass_ax = fig.add_subplot(gs[6:8, 1])
    pass_lines = []
    if tc:
        for r in tc.pass_marks.rules:
            bits = [f"{r.name}:"]
            bits.append(f"  metric={r.metric}")
            if r.at_step:
                bits.append(f"  @step={_fmt_si(r.at_step)}")
            if r.max is not None:
                bits.append(f"  max={r.max}")
            if r.min is not None:
                bits.append(f"  min={r.min}")
            if r.window:
                bits.append(f"  window={_fmt_si(r.window)} trend={r.trend} tol={r.tol}")
            pass_lines.extend(bits)
            pass_lines.append("")
    _panel_text(pass_ax, "pass_marks (early-exit)",
                pass_lines or ["(no pass marks)"],
                boxcolor="#fdedec", titlecolor="#922b21")

    # ── Sidebar column 2 (NT table / formal specs / sheaves) ──
    # [rows 1..4]: NT kinetics table
    nt_ax = fig.add_subplot(gs[1:5, 2])
    _draw_nt_table(nt_ax, g.nt_systems)

    # [rows 4..6]: formal specs
    fs_ax = fig.add_subplot(gs[5:7, 2])
    fs_lines = []
    for fs in g.formal_specs:
        fs_lines.append(f"formal_spec {fs['name']}")
        fs_lines.append(f"  type={fs['spec_type']}")
        for k, v in (fs.get("properties") or {}).items():
            fs_lines.append(f"  {k}={v}")
        fs_lines.append("")
    for sh in g.sheaves:
        fs_lines.append(f"sheaf {sh['name']}")
        if sh.get("contradiction_threshold") is not None:
            fs_lines.append(f"  contrad≥{sh['contradiction_threshold']}")
        if sh.get("mechanism"):
            fs_lines.append(f"  mech={sh['mechanism']}")
        fs_lines.append("")
    _panel_text(fs_ax, "formal specs · sheaves",
                fs_lines or ["(none)"],
                boxcolor="#f4ecf7", titlecolor="#6c3483")

    # [row 7]: param scope membership detail
    scope_ax = fig.add_subplot(gs[7, 2])
    scope_lines = []
    for sc in g.param_scopes:
        scope_lines.append(f"{sc['name']} ({'detached' if sc['gradient']=='detached_from_main_loss' else 'normal'}):")
        # wrap populations to ~3 per line for readability
        pops = sc.get("populations", [])
        for i in range(0, len(pops), 3):
            scope_lines.append("  " + ", ".join(pops[i:i+3]))
        scope_lines.append("")
    _panel_text(scope_ax, "param_scope membership",
                scope_lines or ["(no scopes)"],
                boxcolor="#fef9e7", titlecolor="#9a7d0a")

    fig.savefig(output_path, dpi=140, bbox_inches="tight",
                facecolor="#fafbfc")
    import matplotlib.pyplot as _plt
    _plt.close(fig)


# ── 4. Emit Python source ─────────────────────────────────────────────

def emit_python(g: NeuralFlowGraph, output_path: str) -> None:
    """Write a runnable .py that reconstructs the NFG via plain dicts.

    The emitted module includes EVERY architectural fact the renderer
    consumes — populations, synapses, modulations, NT kinetics, param
    scopes, formal specs, sheaves, and architecture-level metadata — so
    the file is a faithful round-trippable snapshot of the .neuro spec.
    """
    import json
    nodes = [vars(n) for n in g.nodes]
    edges = [vars(e) for e in g.edges]
    body = json.dumps({
        "arch": g.arch_name,
        "architecture_meta": g.architecture_meta,
        "stats": g.stats(),
        "nodes": nodes,
        "edges": edges,
        "nt_systems": g.nt_systems,
        "param_scopes": g.param_scopes,
        "formal_specs": g.formal_specs,
        "sheaves": g.sheaves,
    }, indent=2, default=str)
    src = (
        f'# -*- coding: utf-8 -*-\n'
        f'"""Neural Flow Graph for the {g.arch_name} architecture.\n\n'
        f'Auto-generated by `brian compile nfg`. Load with:\n'
        f'    from {Path(output_path).stem} import NFG\n'
        f'    # NFG[\'nodes\'], NFG[\'edges\'], NFG[\'stats\'], NFG[\'nt_systems\'], ...\n'
        f'"""\n'
        f'NFG = {body}\n'
    )
    Path(output_path).write_text(src, encoding="utf-8")
