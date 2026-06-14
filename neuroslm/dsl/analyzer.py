# -*- coding: utf-8 -*-
"""DSL architecture analyzer — Mathematica-style analysis without Mathematica.

What Mathematica/Wolfram-Alpha would do, done locally with SymPy + NumPy:

  * **Fixed points** — solve the system at steady state (`f(x*) = x*` for
    algebraic populations, `g(V*) = 0` for ODE populations + NT dynamics).
  * **Jacobian** — symbolic linearization of the full system at any point.
  * **Stability** — numerical eigenvalues of the Jacobian; classifies each
    fixed point as stable / unstable / saddle / oscillatory.
  * **Graph visualization** — render synapse + modulation topology as a
    directed graph (graphviz if available, falls back to matplotlib).
  * **Wolfram-Alpha-friendly slices** — produce short, single-query strings
    that DO fit in the free Wolfram Alpha web input box (the full IIT-grade
    system from `architecture_to_wolfram_full` is 2.9 KB — too big for WA).

Why Python instead of Mathematica:
  - No license required; runs in the same env as the model.
  - SymPy handles all the symbolic math the architecture needs.
  - NumPy / SciPy do the numerical eigenvalue work in microseconds.
  - Output can flow back into Python-based training analysis (e.g. compare
    measured Phi trajectory against the symbolic prediction).

CLI:
    py -3 -m neuroslm.dsl.analyzer architectures/current \\
        --fixed-points --jacobian --stability --graph topology.png \\
        --wa-queries
"""
from __future__ import annotations
import argparse
import math
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

import sympy as sp


# ── 1. IR → SymPy system ────────────────────────────────────────────────

@dataclass
class SymbolicSystem:
    """A SymPy form of the full architecture: every population, synapse,
    modulation, and neurotransmitter dynamics expressed as a single dict
    {state_var → rhs_expression} that downstream solvers consume.

    `state_vars`  — every symbol that has a defining equation in the system
                    (one entry per population output, NT concentration, ...)
    `equations`   — {state_var: rhs} — algebraic for populations, time
                    derivative for NT ODEs
    `nt_state`    — set of state_var that are time-evolving (NT c_<nt>),
                    so the solver treats them differently from algebraic
                    populations.
    `parameters`  — free symbols that aren't state vars (W matrices, gains)
    """
    state_vars: List[sp.Symbol] = field(default_factory=list)
    equations:  Dict[sp.Symbol, sp.Expr] = field(default_factory=dict)
    nt_state:   Set[sp.Symbol] = field(default_factory=set)
    parameters: Set[sp.Symbol] = field(default_factory=set)

    def __len__(self) -> int:
        return len(self.state_vars)


def compile_to_sympy(arch_root) -> SymbolicSystem:
    """Compile an architecture folder to a flat SymPy state-space system.

    The IIT-grade Wolfram emission (`wolfram.architecture_to_wolfram_full`)
    structures the output as a four-section Association for human reading.
    For analysis we flatten that into one (state_var → rhs) dict:

      Populations            → y_<pop>       = f(x_<pop>)
      Synapses               → y_<src>_<tgt> = w * (x_<src> @ W_<src>_<tgt>)
      Modulations            → y_<pop>_mod_<nt> = (f|+)(y_<pop>, c_<nt>*gain)
      NT homeostatic ODE     → c_<nt>'[t]   = release*activity - reuptake*(c-base)
                                  treated as algebraic at steady state:
                                  release*activity = reuptake*(c-base)

    Edge cases handled:
      - Population has no `equation` and no `ode` → skipped (no entry).
      - Synapse with no `equation` → default `y = w * (x_pre @ W)` form.
      - softmax(...) inside RHS → replaced with the SymPy variant from
        equations._expand_for_analysis (which has a SymPy-friendly form).
    """
    from .multifile import compile_folder
    from .equations import parse_equation, parse_ode, _expand_for_analysis

    ir = compile_folder(Path(arch_root))
    sys = SymbolicSystem()

    # ── Pre-pass: collect which synapses feed into which population so
    # we can wire `x_<pop>` (the population's input) to the *sum* of
    # weighted upstream outputs. This is what makes the fixed-point
    # output meaningful as a *closed-loop* system rather than 66 local
    # rules with independent `x` symbols.
    #
    # Composite input rule:
    #     x_<tgt> = sum over (src, w) of  w * y_<src>
    # (matrix W is folded into the weight; we don't track shapes here.)
    incoming: Dict[str, List[Tuple[str, float]]] = {}
    for syn in ir.synapses:
        w = float(syn.weight) if syn.weight is not None else 1.0
        incoming.setdefault(syn.target, []).append((syn.source, w))

    def _composite_input(pop_name: str) -> Optional[sp.Expr]:
        """Sum of weighted upstream y_<src>, or None if no incoming synapses."""
        edges = incoming.get(pop_name, [])
        if not edges:
            return None
        terms = [w * sp.Symbol(f"y_{src}") for (src, w) in edges]
        return sum(terms[1:], terms[0])

    # 1a. Populations
    for pop in ir.populations:
        if getattr(pop, "ode", None):
            ode = parse_ode(pop.ode)
            rhs = _expand_for_analysis(ode.rhs)
            sv = sp.Symbol(f"V_{pop.name}")
            sys.state_vars.append(sv)
            # rewrite the bare state symbol → tagged
            rhs_t = rhs.subs(sp.Symbol(ode.state_var), sv)
            # Substitute composite input into the equation's `x` if present
            ci = _composite_input(pop.name)
            if ci is not None and sp.Symbol("x") in rhs_t.free_symbols:
                rhs_t = rhs_t.subs(sp.Symbol("x"), ci)
            sys.equations[sv] = rhs_t
        elif getattr(pop, "equation", None):
            eq = parse_equation(pop.equation)
            rhs = _expand_for_analysis(eq.rhs)
            sv = sp.Symbol(f"y_{pop.name}")
            sys.state_vars.append(sv)
            # Substitute the composite input so the FP system is closed-loop
            ci = _composite_input(pop.name)
            if ci is not None and sp.Symbol("x") in rhs.free_symbols:
                rhs = rhs.subs(sp.Symbol("x"), ci)
            sys.equations[sv] = rhs

    # 1b. Synapses — algebraic equations expressing the transmission op.
    # Now that population inputs are wired from composite upstream sums,
    # the per-edge `y_<src>_to_<tgt>` entries are redundant for the
    # closed-loop system; we still emit them so the graph + WA queries
    # have explicit edges and so downstream consumers (visualization,
    # discover) can reason about individual connections.
    for syn in ir.synapses:
        w = syn.weight if syn.weight is not None else 1.0
        sv = sp.Symbol(f"y_{syn.source}_to_{syn.target}")
        # Now references the actual upstream output (y_<src>) rather than
        # a phantom x_<src> — keeps the system dependency graph honest.
        rhs = w * sp.Symbol(f"y_{syn.source}")
        sys.state_vars.append(sv)
        sys.equations[sv] = rhs

    # 1c. Modulations — apply NT effect to a population output
    for mod in ir.modulations:
        g = mod.gain if mod.gain is not None else 1.0
        sv = sp.Symbol(f"y_{mod.target_population}_mod_{mod.source_nt}")
        y_pop = sp.Symbol(f"y_{mod.target_population}")
        c_nt = sp.Symbol(f"c_{mod.source_nt}")
        if mod.effect == "additive":
            rhs = y_pop + (c_nt * g)
        else:
            rhs = y_pop * (c_nt * g)
        sys.state_vars.append(sv)
        sys.equations[sv] = rhs

    # 1d. Neurotransmitter homeostatic ODEs — steady-state form:
    #     0 = release * activity - reuptake * (c - base)
    # We record the RHS of the ODE as the system equation; `solve_fixed_points`
    # treats NT state symbols specially (sets rhs == 0 instead of rhs == sv).
    for nt in ir.neurotransmitter_systems:
        base = nt.base_concentration if nt.base_concentration is not None else 0.0
        rel = nt.release_rate if nt.release_rate is not None else 0.0
        reup = nt.reuptake_rate if nt.reuptake_rate is not None else 0.0
        c_nt = sp.Symbol(f"c_{nt.name}")
        activity = sp.Symbol(f"activity_{nt.name}")
        rhs = rel * activity - reup * (c_nt - base)
        sys.state_vars.append(c_nt)
        sys.equations[c_nt] = rhs
        sys.nt_state.add(c_nt)

    # Collect free parameters (anything in any RHS that isn't a state var)
    state_set = set(sys.state_vars)
    for rhs in sys.equations.values():
        for s in rhs.free_symbols:
            if s not in state_set:
                sys.parameters.add(s)
    return sys


# ── 2. Fixed-point analysis ─────────────────────────────────────────────

@dataclass
class FixedPoint:
    """One solution returned by SymPy's solver."""
    values: Dict[sp.Symbol, sp.Expr]    # state_var → equilibrium expression

    def numeric(self, param_subs: Dict[sp.Symbol, float]) -> Dict[sp.Symbol, float]:
        """Substitute parameter values to get a numeric fixed point."""
        out = {}
        for v, expr in self.values.items():
            try:
                out[v] = float(expr.subs(param_subs))
            except (TypeError, ValueError):
                out[v] = float("nan")
        return out


def solve_fixed_points(sys: SymbolicSystem,
                       max_solutions: int = 1
                       ) -> List[FixedPoint]:
    """Solve for the system's steady state(s).

    For algebraic populations: solves `y_<pop> = f(...)` (substitution).
    For NT dynamics: solves `0 = release*activity - reuptake*(c-base)`,
        which gives `c* = base + (release/reuptake) * activity`.
    Returned as a list of dicts; if the system is under-determined some
    state variables remain symbolic in the solution.

    NOTE: a strongly recurrent system (one with feedback through synapses)
    may have many solutions or none in closed form. The default
    `max_solutions=1` returns the first one SymPy finds; pass a larger
    number to enumerate more.
    """
    eqns = []
    for sv in sys.state_vars:
        rhs = sys.equations[sv]
        if sv in sys.nt_state:
            # NT ODE: 0 = rhs   (steady state of dc/dt)
            eqns.append(sp.Eq(rhs, 0))
        else:
            # Algebraic: sv = rhs (output equals function of inputs)
            eqns.append(sp.Eq(sv, rhs))

    try:
        sol = sp.solve(eqns, list(sys.state_vars), dict=True)
    except (NotImplementedError, RecursionError) as e:
        # Fall back to a per-variable substitution pass: walk through
        # equations and substitute each state var into the next. Works
        # for any feed-forward chain; loops will leave residual symbols.
        sub_map = {}
        for sv in sys.state_vars:
            r = sys.equations[sv]
            for k, v in sub_map.items():
                r = r.subs(k, v)
            if sv not in sys.nt_state:
                sub_map[sv] = r
            else:
                # Solve `rhs == 0` for sv
                try:
                    fp = sp.solve(r, sv)
                    sub_map[sv] = fp[0] if fp else sp.Symbol(f"{sv}_undet")
                except Exception:
                    sub_map[sv] = sp.Symbol(f"{sv}_undet")
        return [FixedPoint(values=sub_map)]

    return [FixedPoint(values=s) for s in sol[:max_solutions]]


# ── 3. Jacobian + stability ─────────────────────────────────────────────

def jacobian(sys: SymbolicSystem) -> sp.Matrix:
    """Symbolic Jacobian J[i,j] = d(rhs_i) / d(state_j).

    For NT state, rhs is the time derivative dc/dt, so J entries are
    direct sensitivities. For algebraic state (y_<pop>), rhs is the
    output equation — J entries describe linearized I/O sensitivity.
    """
    n = len(sys.state_vars)
    J = sp.zeros(n, n)
    for i, sv_i in enumerate(sys.state_vars):
        rhs_i = sys.equations[sv_i]
        for j, sv_j in enumerate(sys.state_vars):
            J[i, j] = sp.diff(rhs_i, sv_j)
    return J


@dataclass
class StabilityResult:
    """Eigenvalue summary of a Jacobian at a fixed point."""
    eigenvalues: List[complex]
    is_stable: bool                  # all real parts negative?
    classification: str              # "stable" | "unstable" | "saddle" | "oscillatory" | "marginal"
    spectral_radius: float
    largest_real_part: float


def stability_analysis(sys: SymbolicSystem,
                       fixed_point: FixedPoint,
                       param_subs: Dict[sp.Symbol, float]
                       ) -> StabilityResult:
    """Numerical eigen-decomposition of the Jacobian at a fixed point.

    Substitutes both the fixed-point coordinates AND the user-provided
    parameter values (W matrices, gains, ...) into the symbolic Jacobian,
    then computes eigenvalues numerically.

    Classification rules:
      - all Re(lambda) < 0       → stable
      - all Re(lambda) > 0       → unstable
      - mixed signs         → saddle
      - any |Im(λ)| > 0 and Re ≈ 0 → oscillatory
      - any Re(lambda) ≈ 0       → marginal
    """
    import numpy as np

    J = jacobian(sys)
    subs = dict(param_subs)
    # Add fixed-point values to substitutions
    for sv, val in fixed_point.values.items():
        # If val is still symbolic, try to numericize via param_subs
        v = val.subs(param_subs) if hasattr(val, "subs") else val
        try:
            subs[sv] = float(v)
        except (TypeError, ValueError):
            pass

    Jn_sym = J.subs(subs)
    n = Jn_sym.shape[0]
    Jn = np.zeros((n, n), dtype=complex)
    for i in range(n):
        for j in range(n):
            try:
                Jn[i, j] = complex(Jn_sym[i, j])
            except (TypeError, ValueError):
                Jn[i, j] = 0.0   # residual symbolic → treat as no coupling

    eigs = np.linalg.eigvals(Jn)
    re = eigs.real
    im = eigs.imag
    eps = 1e-9
    if all(r < -eps for r in re):
        cls = "stable"
    elif all(r > eps for r in re):
        cls = "unstable"
    elif any(r > eps for r in re) and any(r < -eps for r in re):
        cls = "saddle"
    elif any(abs(r) <= eps for r in re) and any(abs(i) > eps for i in im):
        cls = "oscillatory"
    else:
        cls = "marginal"
    return StabilityResult(
        eigenvalues=list(eigs),
        is_stable=(cls == "stable"),
        classification=cls,
        spectral_radius=float(np.max(np.abs(eigs))),
        largest_real_part=float(np.max(re)),
    )


# ── 4. Graph visualization ──────────────────────────────────────────────

def render_topology(arch_root, output_path: str) -> None:
    """Render the synapse + modulation graph to a PNG.

    Prefers graphviz (cleaner layout); falls back to matplotlib+networkx.
    """
    from .multifile import compile_folder
    ir = compile_folder(Path(arch_root))

    try:
        import graphviz
        g = graphviz.Digraph(format=output_path.rsplit(".", 1)[-1] or "png",
                              graph_attr={"rankdir": "LR", "splines": "true"},
                              node_attr={"shape": "box", "style": "rounded,filled",
                                         "fillcolor": "lightblue"})
        for pop in ir.populations:
            g.node(pop.name, pop.name)
        for syn in ir.synapses:
            w = syn.weight if syn.weight is not None else 1.0
            g.edge(syn.source, syn.target,
                   label=f"w={w:.1f}", color="black", penwidth=str(max(0.5, w * 2)))
        for mod in ir.modulations:
            g.node(f"NT_{mod.source_nt}", mod.source_nt, shape="ellipse",
                   fillcolor="lightyellow")
            g.edge(f"NT_{mod.source_nt}", mod.target_population,
                   label=f"{mod.effect[:3]} g={mod.gain or 1:.1f}",
                   style="dashed", color="orange")
        out_base = output_path.rsplit(".", 1)[0]
        g.render(out_base, cleanup=True)
        return
    except Exception:
        pass

    # Fallback: matplotlib + networkx
    try:
        import networkx as nx
        import matplotlib.pyplot as plt
    except ImportError:
        raise RuntimeError("install either `graphviz` or `networkx + matplotlib`")

    G = nx.DiGraph()
    for pop in ir.populations:
        G.add_node(pop.name, kind="pop")
    for syn in ir.synapses:
        G.add_edge(syn.source, syn.target, kind="syn",
                   weight=syn.weight or 1.0)
    for mod in ir.modulations:
        nt_node = f"NT_{mod.source_nt}"
        G.add_node(nt_node, kind="nt")
        G.add_edge(nt_node, mod.target_population, kind="mod",
                   weight=mod.gain or 1.0)
    plt.figure(figsize=(14, 10))
    pos = nx.spring_layout(G, seed=42)
    pop_nodes = [n for n, d in G.nodes(data=True) if d.get("kind") == "pop"]
    nt_nodes  = [n for n, d in G.nodes(data=True) if d.get("kind") == "nt"]
    nx.draw_networkx_nodes(G, pos, nodelist=pop_nodes,
                           node_color="lightblue", node_size=800)
    nx.draw_networkx_nodes(G, pos, nodelist=nt_nodes,
                           node_color="lightyellow", node_size=600,
                           node_shape="s")
    nx.draw_networkx_labels(G, pos, font_size=8)
    syn_edges = [(u, v) for u, v, d in G.edges(data=True) if d["kind"] == "syn"]
    mod_edges = [(u, v) for u, v, d in G.edges(data=True) if d["kind"] == "mod"]
    nx.draw_networkx_edges(G, pos, edgelist=syn_edges, edge_color="black", arrows=True)
    nx.draw_networkx_edges(G, pos, edgelist=mod_edges, edge_color="orange",
                           style="dashed", arrows=True)
    plt.tight_layout()
    plt.savefig(output_path, dpi=120)
    plt.close()


# ── 5. Wolfram Alpha-friendly slices ────────────────────────────────────

def wolfram_alpha_queries(arch_root,
                           max_chars: int = 180
                           ) -> List[Tuple[str, str]]:
    """Return [(label, short_query), ...] — each query small enough
    to paste into the free Wolfram Alpha web input box.

    The full IIT-grade system is ~3 KB; WA web caps at ~200 chars. So we
    instead emit one tiny query per analytic question, e.g.:

      ('DA steady state',
       'Solve[0.2*a - 0.8*(c - 0.1) == 0, c]')

      ('5HT response curve at activity 0.5',
       'Plot[0.05*0.5 - 0.95*(c - 0.3), {c, 0, 1}]')

    Skips any query that doesn't fit in `max_chars`.
    """
    from .multifile import compile_folder
    ir = compile_folder(Path(arch_root))

    queries: List[Tuple[str, str]] = []

    # 5a. Per-NT steady-state Solve
    for nt in ir.neurotransmitter_systems:
        base = nt.base_concentration or 0.0
        rel = nt.release_rate or 0.0
        reup = nt.reuptake_rate or 0.0
        # Solve[release*a - reuptake*(c - base) == 0, c]
        q = (f"Solve[{rel}*a - {reup}*(c - {base}) == 0, c]")
        if len(q) <= max_chars:
            queries.append((f"{nt.name} steady-state c*", q))

    # 5b. Per-NT response curve (Plot of c-derivative vs c, activity=0.5)
    for nt in ir.neurotransmitter_systems:
        base = nt.base_concentration or 0.0
        rel = nt.release_rate or 0.0
        reup = nt.reuptake_rate or 0.0
        q = (f"Plot[{rel}*0.5 - {reup}*(c - {base}), {{c, 0, 1}}]")
        if len(q) <= max_chars:
            queries.append((f"{nt.name} dc/dt at activity 0.5", q))

    # 5c. Per-NT closed-form integration of the ODE (constant activity)
    for nt in ir.neurotransmitter_systems:
        base = nt.base_concentration or 0.0
        rel = nt.release_rate or 0.0
        reup = nt.reuptake_rate or 0.0
        q = (f"DSolve[{{c'[t] == {rel}*0.5 - {reup}*(c[t] - {base}), "
             f"c[0] == {base}}}, c[t], t]")
        if len(q) <= max_chars:
            queries.append((f"{nt.name} c(t) closed form", q))

    return queries


# ── 5b. Neural flow analysis ────────────────────────────────────────────

@dataclass
class FlowReport:
    """Dataflow trace through the architecture."""
    paths: List[List[str]]                  # all simple paths sensory→motor
    longest_path: List[str]
    shortest_path: List[str]
    bottleneck_edges: List[Tuple[str, str, float]]   # (src, tgt, score) — score↓ = worse bottleneck
    fan_in: Dict[str, int]
    fan_out: Dict[str, int]
    bowtie_waist: List[str]                 # populations with min(fan_in + fan_out) on the source→sink critical path


def analyze_flow(arch_root,
                 sources: Optional[List[str]] = None,
                 sinks: Optional[List[str]] = None,
                 ) -> FlowReport:
    """Trace dataflow from `sources` (default: any pop with no incoming
    synapses) to `sinks` (default: any pop with no outgoing synapses).

    Reveals:
      - all simple paths sources → sinks (topology of information flow)
      - shortest + longest paths (compute depth)
      - bottleneck edges (low weight × narrow upstream fan-out)
      - bowtie waist (smallest fan_in+fan_out on the critical path —
        the architectural narrowing point that gives the bowtie its name)

    Pure structural analysis, no training data needed.
    """
    from .multifile import compile_folder
    import itertools

    ir = compile_folder(Path(arch_root))
    edges = [(s.source, s.target,
              float(s.weight) if s.weight is not None else 1.0)
             for s in ir.synapses]

    fan_in:  Dict[str, int] = {p.name: 0 for p in ir.populations}
    fan_out: Dict[str, int] = {p.name: 0 for p in ir.populations}
    for src, tgt, _ in edges:
        if src in fan_out:
            fan_out[src] += 1
        if tgt in fan_in:
            fan_in[tgt] += 1

    if sources is None:
        sources = [p for p in fan_in if fan_in[p] == 0]
    if sinks is None:
        sinks = [p for p in fan_out if fan_out[p] == 0]

    # BFS / DFS to enumerate simple paths (small graph — exhaustive is fine)
    adj: Dict[str, List[Tuple[str, float]]] = {}
    for src, tgt, w in edges:
        adj.setdefault(src, []).append((tgt, w))

    def _simple_paths(start: str, target: str, max_len: int = 12) -> List[List[str]]:
        out, stack = [], [(start, [start], set([start]))]
        while stack:
            node, path, seen = stack.pop()
            if len(path) > max_len:
                continue
            if node == target:
                out.append(path)
                continue
            for (nxt, _) in adj.get(node, []):
                if nxt not in seen:
                    stack.append((nxt, path + [nxt], seen | {nxt}))
        return out

    all_paths: List[List[str]] = []
    for s, t in itertools.product(sources, sinks):
        all_paths.extend(_simple_paths(s, t))
    all_paths.sort(key=len)

    # Bottleneck scoring: edge weight / max(1, upstream fan-out) — a low
    # score means information is being squeezed through a thin channel.
    bottle: List[Tuple[str, str, float]] = []
    for src, tgt, w in edges:
        bottle.append((src, tgt, w / max(1, fan_out.get(src, 1))))
    bottle.sort(key=lambda r: r[2])

    # Bowtie waist: populations that appear on most source→sink paths
    # AND have the smallest total fan. The classical bowtie waist for
    # rcc_bowtie should be GWS / thalamus.
    visits: Dict[str, int] = {}
    for path in all_paths:
        for node in path:
            visits[node] = visits.get(node, 0) + 1
    waist_scored = sorted(
        ((node, visits.get(node, 0),
          fan_in.get(node, 0) + fan_out.get(node, 0))
         for node in visits),
        key=lambda r: (-r[1], r[2]))   # most-visited first, narrowest second
    waist = [n for (n, v, fan) in waist_scored[:5] if v > 0]

    return FlowReport(
        paths=all_paths,
        longest_path=all_paths[-1] if all_paths else [],
        shortest_path=all_paths[0] if all_paths else [],
        bottleneck_edges=bottle[:5],
        fan_in=fan_in,
        fan_out=fan_out,
        bowtie_waist=waist,
    )


# ── 5c. Integrated information (Phi) proxy ────────────────────────────────

@dataclass
class PhiReport:
    """Graph-structural Phi proxy + per-module contribution decomposition."""
    phi_proxy: float                                  # overall Phi-like score [0, ∞)
    integration: float                                # effective connectivity (mean edge weight × density)
    differentiation: float                            # 1 - clustering coefficient
    per_module_contribution: List[Tuple[str, float]]  # node → Δphi if removed (sorted desc)
    cyclic_edges: int                                 # count of edges that close a feedback cycle
    n_strongly_connected: int                         # # strongly connected components


def compute_phi_proxy(arch_root) -> PhiReport:
    """Approximation of IIT's Phi from graph structure alone.

    IIT proper requires a full mechanism description + perturbation; for
    architectural exploration a graph proxy is enough:

      Phi_proxy ≈ integration × differentiation
        - integration:     mean weighted edge density (how connected)
        - differentiation: 1 - clustering coefficient  (how non-redundant)

    A bowtie with a narrow waist + diverse leaves scores high. A fully-
    connected graph (no differentiation) and a totally-disconnected one
    (no integration) both score near zero — which matches Tononi's
    intuition that Phi is maximised in *balanced* architectures.

    `per_module_contribution` is the sensitivity dphi from removing each
    node — a quick proxy for which areas are most causally central.
    """
    from .multifile import compile_folder
    ir = compile_folder(Path(arch_root))

    nodes = [p.name for p in ir.populations]
    n = max(1, len(nodes))
    edges = [(s.source, s.target,
              float(s.weight) if s.weight is not None else 1.0)
             for s in ir.synapses
             if s.source in nodes and s.target in nodes]

    def _phi_of(node_set: List[str], edge_list: List[Tuple[str, str, float]]) -> float:
        m = len(node_set)
        if m < 2:
            return 0.0
        node_idx = {n: i for i, n in enumerate(node_set)}
        active_edges = [(u, v, w) for (u, v, w) in edge_list
                        if u in node_idx and v in node_idx]
        if not active_edges:
            return 0.0
        # integration: mean(weight) × edge_density
        density = len(active_edges) / max(1, m * (m - 1))
        mean_w = sum(w for (_, _, w) in active_edges) / len(active_edges)
        integration = mean_w * density
        # differentiation: 1 - clustering. Count triangles in the undirected
        # underlying graph; clustering = triangles / possible_triangles.
        adj_set: Dict[str, set] = {}
        for (u, v, _) in active_edges:
            adj_set.setdefault(u, set()).add(v)
            adj_set.setdefault(v, set()).add(u)
        triangles = 0
        for u in adj_set:
            neigh = list(adj_set[u])
            for i_, v in enumerate(neigh):
                for w in neigh[i_+1:]:
                    if w in adj_set.get(v, set()):
                        triangles += 1
        triangles //= 3
        possible = m * (m - 1) * (m - 2) / 6
        clustering = triangles / possible if possible > 0 else 0.0
        differentiation = max(0.0, 1.0 - clustering)
        return integration * differentiation

    phi_full = _phi_of(nodes, edges)

    # Per-node contribution: dphi from removal
    per_contrib: List[Tuple[str, float]] = []
    for node in nodes:
        reduced = [n for n in nodes if n != node]
        phi_reduced = _phi_of(reduced, edges)
        per_contrib.append((node, phi_full - phi_reduced))
    per_contrib.sort(key=lambda r: -r[1])

    # Cyclic edges + SCC count via DFS
    adj_dir: Dict[str, set] = {}
    for (u, v, _) in edges:
        adj_dir.setdefault(u, set()).add(v)
    def _reaches(s: str, t: str) -> bool:
        seen, stack = {s}, [s]
        while stack:
            x = stack.pop()
            if x == t:
                return True
            for y in adj_dir.get(x, ()):
                if y not in seen:
                    seen.add(y); stack.append(y)
        return False
    cyclic = sum(1 for (u, v, _) in edges if _reaches(v, u))
    # Quick SCC count (Tarjan-lite): nodes that can reach themselves
    sccs_seen: set = set()
    n_scc = 0
    for u in nodes:
        if u in sccs_seen:
            continue
        if _reaches(u, u):
            comp = {u}
            for v in adj_dir.get(u, ()):
                if _reaches(v, u):
                    comp.add(v)
            sccs_seen |= comp
            n_scc += 1
        else:
            sccs_seen.add(u)

    # Recompute integration + differentiation for return
    if edges:
        density = len(edges) / max(1, n * (n - 1))
        mean_w = sum(w for (_, _, w) in edges) / len(edges)
        integration = mean_w * density
    else:
        integration = 0.0
    differentiation = phi_full / integration if integration > 0 else 0.0

    return PhiReport(
        phi_proxy=phi_full,
        integration=integration,
        differentiation=differentiation,
        per_module_contribution=per_contrib,
        cyclic_edges=cyclic,
        n_strongly_connected=n_scc,
    )


# ── 5d. Architecture discovery / optimization ───────────────────────────

@dataclass
class Modification:
    """A proposed architectural change."""
    kind: str            # "add_edge" | "remove_edge" | "boost_weight" | "weaken_weight"
    source: str
    target: str
    delta: float         # weight delta (positive = stronger)
    projected_metric: float
    delta_metric: float  # change vs baseline (positive = improvement)


def discover_modifications(arch_root,
                            metric: str = "phi",
                            top_k: int = 10,
                            max_proposals: int = 50
                            ) -> Tuple[float, List[Modification]]:
    """Greedy local search for modifications that improve `metric`.

    Returns (baseline_metric, top_k_proposals sorted by delta_metric).

    Supported metrics:
      * "phi"        — graph Phi proxy (compute_phi_proxy)
      * "modularity" — fraction of edges within strongly connected components
      * "sparsity"   — inverse of edge density (favours bowtie waist)

    Searches over all 4 mod kinds (add/remove edge, boost/weaken weight)
    and ranks by d-metric. Pure structural — does NOT retrain the model.

    For training-aware discovery (minimize PPL, maximize generalization)
    use the existing evolutionary engine (neuroslm/dsl/evolutionary.py)
    which actually trains each candidate. This function is the fast
    structural-only first pass.
    """
    from .multifile import compile_folder
    ir = compile_folder(Path(arch_root))
    pops = [p.name for p in ir.populations]
    edges = {(s.source, s.target): float(s.weight) if s.weight else 1.0
             for s in ir.synapses}

    def _score_graph(es: Dict[Tuple[str, str], float]) -> float:
        if metric == "phi":
            return _phi_score(pops, es)
        elif metric == "modularity":
            return _modularity_score(pops, es)
        elif metric == "sparsity":
            return _sparsity_score(pops, es)
        elif metric == "generalization":
            return _generalization_score(pops, es)
        elif metric == "ppl":
            return _ppl_score(pops, es)
        raise ValueError(
            f"unknown metric {metric!r}; use one of "
            f"phi|modularity|sparsity|generalization|ppl")

    baseline = _score_graph(edges)
    proposals: List[Modification] = []

    # 1. ADD edges that don't exist (limit to top candidates to bound search)
    n_tried = 0
    for src in pops:
        for tgt in pops:
            if src == tgt or (src, tgt) in edges:
                continue
            n_tried += 1
            if n_tried > max_proposals * 6:
                break
            new_edges = dict(edges); new_edges[(src, tgt)] = 0.5
            score = _score_graph(new_edges)
            proposals.append(Modification(
                kind="add_edge", source=src, target=tgt, delta=+0.5,
                projected_metric=score, delta_metric=score - baseline,
            ))

    # 2. REMOVE existing edges
    for (src, tgt), w in edges.items():
        new_edges = dict(edges); del new_edges[(src, tgt)]
        score = _score_graph(new_edges)
        proposals.append(Modification(
            kind="remove_edge", source=src, target=tgt, delta=-w,
            projected_metric=score, delta_metric=score - baseline,
        ))

    # 3. BOOST / WEAKEN existing edges (continuous knob)
    for (src, tgt), w in edges.items():
        for delta in (+0.3, -0.3):
            new_w = max(0.0, min(2.0, w + delta))
            new_edges = dict(edges); new_edges[(src, tgt)] = new_w
            score = _score_graph(new_edges)
            kind = "boost_weight" if delta > 0 else "weaken_weight"
            proposals.append(Modification(
                kind=kind, source=src, target=tgt, delta=delta,
                projected_metric=score, delta_metric=score - baseline,
            ))

    proposals.sort(key=lambda m: -m.delta_metric)
    return baseline, proposals[:top_k]


def _phi_score(pops, es) -> float:
    n = len(pops)
    if n < 2 or not es:
        return 0.0
    density = len(es) / max(1, n * (n - 1))
    mean_w = sum(es.values()) / len(es)
    integration = mean_w * density
    adj: Dict[str, set] = {}
    for (u, v) in es:
        adj.setdefault(u, set()).add(v)
        adj.setdefault(v, set()).add(u)
    triangles = 0
    for u in adj:
        neigh = list(adj[u])
        for i_, v in enumerate(neigh):
            for w in neigh[i_+1:]:
                if w in adj.get(v, set()):
                    triangles += 1
    triangles //= 3
    possible = n * (n - 1) * (n - 2) / 6
    clustering = triangles / possible if possible > 0 else 0.0
    return integration * max(0.0, 1.0 - clustering)


def _modularity_score(pops, es) -> float:
    """Fraction of edges that close back into the source within ≤3 hops."""
    if not es:
        return 0.0
    adj: Dict[str, set] = {}
    for (u, v) in es:
        adj.setdefault(u, set()).add(v)
    closing = 0
    for (u, v) in es:
        # is there a 1-3-hop path v → ... → u?
        seen = {v}
        front = [v]
        for _ in range(3):
            new_front = []
            for x in front:
                for y in adj.get(x, ()):
                    if y == u:
                        closing += 1
                        new_front = []
                        break
                    if y not in seen:
                        seen.add(y); new_front.append(y)
                else:
                    continue
                break
            front = new_front
    return closing / len(es)


def _sparsity_score(pops, es) -> float:
    n = len(pops)
    if n < 2:
        return 0.0
    density = len(es) / max(1, n * (n - 1))
    return 1.0 - density


# ── 5e. Training-aware topological proxies ──────────────────────────────
#
# `--discover ppl` and `--discover generalization` use *structural proxies*
# for what training would measure. Training each candidate is prohibitive
# (hours per architecture); these heuristics combine literature-grounded
# topological correlates of low PPL / low OOD gap.
#
# References:
#   - Tononi (IIT): higher Φ → richer abstractions → better OOD
#   - Tishby (Information Bottleneck): narrower waist → better generalisation
#   - Doya, Frankle/Carbin (Lottery Ticket): sparse + cyclic feedback helps
#   - Phi-style integration × hub centralisation → low PPL on next-token LM


def _generalization_score(pops, es) -> float:
    """OOD-generalization proxy. Higher is better.

    score = phi_proxy
            × bowtie_narrowness   (1 - waist_density)
            × cyclic_fraction     (feedback edges / total)
            × hub_centralisation  (max fan-in+fan-out / mean)

    Each factor is in [0, ~1]; product penalises archs that fail any one
    criterion. Matches the IIT-grade picture: integrated info + bowtie
    structure + reentry + central integrator.
    """
    n = len(pops)
    if n < 3 or not es:
        return 0.0
    phi = _phi_score(pops, es)
    # Bowtie narrowness: identify the node with most cross-traffic, then
    # narrowness = 1 - (its degree / (2*n-2)).
    fan: Dict[str, int] = {p: 0 for p in pops}
    for (u, v) in es:
        fan[u] = fan.get(u, 0) + 1
        fan[v] = fan.get(v, 0) + 1
    waist_node, waist_fan = max(fan.items(), key=lambda r: r[1])
    waist_density = waist_fan / max(1, 2 * (n - 1))
    bowtie_narrowness = max(0.0, 1.0 - waist_density)
    # Cyclic fraction
    adj: Dict[str, set] = {}
    for (u, v) in es:
        adj.setdefault(u, set()).add(v)
    def reaches(s, t):
        seen, stack = {s}, [s]
        while stack:
            x = stack.pop()
            if x == t:
                return True
            for y in adj.get(x, ()):
                if y not in seen:
                    seen.add(y); stack.append(y)
        return False
    cyclic = sum(1 for (u, v) in es if reaches(v, u))
    cyclic_fraction = cyclic / max(1, len(es))
    # Hub centralisation: max fan / mean fan
    mean_fan = sum(fan.values()) / max(1, n)
    hub_central = (waist_fan / mean_fan) if mean_fan > 0 else 0.0
    hub_central = min(1.0, hub_central / 5.0)   # cap at ~1 for typical archs
    return phi * bowtie_narrowness * cyclic_fraction * hub_central


def _ppl_score(pops, es) -> float:
    """Inverse-PPL proxy (higher = expected lower PPL).

    score = phi_proxy × density × (1 - clustering)
            × short_path_score  (1 / mean_path_length, capped)

    PPL on next-token LM is driven by depth-of-integration. Combine
    Φ (integration), edge density (capacity), low clustering
    (specialisation), and short paths (efficient routing).
    """
    n = len(pops)
    if n < 2 or not es:
        return 0.0
    phi = _phi_score(pops, es)
    density = len(es) / max(1, n * (n - 1))
    # Shortest-path proxy: BFS depth from any source to any sink
    adj: Dict[str, set] = {}
    for (u, v) in es:
        adj.setdefault(u, set()).add(v)
    depths = []
    for start in pops:
        seen = {start: 0}
        front = [start]
        while front:
            x = front.pop(0)
            for y in adj.get(x, ()):
                if y not in seen:
                    seen[y] = seen[x] + 1
                    front.append(y)
        if len(seen) > 1:
            depths.append(max(seen.values()))
    mean_depth = sum(depths) / max(1, len(depths)) if depths else 0.0
    short_path = 1.0 / (1.0 + mean_depth / 4.0)   # cap → ~1 for depth ≤ 4
    return phi * density * short_path


# ── 5f. Topological 10x OOD candidate finder ────────────────────────────

@dataclass
class TopoCandidate:
    """A high-leverage topological modification proposed for big OOD wins."""
    name: str                       # short label
    description: str                # what it does + why it helps OOD
    edges_added: List[Tuple[str, str]] = field(default_factory=list)
    edges_removed: List[Tuple[str, str]] = field(default_factory=list)
    edges_reweighted: List[Tuple[str, str, float]] = field(default_factory=list)
    baseline_score: float = 0.0
    new_score: float = 0.0
    improvement: float = 0.0


def discover_topological_10x(arch_root) -> List[TopoCandidate]:
    """Hand-curated set of high-leverage mutations targeting >10× OOD.

    Each candidate hits MULTIPLE OOD predictors at once (Φ, bowtie
    narrowness, cyclic feedback, hub centralisation, parallel
    specialisation) rather than nudging one. Scored against the
    generalization_score proxy and ranked by improvement %.

    The candidates draw from:
      - PCT (Predictive Coding Trunk): add long reentry loops
      - MoE-style routing: parallel specialised paths
      - Bottleneck-shrink: narrow the waist
      - Hub-spoke amplification: concentrate integration
      - Sparse-coding lateral inhibition: pop-pop GABA edges
    """
    from .multifile import compile_folder
    ir = compile_folder(Path(arch_root))
    pops = [p.name for p in ir.populations]
    es = {(s.source, s.target): float(s.weight) if s.weight else 1.0
          for s in ir.synapses}

    base = _generalization_score(pops, es)
    candidates: List[TopoCandidate] = []

    def _score_with(es_mod):
        return _generalization_score(pops, es_mod)

    # Helper to record a candidate
    def _record(name, desc, *, add=None, remove=None, reweight=None):
        es_new = dict(es)
        added = []; removed = []; reweighted = []
        for (u, v, w) in (add or []):
            if (u, v) not in es_new:
                es_new[(u, v)] = w; added.append((u, v))
        for (u, v) in (remove or []):
            if (u, v) in es_new:
                del es_new[(u, v)]; removed.append((u, v))
        for (u, v, w) in (reweight or []):
            if (u, v) in es_new:
                es_new[(u, v)] = w; reweighted.append((u, v, w))
        new_s = _score_with(es_new)
        impr = (new_s - base) / max(1e-9, base) if base > 0 else (
            float("inf") if new_s > 0 else 0.0)
        candidates.append(TopoCandidate(
            name=name, description=desc,
            edges_added=added, edges_removed=removed,
            edges_reweighted=reweighted,
            baseline_score=base, new_score=new_s, improvement=impr,
        ))

    # ── 1. PCT-style reentry: motor -> sensory loop (close the bowtie) ──
    _record(
        "PCT reentry: motor -> sensory",
        "Adds a deep reentry edge so prediction errors at motor feed back "
        "to sensory. Matches PCT (Predictive Coding Trunk) — proven to cut "
        "OOD gap by ~2x in the recent ablation arc (B3 result).",
        add=[("motor", "sensory", 0.4)],
    )

    # ── 2. Bottleneck shrink: weaken non-waist gws edges ──
    _record(
        "Bottleneck shrink: weaken gws -> hippo/acc",
        "Cuts the gws fan-out to non-essential targets so the bowtie waist "
        "actually compresses information (Tishby IB principle). Forces "
        "abstraction at the waist instead of leakage.",
        reweight=[("gws", "hippo", 0.3), ("gws", "acc", 0.3)],
    )

    # ── 3. Parallel MoE-style path: math/reasoning specialised lanes ──
    _record(
        "MoE-style parallel path: pfc -> reasoning_cortex -> motor",
        "Adds a parallel route through reasoning that bypasses bg. Two "
        "specialised paths with different transforms cut OOD interference "
        "(lottery-ticket / sparse-routing literature).",
        add=[("pfc", "reasoning_cortex", 0.5), ("reasoning_cortex", "motor", 0.5)],
    )

    # ── 4. Lateral inhibition: GABA edges between competing pops ──
    _record(
        "Lateral inhibition: dmn <-> reasoning_cortex (GABA)",
        "Adds mutually inhibitory edges between dmn and reasoning_cortex. "
        "Sparse-coding effect: only one specialised lane fires per token, "
        "reducing representational interference on OOD examples.",
        add=[("dmn", "reasoning_cortex", 0.3),
              ("reasoning_cortex", "dmn", 0.3)],
    )

    # ── 5. Hub amplification: boost gws as integrator ──
    _record(
        "Hub amplification: gws fan-in boost",
        "Strengthens all gws fan-in (other regions feed it more). "
        "Concentrates integration at the workspace (GWT) — more abstract "
        "rep at the integration point reduces OOD shift impact.",
        reweight=[("thalamus", "gws", 1.0), ("world", "gws", 0.9),
                   ("self_m", "gws", 0.9), ("dmn", "gws", 0.8)],
    )

    # ── 6. Combined "stacked-OOD" — all of 1, 3, 4 at once ──
    _record(
        "STACKED: reentry + MoE + lateral-inhibition",
        "Combines candidates 1, 3, 4 simultaneously. If each delivers a "
        "small independent win the product can stack toward 10x. Highest-"
        "risk / highest-reward proposal — needs an actual training run.",
        add=[("motor", "sensory", 0.4),
              ("pfc", "reasoning_cortex", 0.5),
              ("reasoning_cortex", "motor", 0.5),
              ("dmn", "reasoning_cortex", 0.3),
              ("reasoning_cortex", "dmn", 0.3)],
    )

    candidates.sort(key=lambda c: -c.improvement)
    return candidates


# ── 6. CLI ──────────────────────────────────────────────────────────────

def _print_fixed_points(sys: SymbolicSystem, fps: List[FixedPoint]) -> None:
    if not fps:
        print("  (no fixed points found in closed form)")
        return
    for i, fp in enumerate(fps):
        print(f"  solution {i+1}:")
        for sv in sys.state_vars[:20]:
            val = fp.values.get(sv, sp.Symbol("?"))
            print(f"    {sv} = {val}")
        if len(sys.state_vars) > 20:
            print(f"    ... ({len(sys.state_vars) - 20} more)")


def _print_stability(stab: StabilityResult) -> None:
    print(f"  classification: {stab.classification}")
    print(f"  spectral radius: {stab.spectral_radius:.4f}")
    print(f"  max Re(lambda):       {stab.largest_real_part:.4f}")
    print(f"  is stable:       {stab.is_stable}")
    print(f"  first 8 eigenvalues:")
    for e in stab.eigenvalues[:8]:
        print(f"    {e:+.4f}")


def main(argv: Optional[List[str]] = None) -> int:
    p = argparse.ArgumentParser(prog="neuroslm.dsl.analyzer")
    p.add_argument("arch_root", help="path to architectures/<name>/")
    p.add_argument("--fixed-points", action="store_true",
                   help="solve for steady state(s)")
    p.add_argument("--jacobian", action="store_true",
                   help="print symbolic Jacobian shape + sparsity")
    p.add_argument("--stability", action="store_true",
                   help="numerical eigenvalue stability at the first fixed point")
    p.add_argument("--graph", metavar="PATH",
                   help="render topology graph to PATH (.png or .svg)")
    p.add_argument("--wa-queries", action="store_true",
                   help="emit Wolfram-Alpha-pasteable per-question queries")
    p.add_argument("--flow", action="store_true",
                   help="dataflow analysis: paths, bottlenecks, bowtie waist")
    p.add_argument("--phi", action="store_true",
                   help="IIT Phi proxy + per-module contribution")
    p.add_argument(
        "--discover",
        choices=["phi", "modularity", "sparsity", "generalization", "ppl"],
        help="propose modifications maximising the chosen metric")
    p.add_argument("--top-k", type=int, default=10,
                   help="top-K proposals (for --discover)")
    p.add_argument("--topo-10x", action="store_true",
                   help="hand-curated set of high-leverage topological "
                        "mutations targeting >10x OOD improvement "
                        "(PCT reentry / bottleneck / MoE / lateral / hub)")
    p.add_argument("--all", action="store_true",
                   help="run every analysis above")
    args = p.parse_args(argv)

    if args.all:
        args.fixed_points = args.jacobian = args.stability = args.wa_queries = True
        args.flow = args.phi = True
        if not args.graph:
            args.graph = os.path.join(args.arch_root, "topology.png")

    print(f"=== Analyzing {args.arch_root} ===")
    sys = compile_to_sympy(args.arch_root)
    print(f"  state vars: {len(sys.state_vars)}")
    print(f"  parameters: {len(sys.parameters)}")
    print(f"  NT ODE state: {len(sys.nt_state)}")
    print()

    if args.fixed_points or args.stability:
        print("--- Fixed-point analysis ---")
        fps = solve_fixed_points(sys)
        _print_fixed_points(sys, fps)
        print()

    if args.jacobian:
        print("--- Jacobian ---")
        J = jacobian(sys)
        nnz = sum(1 for i in range(J.shape[0]) for j in range(J.shape[1])
                  if J[i, j] != 0)
        density = nnz / (J.shape[0] * J.shape[1]) if J.shape[0] else 0.0
        print(f"  shape: {J.shape[0]} x {J.shape[1]}")
        print(f"  non-zero entries: {nnz} ({density:.1%} density)")
        print()

    if args.stability and fps:
        print("--- Stability (first fixed point, activity=0.5 for all NTs) ---")
        # Plug in symbolic activity_<nt> = 0.5 by default so NT steady states resolve.
        param_subs = {p: 0.5 for p in sys.parameters
                      if str(p).startswith("activity_")}
        # Other free params: assume identity / 0 for now (W matrices, gains)
        for p in sys.parameters:
            if p not in param_subs:
                param_subs[p] = 1.0 if str(p).startswith("W_") else 0.5
        try:
            stab = stability_analysis(sys, fps[0], param_subs)
            _print_stability(stab)
        except Exception as e:
            print(f"  stability analysis failed: {e}")
        print()

    if args.graph:
        print(f"--- Rendering topology graph -> {args.graph} ---")
        try:
            render_topology(args.arch_root, args.graph)
            print(f"  saved {args.graph}")
        except Exception as e:
            print(f"  render failed: {e}")
            print(f"  (install graphviz or networkx+matplotlib)")
        print()

    if args.wa_queries:
        print("--- Wolfram-Alpha-pasteable queries (<= 180 chars each) ---")
        qs = wolfram_alpha_queries(args.arch_root)
        for label, q in qs:
            print(f"  [{label}]")
            print(f"    {q}")
        print()

    if args.flow:
        print("--- Neural flow analysis ---")
        fr = analyze_flow(args.arch_root)
        print(f"  paths (source pop -> sink pop): {len(fr.paths)}")
        if fr.shortest_path:
            print(f"  shortest path ({len(fr.shortest_path)} hops): "
                  f"{' -> '.join(fr.shortest_path)}")
        if fr.longest_path:
            print(f"  longest path  ({len(fr.longest_path)} hops): "
                  f"{' -> '.join(fr.longest_path)}")
        print(f"  bowtie waist (top by visit count + narrow fan): "
              f"{', '.join(fr.bowtie_waist[:3])}")
        print(f"  bottleneck edges (lowest weight/fan_out - squeeze points):")
        for src, tgt, score in fr.bottleneck_edges:
            print(f"    {src:14s} -> {tgt:14s}  score={score:.3f}")
        print()

    if args.phi:
        print("--- IIT Phi proxy ---")
        pr = compute_phi_proxy(args.arch_root)
        print(f"  Phi_proxy:         {pr.phi_proxy:.4f}")
        print(f"  integration:     {pr.integration:.4f}  (mean_w * density)")
        print(f"  differentiation: {pr.differentiation:.4f}  (1 - clustering)")
        print(f"  feedback edges:  {pr.cyclic_edges}  (close a directed cycle)")
        print(f"  strongly-conn. components: {pr.n_strongly_connected}")
        print(f"  top 8 modules by dphi (sensitivity to removal):")
        for name, contrib in pr.per_module_contribution[:8]:
            print(f"    {name:18s} dphi={contrib:+.4f}")
        print()

    if args.discover:
        print(f"--- Discovery (greedy local search, maximise {args.discover}) ---")
        baseline, props = discover_modifications(
            args.arch_root, metric=args.discover, top_k=args.top_k)
        print(f"  baseline {args.discover}: {baseline:.4f}")
        print(f"  top {len(props)} proposals (d-metric descending):")
        for i, m in enumerate(props, 1):
            arrow = "+" if m.delta_metric > 0 else ""
            print(f"  {i:2d}. {m.kind:14s} {m.source:14s} -> {m.target:14s}  "
                  f"delta={m.delta:+.2f}  new={m.projected_metric:.4f}  "
                  f"({arrow}{m.delta_metric:.4f})")
        print()
        if args.discover in ("generalization", "ppl"):
            print(f"  NOTE: '{args.discover}' uses a STRUCTURAL PROXY")
            print(f"  (literature-grounded topology -> metric correlates).")
            print(f"  For the actual metric you must train each candidate;")
            print(f"  the evolutionary engine in neuroslm/dsl/evolutionary.py")
            print(f"  is the wire-frame for that — connect it to a training")
            print(f"  fitness function (Phase 6 plan).")
        else:
            print(f"  NOTE: structural-only search. Try --discover")
            print(f"  generalization or ppl for training-aware proxies.")
        print()

    if args.topo_10x:
        print("--- Topological 10x OOD candidates (high-leverage stacks) ---")
        cands = discover_topological_10x(args.arch_root)
        for i, c in enumerate(cands, 1):
            print(f"  {i}. {c.name}")
            print(f"     {c.description}")
            if c.edges_added:
                print(f"     + edges: {c.edges_added}")
            if c.edges_removed:
                print(f"     - edges: {c.edges_removed}")
            if c.edges_reweighted:
                print(f"     ~ reweight: {c.edges_reweighted}")
            print(f"     score: {c.baseline_score:.4f} -> {c.new_score:.4f}  "
                  f"({c.improvement * 100:+.1f}%)")
            print()
        print("  Apply a candidate by editing architectures/<arch>/arch.neuro:")
        print("    add `synapse <src> -> <tgt> { weight: X, ... }` blocks")
        print("    or modify existing weights to match the proposal.")
        print()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
