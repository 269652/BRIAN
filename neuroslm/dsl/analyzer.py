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
    measured Φ trajectory against the symbolic prediction).

CLI:
    py -3 -m neuroslm.dsl.analyzer architectures/rcc_bowtie \\
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

    # 1a. Populations
    for pop in ir.populations:
        if getattr(pop, "ode", None):
            ode = parse_ode(pop.ode)
            rhs = _expand_for_analysis(ode.rhs)
            sv = sp.Symbol(f"V_{pop.name}")
            sys.state_vars.append(sv)
            # rewrite the bare state symbol → tagged
            rhs_t = rhs.subs(sp.Symbol(ode.state_var), sv)
            sys.equations[sv] = rhs_t
        elif getattr(pop, "equation", None):
            eq = parse_equation(pop.equation)
            rhs = _expand_for_analysis(eq.rhs)
            sv = sp.Symbol(f"y_{pop.name}")
            sys.state_vars.append(sv)
            sys.equations[sv] = rhs

    # 1b. Synapses — algebraic equations, no time evolution
    for syn in ir.synapses:
        w = syn.weight if syn.weight is not None else 1.0
        sv = sp.Symbol(f"y_{syn.source}_to_{syn.target}")
        x_pre = sp.Symbol(f"x_{syn.source}")
        W = sp.Symbol(f"W_{syn.source}_{syn.target}")
        # Default form `y = w * (x_pre @ W)` — at the symbolic level
        # `@` becomes ordinary multiplication (we're not tracking shapes
        # here, only the closed-form dependency structure).
        rhs = w * x_pre * W
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
      - all Re(λ) < 0       → stable
      - all Re(λ) > 0       → unstable
      - mixed signs         → saddle
      - any |Im(λ)| > 0 and Re ≈ 0 → oscillatory
      - any Re(λ) ≈ 0       → marginal
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
    print(f"  max Re(λ):       {stab.largest_real_part:.4f}")
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
    p.add_argument("--all", action="store_true",
                   help="run every analysis above")
    args = p.parse_args(argv)

    if args.all:
        args.fixed_points = args.jacobian = args.stability = args.wa_queries = True
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
        print(f"  shape: {J.shape[0]} × {J.shape[1]}")
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
        print(f"--- Rendering topology graph → {args.graph} ---")
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

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
