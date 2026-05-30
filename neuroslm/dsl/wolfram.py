# -*- coding: utf-8 -*-
"""Compile .neuro equations to Wolfram Alpha / Mathematica syntax.

The DSL's equations already lower to SymPy expressions (equations.py).
This module expands the neural nonlinearities into closed forms and emits
Wolfram-language code via SymPy's Mathematica printer, so any population
equation, synapse, ODE, or whole-architecture system can be pasted into
Wolfram Alpha and solved / DSolved against arbitrary variables.

Nonlinearity mapping:
    ReLU(x)    → Max[0, x]
    sigmoid(x) → 1/(1 + E^(-x))
    tanh(x)    → Tanh[x]
    softmax(x) → Exp[x]/Total[Exp[x]]   (vector; emitted symbolically)

This is the bridge toward the N10/N11 hypershape goal: the model's
mechanics become closed-form math an external CAS can reason about.
"""
from __future__ import annotations
from pathlib import Path
from typing import List, Optional

import sympy as sp
from sympy import mathematica_code

from .equations import parse_equation, parse_ode, _expand_for_analysis


def _to_wolfram_expr(expr: sp.Expr) -> str:
    """Expand nonlinearities to closed forms, then print Wolfram code."""
    return mathematica_code(_expand_for_analysis(expr))


# ── Algebraic equations ────────────────────────────────────────────────

def equation_to_wolfram(eq_str: str) -> str:
    """`y = ReLU(x)` → `y == Max[0, x]` (Wolfram equation)."""
    eq = parse_equation(eq_str)
    rhs = _to_wolfram_expr(eq.rhs)
    return f"{eq.lhs} == {rhs}"


# ── ODEs ───────────────────────────────────────────────────────────────

def ode_to_wolfram(ode_str: str, dsolve: bool = False) -> str:
    """`dV/dt = -V + x` → `V'[t] == -V[t] + x` (or wrapped in DSolve[...]).

    The state variable V is rewritten as the time-dependent V[t]; the
    derivative as V'[t], which is what Wolfram's DSolve expects.
    """
    ode = parse_ode(ode_str)
    var = ode.state_var
    t = sp.Symbol("t")
    Vt = sp.Function(var)(t)
    # Replace the bare state symbol with V[t] in the RHS.
    rhs = _expand_for_analysis(ode.rhs).subs(sp.Symbol(var), Vt)
    rhs_w = mathematica_code(rhs)
    eqn = f"{var}'[t] == {rhs_w}"
    if dsolve:
        return f"DSolve[{eqn}, {var}[t], t]"
    return eqn


# ── Fixed-point / steady-state solving ─────────────────────────────────

def solve_fixed_point_wolfram(body_str: str, is_ode: bool = False,
                              input_symbol: str = "x") -> str:
    """Emit a Wolfram `Solve[...]` for the equation's fixed point.

    Algebraic `y = f(x)`: solves `f(x) - x == 0` for x (the recurrence
    fixed point). ODE `dV/dt = g(...)`: solves `g == 0` for V (steady
    state).
    """
    if is_ode:
        ode = parse_ode(body_str)
        rhs = _expand_for_analysis(ode.rhs)
        var = ode.state_var
        return f"Solve[{mathematica_code(rhs)} == 0, {var}]"
    eq = parse_equation(body_str)
    rhs = _expand_for_analysis(eq.rhs)
    sym = sp.Symbol(input_symbol)
    expr = rhs - sym
    return f"Solve[{mathematica_code(expr)} == 0, {input_symbol}]"


# ── Whole-architecture system (populations only — legacy) ─────────────

def architecture_to_wolfram(arch_root) -> str:
    """Compile every population's equation/ode in an architecture folder
    into a Wolfram list of equations `{eq1, eq2, ...}` — a system a CAS
    can analyze (steady states, Jacobians, stability) as a whole.

    Legacy: only emits *population* activations. For the IIT-grade full
    formulation (populations + synapses + modulations + NT dynamics)
    use `architecture_to_wolfram_full`.
    """
    from .multifile import compile_folder
    ir = compile_folder(Path(arch_root))

    eqns: List[str] = []
    for pop in ir.populations:
        if getattr(pop, "ode", None):
            eqns.append(ode_to_wolfram(pop.ode))
        elif getattr(pop, "equation", None):
            eq = parse_equation(pop.equation)
            rhs = _to_wolfram_expr(eq.rhs)
            eqns.append(f"{pop.name} == {rhs}")
    return "{" + ", ".join(eqns) + "}"


# ── Whole-architecture system (IIT-grade) ────────────────────────────

def architecture_to_wolfram_full(arch_root, *,
                                  include_populations: bool = True,
                                  include_synapses: bool = True,
                                  include_modulations: bool = True,
                                  include_nt_dynamics: bool = True
                                  ) -> str:
    """Compile the *complete* architecture into one Wolfram system.

    For an IIT-grade formulation the system needs every causal element:
      - Population activations           — y_<pop> == f(x)
      - Synapse transmissions           — y_<src>_<tgt> == w * (x_src @ W)
      - Modulation rules                — y_<pop>_mod  == f(y_<pop>, c_<nt>)
      - NT homeostatic ODEs              — dc_<nt>/dt   == release - reuptake*(c - base)

    Together these form a closed dynamical system Wolfram can analyze
    for fixed points (Solve[..., {vars}]), stability (Eigenvalues of the
    Jacobian), bifurcations, and structural Φ proxies (graph properties
    over the synapse adjacency).

    Use `include_*` flags to slice the formulation when a sub-system is
    enough (e.g. populations-only for a smaller paste into Wolfram Alpha,
    which has an input-length cap).
    """
    from .multifile import compile_folder
    ir = compile_folder(Path(arch_root))

    sections: List[str] = []

    # ── 1. Populations ────────────────────────────────────────────────
    if include_populations:
        pop_eqns: List[str] = []
        for pop in ir.populations:
            if getattr(pop, "ode", None):
                pop_eqns.append(ode_to_wolfram(pop.ode))
            elif getattr(pop, "equation", None):
                eq = parse_equation(pop.equation)
                rhs = _to_wolfram_expr(eq.rhs)
                pop_eqns.append(f"y_{pop.name} == {rhs}")
        if pop_eqns:
            sections.append(("Populations", pop_eqns))

    # ── 2. Synapses ───────────────────────────────────────────────────
    # Each synapse adds  y_<src>_to_<tgt> == w * f(x_<src>, W_<src>_<tgt>)
    if include_synapses:
        syn_eqns: List[str] = []
        for syn in ir.synapses:
            w = syn.weight if syn.weight is not None else 1.0
            edge_name = f"y_{syn.source}_to_{syn.target}"
            # Use the synapse's own equation if specified; else a plain
            # weighted matrix product (the rcc_bowtie default).
            if getattr(syn, "equation", None):
                # Rewrite the symbolic `x_pre @ W` into the source-tagged
                # variables that this synapse instance owns.
                try:
                    eq = parse_equation(syn.equation)
                    rhs_w = _to_wolfram_expr(eq.rhs)
                    rhs_w = rhs_w.replace("x_pre",  f"x_{syn.source}")
                    rhs_w = rhs_w.replace("W",      f"W_{syn.source}_{syn.target}")
                    rhs_w = rhs_w.replace("weight", str(w))
                    syn_eqns.append(f"{edge_name} == {rhs_w}")
                    continue
                except Exception:
                    pass   # fall through to the default linear form
            syn_eqns.append(
                f"{edge_name} == {w}*(x_{syn.source} . W_{syn.source}_{syn.target})"
            )
        if syn_eqns:
            sections.append(("Synapses", syn_eqns))

    # ── 3. Modulations (NT effects on regions) ────────────────────────
    # Multiplicative: y_<pop>_mod == y_<pop> * (c_<nt> * gain)
    # Additive:       y_<pop>_mod == y_<pop> + (c_<nt> * gain)
    if include_modulations:
        mod_eqns: List[str] = []
        for mod in ir.modulations:
            g = mod.gain if mod.gain is not None else 1.0
            mod_name = f"y_{mod.target_population}_mod_{mod.source_nt}"
            if mod.effect == "additive":
                mod_eqns.append(
                    f"{mod_name} == y_{mod.target_population} + (c_{mod.source_nt}*{g})"
                )
            else:   # default multiplicative
                mod_eqns.append(
                    f"{mod_name} == y_{mod.target_population}*(c_{mod.source_nt}*{g})"
                )
        if mod_eqns:
            sections.append(("Modulations", mod_eqns))

    # ── 4. Neurotransmitter homeostatic dynamics ──────────────────────
    # dc/dt == release_rate * activity_<nt> - reuptake_rate * (c - base)
    # The "activity_<nt>" term is left symbolic — it is the per-step
    # release driven by upstream firing, which the CAS user fills in.
    if include_nt_dynamics:
        nt_eqns: List[str] = []
        for nt in ir.neurotransmitter_systems:
            base = nt.base_concentration if nt.base_concentration is not None else 0.0
            rel = nt.release_rate if nt.release_rate is not None else 0.0
            reup = nt.reuptake_rate if nt.reuptake_rate is not None else 0.0
            nt_eqns.append(
                f"c_{nt.name}'[t] == {rel}*activity_{nt.name}[t] "
                f"- {reup}*(c_{nt.name}[t] - {base})"
            )
        if nt_eqns:
            sections.append(("NeurotransmitterDynamics", nt_eqns))

    # ── Assemble into a single Wolfram Association for clarity ───────
    # Output format:
    #   <|
    #     "Populations" -> {eq1, eq2, ...},
    #     "Synapses"    -> {eq1, eq2, ...},
    #     ...
    #   |>
    # which is a Wolfram Association — directly pasteable; each branch
    # can be Solve'd / DSolve'd / Jacobian'd independently or combined
    # via `Join @@ Values[%]` to get the flat system.
    parts = [
        f'  "{name}" -> {{{", ".join(eqs)}}}'
        for name, eqs in sections
    ]
    return "<|\n" + ",\n".join(parts) + "\n|>"


def save_wolfram(arch_root, path: str, full: bool = False) -> None:
    """Write the architecture's Wolfram system to a file for inspection.

    Args:
        full: when True, emits the IIT-grade formulation (populations +
              synapses + modulations + NT dynamics). When False, only
              population equations (legacy behavior).
    """
    if full:
        code = architecture_to_wolfram_full(arch_root)
    else:
        code = architecture_to_wolfram(arch_root)
    with open(path, "w", encoding="utf-8") as f:
        f.write(code + "\n")
