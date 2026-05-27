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


# ── Whole-architecture system ──────────────────────────────────────────

def architecture_to_wolfram(arch_root) -> str:
    """Compile every population's equation/ode in an architecture folder
    into a Wolfram list of equations `{eq1, eq2, ...}` — a system a CAS
    can analyze (steady states, Jacobians, stability) as a whole."""
    from .multifile import compile_folder
    ir = compile_folder(Path(arch_root))

    eqns: List[str] = []
    for pop in ir.populations:
        if getattr(pop, "ode", None):
            eqns.append(ode_to_wolfram(pop.ode))
        elif getattr(pop, "equation", None):
            # Tag the output with the population name for readability.
            eq = parse_equation(pop.equation)
            rhs = _to_wolfram_expr(eq.rhs)
            eqns.append(f"{pop.name} == {rhs}")
    return "{" + ", ".join(eqns) + "}"


def save_wolfram(arch_root, path: str) -> None:
    """Write the architecture's Wolfram system to a file for inspection."""
    code = architecture_to_wolfram(arch_root)
    with open(path, "w", encoding="utf-8") as f:
        f.write(code + "\n")
