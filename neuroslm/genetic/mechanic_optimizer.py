# -*- coding: utf-8 -*-
"""Optimize commonly-used mechanics — CSE + superoptimization + shared-subexpr.

Given a mechanic as an NGL program, run the full compiler pipeline (DCE → CSE →
constant-fold → algebraic rewriting → probe-verified try-delete) and report
whether it can be reduced without changing behaviour — i.e. whether a
commonly-used computation is carrying redundancy that can be replaced or
optimized away. Also detects subexpressions **shared across** mechanics: the
common computation you would factor out and compute once.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List

from neuroslm.genetic.language import Program
from neuroslm.genetic.simplify import simplify, programs_equivalent
from neuroslm.genetic.rewrite import optimize, to_expr, Node, Leaf, Const


@dataclass
class OptimizeReport:
    name: str
    before: int
    after: int
    equivalent: bool
    reducible: bool
    program: Program


def optimize_mechanic(program: Program, name: str = "mechanic",
                      n_probes: int = 10, seed: int = 0,
                      probes=None) -> OptimizeReport:
    """Run CSE + algebraic + superopt on a mechanic; report the reduction."""
    before = len(program.instructions)
    slim = optimize(program, n_probes=n_probes, seed=seed, probes=probes)
    slim = simplify(slim, n_probes=n_probes, seed=seed, probes=probes)
    after = len(slim.instructions)
    equiv = programs_equivalent(program, slim, n_probes=max(n_probes, 12),
                                seed=seed + 1, probes=probes)
    if not equiv:
        # never return an unsound "optimization" — fall back to the original
        slim, after, reducible = program, before, False
    else:
        reducible = after < before
    return OptimizeReport(name=name, before=before, after=after,
                          equivalent=equiv, reducible=reducible, program=slim)


# ---------------------------------------------------------------------------
# Shared-subexpression detection across mechanics.
# ---------------------------------------------------------------------------
def _collect_subexprs(expr, acc: Dict):
    """Count every non-trivial (Node) subexpression, keyed by a stable string."""
    if isinstance(expr, Node):
        key = _expr_key(expr)
        acc[key] = acc.get(key, 0) + 1
        for a in expr.args:
            _collect_subexprs(a, acc)


def _expr_key(expr) -> str:
    if isinstance(expr, Leaf):
        return expr.reg
    if isinstance(expr, Const):
        return f"const({expr.value:g})"
    if isinstance(expr, Node):
        inner = ",".join(_expr_key(a) for a in expr.args)
        cfg = f",c={expr.const}" if expr.const is not None else ""
        mac = f",@{expr.macro}" if expr.macro else ""
        return f"{expr.op}({inner}{cfg}{mac})"
    return str(expr)


def shared_subexpressions(mechanics: Dict[str, Program]) -> List[dict]:
    """Subexpressions that occur in more than one mechanic (factor-once targets)."""
    # per-mechanic set of subexpressions, then count in how many mechanics each occurs
    in_mechanics: Dict[str, set] = {}
    for name, prog in mechanics.items():
        local: Dict[str, int] = {}
        _collect_subexprs(to_expr(prog), local)
        for key in local:
            in_mechanics.setdefault(key, set()).add(name)
    out = []
    for key, names in in_mechanics.items():
        if len(names) >= 2:
            out.append({"expr": key, "count": len(names), "mechanics": sorted(names)})
    return sorted(out, key=lambda d: -d["count"])


# ---------------------------------------------------------------------------
# The catalog of commonly-used mechanics + a full report.
# ---------------------------------------------------------------------------
def common_mechanics() -> Dict[str, Program]:
    """Commonly-used mechanics as NGL programs (optimizers + arch fragments)."""
    from neuroslm.genetic.optimizer import (
        sgd_program, momentum_program, rmsprop_program, adam_program, lion_program,
    )
    from neuroslm.genetic.attention_primitives import single_head_attention_program
    out = {
        "sgd": sgd_program(),
        "momentum": momentum_program(),
        "rmsprop": rmsprop_program(),
        "adam": adam_program(),
        "lion": lion_program(),
        "attention_1head": single_head_attention_program(),
    }
    # a compiled FFN block, if it lowers cleanly
    try:
        from neuroslm.genetic.compile_arch import compile_layer_to_ngl
        ffn = """
        layer FFN(D, H) {
            param gamma: (D,) init=ones
            param w1: (H, D) init=xavier
            param w2: (H, D) init=xavier
            param w3: (D, H) init=xavier
            forward(x) {
                h = rmsnorm(x, gamma)
                m = swiglu(h, w1, w2, w3)
                return x + m
            }
        }
        """
        out["ffn_block"] = compile_layer_to_ngl(ffn).program
    except Exception:
        pass
    return out


def analyze_common_mechanics(n_probes: int = 8, seed: int = 0) -> List[dict]:
    """Optimize each common mechanic and report the reduction."""
    reports = []
    for name, prog in common_mechanics().items():
        rep = optimize_mechanic(prog, name=name, n_probes=n_probes, seed=seed)
        reports.append({
            "name": name, "before": rep.before, "after": rep.after,
            "removed": rep.before - rep.after, "reducible": rep.reducible,
            "equivalent": rep.equivalent,
        })
    return reports
