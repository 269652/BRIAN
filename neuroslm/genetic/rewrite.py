# -*- coding: utf-8 -*-
"""Algebraic rewrite engine for NGL programs — an equality-driven simplifier.

Pipeline:
1. ``to_expr`` — forward symbolic evaluation of a program into an expression DAG
   (each register's value at each read is its most-recent definition, so register
   reuse is handled correctly). Leaves are input registers; ``const`` ops become
   ``Const`` nodes.
2. rewrite rules — value-preserving algebraic identities (add-0, sub-0, mul-1,
   neg-neg, transpose², ``(a+b)-b → a``, constant folding of ``cscale``, and
   like-term combination ``a·x + b·x → (a+b)·x``). Applied greedily; **every
   accepted rewrite is globally verified** against the original on random probes,
   so an identity that doesn't hold for some shape is rejected rather than
   miscompiled.
3. ``from_expr`` — lower the DAG back to a linear instruction list with
   common-subexpression elimination (shared subterms → one register).

This is what lets ``simplify`` discover reductions a peephole pass can't, e.g.
``(x+x)-x = x`` inside an architecture lowered via ``compile_arch``.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

from neuroslm.genetic.language import Instruction, Program
from neuroslm.genetic.simplify import dead_code_eliminate, programs_equivalent


# ---------------------------------------------------------------------------
# Expression DAG.
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class Leaf:
    reg: str


@dataclass(frozen=True)
class Const:
    value: float


@dataclass(frozen=True)
class Node:
    op: str
    args: Tuple["object", ...]
    const: Optional[float] = None
    config: Tuple[Tuple[str, float], ...] = ()   # scalar-config kwargs (attention …)


Expr = object


def to_expr(program: Program) -> Expr:
    """Forward symbolic evaluation → expression DAG rooted at ``out_reg``."""
    env: Dict[str, Expr] = {}
    for ins in program.instructions:
        if ins.op == "const":
            e: Expr = Const(float(ins.const) if ins.const is not None else 0.0)
        else:
            args = tuple(env.get(r, Leaf(r)) for r in ins.ins)
            e = Node(ins.op, args, ins.const, ins.config)
        env[ins.out] = e
    return env.get(program.out_reg, Leaf(program.out_reg))


def _max_input_index(expr: Expr, acc: int = -1) -> int:
    if isinstance(expr, Leaf):
        if expr.reg.startswith("t"):
            try:
                return max(acc, int(expr.reg[1:]))
            except ValueError:
                return acc
        return acc
    if isinstance(expr, Node):
        for a in expr.args:
            acc = _max_input_index(a, acc)
    return acc


def from_expr(root: Expr, template: Program) -> Program:
    """Lower an expression DAG back to a program (with CSE)."""
    start = max(_max_input_index(root), template.n_tensor - 1) + 1
    counter = [start]
    instrs: List[Instruction] = []
    memo: Dict[Expr, str] = {}

    def fresh() -> str:
        r = f"t{counter[0]}"
        counter[0] += 1
        return r

    def emit(e: Expr) -> str:
        if isinstance(e, Leaf):
            return e.reg
        if e in memo:
            return memo[e]
        if isinstance(e, Const):
            r = fresh()
            instrs.append(Instruction("const", r, (), const=e.value))
            memo[e] = r
            return r
        arg_regs = tuple(emit(a) for a in e.args)
        r = fresh()
        instrs.append(Instruction(e.op, r, arg_regs, e.const, e.config))
        memo[e] = r
        return r

    out = emit(root)
    n_tensor = max(template.n_tensor, counter[0] + 4)
    return Program(
        instructions=instrs,
        n_scalar=template.n_scalar,
        n_tensor=n_tensor,
        out_reg=out,
        meta=dict(template.meta),
    )


# ---------------------------------------------------------------------------
# Rewrite rules — each maps an Expr to a value-equal Expr, or None.
# All are genuine identities in real arithmetic; the guarded ops (div/sqrt/…)
# are never algebraically rewritten here, and the global probe check is the
# safety net for any shape-dependent identity.
# ---------------------------------------------------------------------------
def _is_const(e: Expr, v: float) -> bool:
    return isinstance(e, Const) and abs(e.value - v) < 1e-12


def _r_add_zero(e):
    if isinstance(e, Node) and e.op == "add":
        a, b = e.args
        if _is_const(b, 0.0):
            return a
        if _is_const(a, 0.0):
            return b
    return None


def _r_sub_zero(e):
    if isinstance(e, Node) and e.op == "sub" and _is_const(e.args[1], 0.0):
        return e.args[0]
    return None


def _r_mul_one(e):
    if isinstance(e, Node) and e.op == "mul":
        a, b = e.args
        if _is_const(b, 1.0):
            return a
        if _is_const(a, 1.0):
            return b
    return None


def _r_cscale_one(e):
    if isinstance(e, Node) and e.op == "cscale" and e.const is not None and abs(e.const - 1.0) < 1e-12:
        return e.args[0]
    return None


def _r_neg_neg(e):
    if isinstance(e, Node) and e.op == "neg":
        inner = e.args[0]
        if isinstance(inner, Node) and inner.op == "neg":
            return inner.args[0]
    return None


def _r_transpose_transpose(e):
    if isinstance(e, Node) and e.op == "transpose":
        inner = e.args[0]
        if isinstance(inner, Node) and inner.op == "transpose":
            return inner.args[0]
    return None


def _r_sub_add(e):
    # (a + b) - b -> a ;  (a + b) - a -> b
    if isinstance(e, Node) and e.op == "sub":
        a, b = e.args
        if isinstance(a, Node) and a.op == "add":
            x, y = a.args
            if y == b:
                return x
            if x == b:
                return y
    return None


def _r_add_dup(e):
    # x + x -> cscale(x, 2)
    if isinstance(e, Node) and e.op == "add":
        a, b = e.args
        if a == b:
            return Node("cscale", (a,), 2.0)
    return None


def _r_cscale_fold(e):
    # cscale(cscale(x, a), b) -> cscale(x, a*b)
    if isinstance(e, Node) and e.op == "cscale" and e.const is not None:
        inner = e.args[0]
        if isinstance(inner, Node) and inner.op == "cscale" and inner.const is not None:
            return Node("cscale", (inner.args[0],), e.const * inner.const)
    return None


def _cscale_of(e) -> Optional[Tuple[Expr, float]]:
    if isinstance(e, Node) and e.op == "cscale" and e.const is not None:
        return e.args[0], e.const
    return None


def _r_combine_like_terms(e):
    # a*x + b*x -> (a+b)*x ; a*x + x -> (a+1)*x ; x + a*x -> (a+1)*x ; x + x handled by _r_add_dup
    if isinstance(e, Node) and e.op == "add":
        a, b = e.args
        ca, cb = _cscale_of(a), _cscale_of(b)
        if ca and cb and ca[0] == cb[0]:
            return Node("cscale", (ca[0],), ca[1] + cb[1])
        if ca and ca[0] == b:
            return Node("cscale", (b,), ca[1] + 1.0)
        if cb and cb[0] == a:
            return Node("cscale", (a,), cb[1] + 1.0)
    return None


def _r_sub_cscale(e):
    # (a*x) - x -> (a-1)*x
    if isinstance(e, Node) and e.op == "sub":
        a, b = e.args
        ca = _cscale_of(a)
        if ca and ca[0] == b:
            return Node("cscale", (b,), ca[1] - 1.0)
    return None


_RULES = [
    _r_add_zero, _r_sub_zero, _r_mul_one, _r_cscale_one,
    _r_neg_neg, _r_transpose_transpose, _r_sub_add, _r_add_dup,
    _r_cscale_fold, _r_combine_like_terms, _r_sub_cscale,
]


def _rewrite_candidates(expr: Expr):
    """Yield every Expr obtainable by applying one rule at one node."""
    for rule in _RULES:
        out = rule(expr)
        if out is not None and out != expr:
            yield out
    if isinstance(expr, Node):
        for i, child in enumerate(expr.args):
            for new_child in _rewrite_candidates(child):
                yield Node(expr.op, expr.args[:i] + (new_child,) + expr.args[i + 1:],
                           expr.const, expr.config)


def _cost(program: Program) -> int:
    return len(program.instructions)


def algebraic_simplify(program: Program, n_probes: int = 8, seed: int = 0,
                       max_iters: int = 64, probes=None) -> Program:
    """Greedily apply verified algebraic rewrites to a fixpoint."""
    cur = dead_code_eliminate(program)
    for _ in range(max_iters):
        expr = to_expr(cur)
        improved = False
        for cand_expr in _rewrite_candidates(expr):
            cand = dead_code_eliminate(from_expr(cand_expr, cur))
            if _cost(cand) < _cost(cur) and programs_equivalent(
                    cur, cand, n_probes=n_probes, seed=seed, probes=probes):
                cur = cand
                improved = True
                break
        if not improved:
            break
    return cur
