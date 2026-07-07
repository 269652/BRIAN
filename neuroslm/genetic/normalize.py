# -*- coding: utf-8 -*-
"""Semantic normalization — reduce equivalent expressions to a canonical form.

The NGL search produces the *same* computation written a hundred ways:
``neg(neg(x))``, ``x + 0``, ``x · 1`` all mean ``x``; ``x · x`` and ``square(x)``
mean the same thing. Left alone, the explorer re-searches every syntactic
variant. This pass collapses each equivalence class to **one canonical
representative** and substitutes it everywhere, so exploration only ever sees the
normal form.

Two soundly-verified layers of equality (see the module docstring in
``tests/genetic/test_normalize.py`` for the contract):

1. **Rewrite-equal** — both programs reduce to the *identical* canonical form
   under the repo's convergent rewrite system (``optimize`` + ``simplify`` run to
   a fixpoint). Identical normal form is a *proof* of equality within that
   theory — this is the decidable core (a confluent, terminating TRS gives unique
   normal forms; Church–Rosser).
2. **Probe-equal** — canonical forms differ (the rewrite theory can't connect
   ``x·x`` to ``square(x)``) but the two agree on many random probe inputs. For
   NGL's total, side-effect-free, polynomial/analytic ops this is behavioural
   equivalence with high confidence (finite-probe polynomial identity testing;
   Schwartz–Zippel). We merge these classes and keep the simpler form.

Canonical selection: the most-used member (``prefer="frequency"``) or the
lowest-complexity one (``prefer="simplest"``). Complexity is
``(n_instructions, n_branches, n_distinct_ops)`` — a straight-line DAG proxy for
cyclomatic complexity, with data-selection ops (``select/gt/min/max``) counted as
decision points.

**What this is and isn't.** General program-equivalence is undecidable (Rice's
theorem), and the minimal program for an arbitrary intent is uncomputable
(Kolmogorov complexity). So this pass is *complete only relative to its rewrite
theory and probe budget* — it is sound (never merges things it can't verify) and
practically strong on the bounded fragment NGL lives in, not a universal
intent-minimizer. That honest boundary is the whole design.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from neuroslm.genetic.language import Program
from neuroslm.genetic.rewrite import optimize, to_expr, Leaf, Const, Node
from neuroslm.genetic.simplify import simplify, programs_equivalent


# ---------------------------------------------------------------------------
# Canonical form + signature.
# ---------------------------------------------------------------------------
def canonical_form(program: Program, n_probes: int = 10, seed: int = 0,
                   probes=None, max_rounds: int = 6) -> Program:
    """Reduce a program to its normal form under the convergent rewrite system."""
    cur = program
    for _ in range(max_rounds):
        n0 = len(cur.instructions)
        cur = optimize(cur, n_probes=n_probes, seed=seed, probes=probes)
        cur = simplify(cur, n_probes=n_probes, seed=seed, probes=probes)
        if len(cur.instructions) >= n0:
            break
    return cur


def _expr_str(e) -> str:
    if isinstance(e, Leaf):
        return e.reg
    if isinstance(e, Const):
        return f"c{e.value:g}"
    if isinstance(e, Node):
        inner = ",".join(_expr_str(a) for a in e.args)
        cfg = f"|{e.const:g}" if e.const is not None else ""
        conf = f"|{e.config}" if e.config else ""
        mac = f"@{e.macro}" if e.macro else ""
        return f"{e.op}({inner}{cfg}{conf}{mac})"
    return str(e)


def semantic_signature(program: Program, **kw) -> str:
    """Structural key of the canonical form — equal ⇒ rewrite-provably equal."""
    return _expr_str(to_expr(canonical_form(program, **kw)))


# ---------------------------------------------------------------------------
# Complexity metrics.
# ---------------------------------------------------------------------------
_DECISION_OPS = frozenset({"select", "gt", "min", "max"})


def cyclomatic(program: Program) -> int:
    """McCabe-style complexity: 1 + number of data-selection decision points.

    NGL programs are straight-line, so control-flow cyclomatic complexity is
    trivially 1; we extend the count to data-selection ops (``select``/``gt``/
    ``min``/``max``), which are the value-level branches an NGL program expresses.
    """
    return 1 + sum(1 for i in program.instructions if i.op in _DECISION_OPS)


def complexity(program: Program) -> Tuple[int, int, int]:
    """Lexicographic cost — lower is simpler. (size, branches, distinct ops)."""
    n = len(program.instructions)
    branches = sum(1 for i in program.instructions if i.op in _DECISION_OPS)
    distinct = len({i.op for i in program.instructions})
    return (n, branches, distinct)


# ---------------------------------------------------------------------------
# The normalization pass.
# ---------------------------------------------------------------------------
@dataclass
class SemanticClass:
    canonical: str                       # the chosen representative's name
    members: List[str]
    signature: str
    reason: str                          # "frequency" | "simplest"
    program: Program                     # canonical form substituted for the class


@dataclass
class NormalizeResult:
    classes: List[SemanticClass]
    canonical_of: Dict[str, str]         # name → canonical name
    programs: Dict[str, Program]         # name → substituted (canonical) program

    def to_dict(self) -> dict:
        return {
            "classes": [
                {"canonical": c.canonical, "members": c.members,
                 "signature": c.signature[:80], "reason": c.reason,
                 "n_members": len(c.members)}
                for c in self.classes
            ],
            "canonical_of": self.canonical_of,
            "n_classes": len(self.classes),
        }


def _merge_probe_equal(reps: List[Tuple[str, Program]], roles: Dict[str, str],
                       stateful: Dict[str, bool], n_probes: int, seed: int) -> List[List[str]]:
    """Union-find over per-signature reps: merge probe-equal ones (same role).

    Stateful programs are never probe-merged: their state registers read as zero
    in a single-shot probe, so ``momentum`` and ``sgd`` *look* equal on one step
    even though they diverge across steps. Single-shot observational equivalence
    is only a sound equality witness for stateless (pure) programs.
    """
    parent = {name: name for name, _ in reps}

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a, b):
        parent[find(a)] = find(b)

    for i in range(len(reps)):
        ni, pi = reps[i]
        for j in range(i + 1, len(reps)):
            nj, pj = reps[j]
            if find(ni) == find(nj):
                continue
            if roles.get(ni) != roles.get(nj):
                continue     # only merge within the same semantic role
            if stateful.get(ni) or stateful.get(nj):
                continue     # single-shot probes can't witness stateful equality
            if programs_equivalent(pi, pj, n_probes=n_probes, seed=seed):
                union(ni, nj)
    groups: Dict[str, List[str]] = {}
    for name, _ in reps:
        groups.setdefault(find(name), []).append(name)
    return list(groups.values())


def normalize_semantics(programs: Dict[str, Program],
                        counts: Optional[Dict[str, int]] = None,
                        prefer: str = "frequency",
                        n_probes: int = 10, seed: int = 0,
                        role_of: Optional[Dict[str, str]] = None) -> NormalizeResult:
    """Cluster equivalent programs and substitute one canonical form per class.

    ``role_of`` supplies the semantic labels (from ``semantics.analyze``) so
    normalization runs *after* labelling: only same-role programs are ever probe-
    merged, which keeps the O(k²) merge both cheap and semantically honest.
    """
    counts = counts or {}
    # semantic labels per program (used to gate probe merges): role + statefulness
    from neuroslm.genetic.semantics import analyze
    stateful_of: Dict[str, bool] = {}
    if role_of is None:
        role_of = {}
    for name, prog in programs.items():
        try:
            summ = analyze(prog)
            role_of.setdefault(name, summ.role)
            stateful_of[name] = summ.stateful
        except Exception:
            role_of.setdefault(name, "generic")
            stateful_of[name] = False

    # 1. canonicalize + signature each program
    canon: Dict[str, Program] = {}
    sig: Dict[str, str] = {}
    for name, prog in programs.items():
        c = canonical_form(prog, n_probes=n_probes, seed=seed)
        canon[name] = c
        sig[name] = _expr_str(to_expr(c))

    # 2. exact-signature buckets (rewrite-provably equal)
    by_sig: Dict[str, List[str]] = {}
    for name in programs:
        by_sig.setdefault(sig[name], []).append(name)

    # 3. merge buckets that are probe-equal (same role) — one rep per bucket
    reps = [(names[0], canon[names[0]]) for names in by_sig.values()]
    rep_role = {names[0]: role_of.get(names[0], "generic") for names in by_sig.values()}
    rep_state = {names[0]: stateful_of.get(names[0], False) for names in by_sig.values()}
    merged_groups = _merge_probe_equal(reps, rep_role, rep_state,
                                       n_probes=n_probes, seed=seed + 1)

    # 4. build final clusters (union the sig-buckets a rep stands for)
    def _pick(members: List[str]) -> Tuple[str, str]:
        if prefer == "simplest":
            key = lambda n: (complexity(canon[n]), -counts.get(n, 0), n)
            return min(members, key=key), "simplest"
        # frequency: most-used, tie-break simplest then name
        key = lambda n: (-counts.get(n, 0), complexity(canon[n]), n)
        return min(members, key=key), "frequency"

    classes: List[SemanticClass] = []
    canonical_of: Dict[str, str] = {}
    out_programs: Dict[str, Program] = {}
    for rep_group in merged_groups:
        members: List[str] = []
        for rep in rep_group:
            members.extend(by_sig[sig[rep]])
        members = sorted(set(members))
        chosen, reason = _pick(members)
        canon_prog = canon[chosen]
        cls = SemanticClass(canonical=chosen, members=members,
                            signature=sig[chosen], reason=reason, program=canon_prog)
        classes.append(cls)
        for m in members:
            canonical_of[m] = chosen
            out_programs[m] = canon_prog
    classes.sort(key=lambda c: (-len(c.members), c.canonical))
    return NormalizeResult(classes=classes, canonical_of=canonical_of,
                           programs=out_programs)
