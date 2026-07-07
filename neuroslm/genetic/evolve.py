# -*- coding: utf-8 -*-
"""Genetic operators + Pareto GA over NGL programs.

This is the "search the language space" engine. Because NGL execution is total
(``language.py``), mutation/crossover never need to reject a child for crashing —
they only keep programs *structurally* valid (known ops, in-range registers, a
real ``out_reg``). Selection is multi-objective (Pareto), so a discovery run can
push loss down *and* throughput / effective-information up at once.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, List, Sequence, Tuple

import numpy as np

from neuroslm.genetic.language import (
    Instruction,
    Program,
    REGISTRY,
    Memory,
)

# Searchable op pool. The composite `nn` family (linear/rmsnorm/swiglu/…) is
# excluded: those need bound parameter registers and are only meaningful when an
# architecture is *lowered* into NGL (compile_arch.py), not when a program is
# grown from scratch. Restricting the pool keeps blind search focused on the
# primitive grammar (arith/reduce/control/nonlin/linalg/const).
_OP_NAMES = sorted(n for n, s in REGISTRY.items() if s.family != "nn")


# ---------------------------------------------------------------------------
# Register addressing helpers.
# ---------------------------------------------------------------------------
def _rand_reg(rng: np.random.Generator, n_scalar: int, n_tensor: int,
              tensor_bias: float = 0.75) -> str:
    if rng.random() < tensor_bias and n_tensor > 0:
        return f"t{rng.integers(n_tensor)}"
    return f"s{rng.integers(max(1, n_scalar))}"


def _rand_instruction(rng: np.random.Generator, n_scalar: int, n_tensor: int) -> Instruction:
    op = _OP_NAMES[rng.integers(len(_OP_NAMES))]
    spec = REGISTRY[op]
    ins = tuple(_rand_reg(rng, n_scalar, n_tensor) for _ in range(spec.n_in))
    out = _rand_reg(rng, n_scalar, n_tensor)
    const = float(rng.normal(0, 0.5)) if spec.uses_const else None
    return Instruction(op, out, ins, const)


def random_program(rng: np.random.Generator, length: int = 8,
                   n_scalar: int = 4, n_tensor: int = 8) -> Program:
    """A random but structurally-valid program.

    ``t0``/``t1`` are the conventional grad/param inputs; the out register is
    biased toward a tensor register that is actually written, so the program's
    output is rarely a trivial zero.
    """
    instrs = [_rand_instruction(rng, n_scalar, n_tensor) for _ in range(max(1, length))]
    # prefer an out_reg that some instruction writes to (non-trivial output)
    written = [i.out for i in instrs if i.out.startswith("t")]
    out_reg = written[-1] if written else "t0"
    return Program(instrs, n_scalar=n_scalar, n_tensor=n_tensor, out_reg=out_reg)


# ---------------------------------------------------------------------------
# Mutation.
# ---------------------------------------------------------------------------
_MUT_KINDS = ("point_op", "point_reg", "point_const", "insert", "delete", "out_reg")


def mutate(program: Program, rng: np.random.Generator,
           kind: str | None = None, library=None) -> Program:
    """Return a mutated copy. The parent is never modified.

    When ``library`` (a ``MacroLibrary``) is given, an ``insert_call`` move can
    graft a whole macro as one instruction — the abstraction lever that lets the
    search compose higher-order algorithms instead of re-deriving primitives.
    """
    child = program.copy()
    if library is not None:
        child.library = library
    instrs = list(child.instructions)
    ns, nt = child.n_scalar, child.n_tensor

    if kind is None:
        choices = list(_MUT_KINDS)
        if len(instrs) <= 1:
            choices = [k for k in choices if k != "delete"]
        if library is not None and len(library):
            choices = choices + ["insert_call"]
        kind = choices[rng.integers(len(choices))]

    if kind == "insert_call" and library is not None and len(library):
        macro = library.macros()[rng.integers(len(library))]
        ins_regs = tuple(_rand_reg(rng, ns, nt) for _ in range(macro.n_inputs))
        out = f"t{rng.integers(nt)}"
        pos = rng.integers(len(instrs) + 1) if instrs else 0
        instrs.insert(int(pos), Instruction("call", out, ins_regs, macro=macro.name))
        if child.out_reg not in ("t0", "t1"):
            child.out_reg = out
    elif kind == "insert" or not instrs:
        pos = rng.integers(len(instrs) + 1) if instrs else 0
        instrs.insert(int(pos), _rand_instruction(rng, ns, nt))
    elif kind == "delete" and len(instrs) > 1:
        del instrs[int(rng.integers(len(instrs)))]
    else:
        i = int(rng.integers(len(instrs)))
        old = instrs[i]
        if kind == "point_op":
            new_op = _OP_NAMES[rng.integers(len(_OP_NAMES))]
            spec = REGISTRY[new_op]
            ins = tuple(
                old.ins[j] if j < len(old.ins) else _rand_reg(rng, ns, nt)
                for j in range(spec.n_in)
            )
            const = old.const if spec.uses_const else None
            if spec.uses_const and const is None:
                const = float(rng.normal(0, 0.5))
            instrs[i] = Instruction(new_op, old.out, ins, const)
        elif kind == "point_reg":
            spec = REGISTRY[old.op]
            if spec.n_in and rng.random() < 0.5:
                j = int(rng.integers(spec.n_in))
                ins = list(old.ins)
                ins[j] = _rand_reg(rng, ns, nt)
                instrs[i] = Instruction(old.op, old.out, tuple(ins), old.const)
            else:
                instrs[i] = Instruction(old.op, _rand_reg(rng, ns, nt), old.ins, old.const)
        elif kind == "point_const":
            spec = REGISTRY[old.op]
            base = old.const if (old.const is not None) else 0.0
            new_const = base + float(rng.normal(0, 0.3)) if spec.uses_const else old.const
            instrs[i] = Instruction(old.op, old.out, old.ins, new_const)
        elif kind == "out_reg":
            written = [k.out for k in instrs if k.out.startswith("t")]
            child.out_reg = written[rng.integers(len(written))] if written else child.out_reg

    child.instructions = instrs
    # keep out_reg pointing at a written register when possible
    if child.out_reg not in {k.out for k in instrs} and child.out_reg not in ("t0", "t1"):
        written = [k.out for k in instrs if k.out.startswith("t")]
        if written:
            child.out_reg = written[-1]
    return child


# ---------------------------------------------------------------------------
# Crossover.
# ---------------------------------------------------------------------------
def crossover(a: Program, b: Program, rng: np.random.Generator) -> Program:
    """One-point splice of two instruction lists into a valid child."""
    ns = max(a.n_scalar, b.n_scalar)
    nt = max(a.n_tensor, b.n_tensor)
    ca = list(a.instructions)
    cb = list(b.instructions)
    ia = int(rng.integers(len(ca) + 1)) if ca else 0
    ib = int(rng.integers(len(cb) + 1)) if cb else 0
    instrs = ca[:ia] + cb[ib:]
    if not instrs:
        instrs = ca or cb or [_rand_instruction(rng, ns, nt)]
    written = [k.out for k in instrs if k.out.startswith("t")]
    out_reg = written[-1] if written else a.out_reg
    return Program(instrs, n_scalar=ns, n_tensor=nt, out_reg=out_reg)


# ---------------------------------------------------------------------------
# Multi-objective machinery.
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class Objective:
    """A tuple of objective values, **all maximised**.

    (Minimised metrics like loss should be negated before being wrapped.)
    """
    values: Tuple[float, ...]

    def dominates(self, other: "Objective") -> bool:
        ge = all(x >= y for x, y in zip(self.values, other.values))
        gt = any(x > y for x, y in zip(self.values, other.values))
        return ge and gt

    def scalar(self, weights: Sequence[float] | None = None) -> float:
        if weights is None:
            return float(sum(self.values))
        return float(sum(w * v for w, v in zip(weights, self.values)))


def pareto_front(pop: Sequence[Tuple[object, Objective]]) -> List[object]:
    """Return the non-dominated individuals (first objects of each pair)."""
    front = []
    for i, (indiv, obj) in enumerate(pop):
        dominated = False
        for j, (_, other) in enumerate(pop):
            if i != j and other.dominates(obj):
                dominated = True
                break
        if not dominated:
            front.append(indiv)
    return front


# ---------------------------------------------------------------------------
# The GA loop.
# ---------------------------------------------------------------------------
@dataclass
class EvolveResult:
    best_program: Program
    best_objective: Objective
    gen0_best: float
    history: List[float]
    front: List[Program]


def _tournament(pop, objs, rng, k=3, weights=None):
    idx = rng.integers(len(pop), size=k)
    best = idx[0]
    for j in idx[1:]:
        if objs[j].scalar(weights) > objs[best].scalar(weights):
            best = j
    return pop[best]


def auto_evolve(
    evaluate: Callable[[Program], Objective],
    rng: np.random.Generator,
    pop_size: int = 32,
    generations: int = 20,
    length: int = 8,
    n_scalar: int = 4,
    n_tensor: int = 8,
    seeds: Sequence[Program] | None = None,
    elite_frac: float = 0.2,
    crossover_rate: float = 0.5,
    weights: Sequence[float] | None = None,
    novelty_weight: float = 0.0,
    on_generation: Callable[[int, int, "Objective"], None] | None = None,
) -> EvolveResult:
    """Evolve a population of NGL programs against ``evaluate``.

    ``evaluate`` returns an ``Objective`` (all-maximised). Selection is a scalar
    tournament (weighted sum, optionally + a novelty bonus in semantic space)
    with elitism carrying the current Pareto front forward.

    ``on_generation(gen, total, best_objective)`` — if given, is called once for
    the initial population (gen 0) and once after each generation, so a caller
    can stream progress during a long run.
    """
    pop: List[Program] = []
    if seeds:
        pop.extend(p.copy() for p in seeds)
    while len(pop) < pop_size:
        pop.append(random_program(rng, length, n_scalar, n_tensor))
    pop = pop[:pop_size]

    def scored(programs):
        objs = [evaluate(p) for p in programs]
        if novelty_weight > 0.0:
            embs = np.stack([p.semantic_vector() for p in programs])
            objs = _add_novelty(programs, objs, embs, novelty_weight)
        return objs

    objs = scored(pop)
    gen0_best = max(o.scalar(weights) for o in objs)
    best_i = int(np.argmax([o.scalar(weights) for o in objs]))
    best_prog, best_obj = pop[best_i].copy(), objs[best_i]
    history = [best_obj.scalar(weights)]
    if on_generation is not None:
        on_generation(0, generations, best_obj)

    n_elite = max(1, int(elite_frac * pop_size))
    for _gen in range(generations):
        order = np.argsort([-o.scalar(weights) for o in objs])
        new_pop = [pop[i].copy() for i in order[:n_elite]]
        while len(new_pop) < pop_size:
            if rng.random() < crossover_rate:
                pa = _tournament(pop, objs, rng, weights=weights)
                pb = _tournament(pop, objs, rng, weights=weights)
                child = crossover(pa, pb, rng)
                child = mutate(child, rng)
            else:
                pa = _tournament(pop, objs, rng, weights=weights)
                child = mutate(pa, rng)
            new_pop.append(child)
        pop = new_pop
        objs = scored(pop)
        gi = int(np.argmax([o.scalar(weights) for o in objs]))
        if objs[gi].scalar(weights) > best_obj.scalar(weights):
            best_prog, best_obj = pop[gi].copy(), objs[gi]
        history.append(best_obj.scalar(weights))
        if on_generation is not None:
            on_generation(_gen + 1, generations, best_obj)

    front = pareto_front(list(zip(pop, objs)))
    return EvolveResult(best_prog, best_obj, gen0_best, history, front)


def _add_novelty(programs, objs, embs, w):
    # mean distance to the rest of the population in semantic space
    out = []
    for i, obj in enumerate(objs):
        d = np.linalg.norm(embs - embs[i], axis=1)
        nov = float(d.sum() / max(1, len(d) - 1))
        out.append(Objective(obj.values + (w * nov,)))
    return out
