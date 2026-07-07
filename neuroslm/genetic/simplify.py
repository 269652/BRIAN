# -*- coding: utf-8 -*-
"""NGL simplification discovery — find shorter, equivalent programs.

Two layers, cheap to expensive:

1. ``dead_code_eliminate`` — exact backward liveness. An instruction is kept only
   if its output register is read (directly or transitively) before ``out_reg``
   is produced. Because registers are reused, liveness is computed by walking the
   instruction list *backwards*. This preserves behaviour exactly — no probing.

2. ``simplify`` — a peephole + shrink superoptimizer. It applies algebraic
   identities (add-0, sub-0, mul/scale-1, neg-neg, transpose-transpose) and then
   greedily tries deleting each remaining instruction, keeping any edit that
   leaves the input→output map unchanged on a batch of random probes. This is how
   the search "discovers simplifications" of an arbitrary program (including an
   architecture lowered via ``compile_arch``).
"""
from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np
import torch

from neuroslm.genetic.language import Instruction, Memory, Program, REGISTRY


# ---------------------------------------------------------------------------
# Behavioural equivalence on random probes.
# ---------------------------------------------------------------------------
def _probe_inputs(rng: np.random.Generator, shape=(3, 4)):
    inputs = {}
    # grad/param-style inputs plus a couple of spare tensor registers
    for reg in ("t0", "t1"):
        inputs[reg] = torch.tensor(rng.standard_normal(shape), dtype=torch.float32)
    return inputs


def programs_equivalent(a: Program, b: Program, n_probes: int = 8,
                        seed: int = 0, atol: float = 1e-5, probes=None) -> bool:
    """True if a and b map the same probe inputs to the same output.

    ``probes`` — an optional list of ``{register: tensor}`` dicts. When given (e.g.
    shape-correct params+inputs for a compiled architecture, via
    ``compile_arch.make_probes``), equivalence is checked against those real values
    instead of the generic scalar probes.
    """
    if probes is not None:
        cases = probes
    else:
        rng = np.random.default_rng(seed)
        cases = [_probe_inputs(rng) for _ in range(n_probes)]
    for inp in cases:
        ma = Memory(a.n_scalar, a.n_tensor)
        mb = Memory(b.n_scalar, b.n_tensor)
        for reg, val in inp.items():
            ma.write(reg, val)
            mb.write(reg, val.clone())
        a.execute(ma)
        b.execute(mb)
        oa = ma.read(a.out_reg)
        ob = mb.read(b.out_reg)
        if oa.shape != ob.shape:
            return False
        if not torch.allclose(oa, ob, atol=atol, rtol=1e-4):
            return False
    return True


# ---------------------------------------------------------------------------
# Exact dead-code elimination (backward liveness).
# ---------------------------------------------------------------------------
def dead_code_eliminate(program: Program) -> Program:
    live = {program.out_reg}
    kept_rev: List[Instruction] = []
    for ins in reversed(program.instructions):
        if ins.out in live:
            kept_rev.append(ins)
            # this instruction defines ins.out; after removing it from the live
            # set, its inputs become live. (out is re-defined here, so earlier
            # writes to the same register are dead unless read by these inputs.)
            live.discard(ins.out)
            for r in ins.ins:
                live.add(r)
        # else: dead — its output is never consumed downstream
    kept = list(reversed(kept_rev))
    return Program(
        instructions=kept,
        n_scalar=program.n_scalar,
        n_tensor=program.n_tensor,
        out_reg=program.out_reg,
        meta=dict(program.meta),
    )


# ---------------------------------------------------------------------------
# Peephole algebraic identities.
# ---------------------------------------------------------------------------
def _is_const(ins: Instruction, value: float, tol=1e-12) -> bool:
    return ins.op == "const" and ins.const is not None and abs(ins.const - value) < tol


def _peephole(program: Program) -> Program:
    """Rewrite obvious identities into register aliases, then DCE.

    We resolve identities by rewriting *consumers*: if instruction I is proven to
    be an identity on register R (e.g. ``add(R, zero)``), every later read of I's
    output register is redirected to R until I's output is next redefined.
    """
    instrs = list(program.instructions)
    # map: which registers currently hold a known-zero / known-one constant
    const_val: Dict[str, float] = {}
    alias: Dict[str, str] = {}  # out_reg → source_reg (identity results)

    def resolve(r: str) -> str:
        seen = set()
        while r in alias and r not in seen:
            seen.add(r)
            r = alias[r]
        return r

    new: List[Instruction] = []
    for ins in instrs:
        ins_in = tuple(resolve(r) for r in ins.ins)
        op = ins.op
        identity_src: Optional[str] = None

        if op == "add" and len(ins_in) == 2:
            if const_val.get(ins_in[1]) == 0.0:
                identity_src = ins_in[0]
            elif const_val.get(ins_in[0]) == 0.0:
                identity_src = ins_in[1]
        elif op == "sub" and len(ins_in) == 2 and const_val.get(ins_in[1]) == 0.0:
            identity_src = ins_in[0]
        elif op == "mul" and len(ins_in) == 2:
            if const_val.get(ins_in[1]) == 1.0:
                identity_src = ins_in[0]
            elif const_val.get(ins_in[0]) == 1.0:
                identity_src = ins_in[1]
        elif op == "cscale" and ins.const is not None and abs(ins.const - 1.0) < 1e-12:
            identity_src = ins_in[0]
        elif op in ("neg", "transpose") and len(ins_in) == 1:
            # double application cancels: find the producer of ins_in[0]
            prod = _last_producer(new, ins_in[0])
            if prod is not None and prod.op == op:
                identity_src = resolve(prod.ins[0])

        # bookkeeping: this instruction redefines ins.out, clearing old facts
        const_val.pop(ins.out, None)
        alias.pop(ins.out, None)

        if identity_src is not None:
            alias[ins.out] = identity_src
            # drop the instruction (its output is now an alias); keep a no-op-free
            # stream. Downstream reads resolve through `alias`.
            continue

        emitted = Instruction(ins.op, ins.out, ins_in, ins.const, ins.config)
        new.append(emitted)
        if op == "const" and ins.const is not None:
            const_val[ins.out] = float(ins.const)

    out_reg = resolve(program.out_reg)
    prog = Program(new, program.n_scalar, program.n_tensor, out_reg, dict(program.meta))
    return dead_code_eliminate(prog)


def _last_producer(instrs: List[Instruction], reg: str) -> Optional[Instruction]:
    for ins in reversed(instrs):
        if ins.out == reg:
            return ins
    return None


# ---------------------------------------------------------------------------
# Shrink search — try deleting each instruction, keep if behaviour preserved.
# ---------------------------------------------------------------------------
def _try_delete_pass(program: Program, n_probes: int, seed: int, probes=None) -> Program:
    best = program
    i = 0
    while i < len(best.instructions):
        candidate_instrs = best.instructions[:i] + best.instructions[i + 1:]
        candidate = Program(candidate_instrs, best.n_scalar, best.n_tensor,
                            best.out_reg, dict(best.meta))
        if programs_equivalent(best, candidate, n_probes=n_probes, seed=seed, probes=probes):
            best = candidate  # accept the deletion, re-check from same index
        else:
            i += 1
    return best


def simplify(program: Program, n_probes: int = 8, seed: int = 0,
             max_rounds: int = 4, return_stats: bool = False, probes=None):
    """Return a shorter program behaviourally equivalent to ``program``.

    Pipeline: exact DCE → peephole identities → verified algebra → verified
    try-delete, iterated to a fixpoint (bounded by ``max_rounds``). ``probes`` (if
    given) makes every verification use real shape-correct values.
    """
    # lazy import: rewrite.py imports this module, so avoid a top-level cycle
    from neuroslm.genetic.rewrite import algebraic_simplify

    before = len(program.instructions)
    cur = dead_code_eliminate(program)
    for _ in range(max_rounds):
        n0 = len(cur.instructions)
        cur = _peephole(cur)
        cur = dead_code_eliminate(cur)
        cur = algebraic_simplify(cur, n_probes=n_probes, seed=seed, probes=probes)
        cur = _try_delete_pass(cur, n_probes=n_probes, seed=seed, probes=probes)
        cur = dead_code_eliminate(cur)
        if len(cur.instructions) == n0:
            break
    after = len(cur.instructions)
    if return_stats:
        return cur, {"before": before, "after": after, "removed": before - after}
    return cur
