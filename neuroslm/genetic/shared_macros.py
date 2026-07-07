# -*- coding: utf-8 -*-
"""Factor shared subexpressions into reusable macros — reuse everywhere, once.

When a computation is common to several mechanics (the search keeps
re-discovering ``sqrt(square(g))``, ``rms(x)``, a QK-norm chain …) it should live
in exactly one place — a ``Macro`` — and every mechanic that needs it should
``call`` it. This is CSE lifted from *inside one program* to *across the whole
model's mechanic set*: the improvement discovered in one algorithm is reused in
all algorithms that share the subexpression, not just where it was found.

The rewrite is verified: for each mechanic we re-expand the ``call`` and check
probe-equivalence against the original before accepting it. Anything unsafe to
factor (a mechanic that re-writes a register — optimizer state buffers — or a
subexpression whose intermediate is read elsewhere) is left untouched.

`promote_modulation` handles the other half: a modulation validated through
training/ablation is stamped as the new *reference* implementation.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from neuroslm.genetic.language import Instruction, Program
from neuroslm.genetic.macros import Macro, MacroLibrary, expand_macros
from neuroslm.genetic.simplify import programs_equivalent


# ---------------------------------------------------------------------------
# Subexpression keys (instruction-fold — consistent across mechanics).
# ---------------------------------------------------------------------------
def _reg_keys(prog: Program) -> Tuple[Dict[str, str], Dict[str, int], bool]:
    """Canonical expr-key of each register's value + single-assignment check.

    Returns ``(key_by_reg, root_idx_by_key, single_assignment)``. A register
    written more than once (optimizer state) flips ``single_assignment`` False —
    the caller then declines to factor that mechanic.
    """
    key: Dict[str, str] = {}
    root_idx: Dict[str, int] = {}
    writes: Dict[str, int] = {}
    single = True
    for i, ins in enumerate(prog.instructions):
        if ins.op == "call":
            single = False    # opaque; do not factor across calls
        writes[ins.out] = writes.get(ins.out, 0) + 1
        if writes[ins.out] > 1:
            single = False
        parts = [key.get(r, r) for r in ins.ins]
        inner = ",".join(parts)
        cfg = f",c={ins.const}" if ins.const is not None else ""
        mac = f",@{ins.macro}" if ins.macro else ""
        k = f"{ins.op}({inner}{cfg}{mac})"
        key[ins.out] = k
        root_idx[k] = i
    return key, root_idx, single


def _op_count(expr_key: str) -> int:
    """Number of ops in a subexpression key (depth-agnostic '(' count)."""
    return expr_key.count("(")


# ---------------------------------------------------------------------------
# Cone extraction inside one mechanic.
# ---------------------------------------------------------------------------
def _cone(prog: Program, root_reg: str) -> Optional[List[int]]:
    """Instruction indices that (transitively) produce ``root_reg``, or None."""
    produced = {}
    for i, ins in enumerate(prog.instructions):
        produced.setdefault(ins.out, i)
    if root_reg not in produced:
        return None
    seen = set()
    stack = [root_reg]
    while stack:
        r = stack.pop()
        idx = produced.get(r)
        if idx is None or idx in seen:
            continue
        seen.add(idx)
        for a in prog.instructions[idx].ins:
            stack.append(a)
    return sorted(seen)


def _external_inputs(prog: Program, cone_idx: List[int]) -> List[str]:
    """Registers read by the cone but produced outside it (macro formals)."""
    inside = set(cone_idx)
    produced_in_cone = {prog.instructions[i].out for i in cone_idx}
    ext: List[str] = []
    for i in cone_idx:
        for r in prog.instructions[i].ins:
            if r not in produced_in_cone and r not in ext:
                ext.append(r)
    return ext


def _internal_reused_outside(prog: Program, cone_idx: List[int], root_reg: str) -> bool:
    """True if a cone-internal (non-root) value is read outside the cone."""
    inside = set(cone_idx)
    produced_in_cone = {prog.instructions[i].out for i in cone_idx}
    for i, ins in enumerate(prog.instructions):
        if i in inside:
            continue
        for r in ins.ins:
            if r in produced_in_cone and r != root_reg:
                return True
    return False


def _build_macro(prog: Program, cone_idx: List[int], root_reg: str,
                 ext_inputs: List[str], name: str) -> Macro:
    """Materialize the cone as a macro body (ext inputs → i0.., root → o0)."""
    remap: Dict[str, str] = {r: f"i{k}" for k, r in enumerate(ext_inputs)}
    tctr = [200]
    for i in cone_idx:
        out = prog.instructions[i].out
        if out == root_reg:
            remap[out] = "o0"
        elif out not in remap:
            remap[out] = f"t{tctr[0]}"
            tctr[0] += 1

    def rr(r: str) -> str:
        return remap.get(r, r)

    body_instrs = []
    for i in cone_idx:
        ins = prog.instructions[i]
        body_instrs.append(Instruction(
            ins.op, rr(ins.out), tuple(rr(r) for r in ins.ins),
            ins.const, ins.config, ins.macro))
    body = Program(body_instrs, n_scalar=8, n_tensor=max(16, tctr[0] + 4), out_reg="o0")
    return Macro(name=name, body=body, n_inputs=len(ext_inputs),
                 doc=f"shared subexpression ({len(cone_idx)} ops)")


def _rewrite_with_call(prog: Program, cone_idx: List[int], root_reg: str,
                       ext_inputs: List[str], macro_name: str) -> Program:
    """Replace the cone with a single ``call macro(ext…) → root_reg``."""
    drop = set(cone_idx)
    root_pos = max(cone_idx)
    new_instrs: List[Instruction] = []
    for i, ins in enumerate(prog.instructions):
        if i in drop:
            if i == root_pos:
                new_instrs.append(Instruction("call", root_reg,
                                              tuple(ext_inputs), macro=macro_name))
            continue
        new_instrs.append(ins)
    return Program(new_instrs, n_scalar=prog.n_scalar, n_tensor=prog.n_tensor,
                   out_reg=prog.out_reg, meta=dict(prog.meta))


# ---------------------------------------------------------------------------
# Driver.
# ---------------------------------------------------------------------------
@dataclass
class ExtractionResult:
    library: MacroLibrary
    mechanics: Dict[str, Program]
    extracted: List[dict] = field(default_factory=list)


def extract_shared_as_macros(mechanics: Dict[str, Program], min_ops: int = 2,
                             n_probes: int = 12, seed: int = 0) -> ExtractionResult:
    """Factor multi-op subexpressions shared by ≥2 mechanics into macros.

    Returns the new ``MacroLibrary`` and the mechanics rewritten to ``call`` the
    macros. Every rewrite is probe-verified; unsafe or unverifiable ones are left
    as the original program.
    """
    # per-mechanic canonical keys (skip mechanics that aren't single-assignment)
    keyed: Dict[str, Tuple[Dict[str, str], Dict[str, int]]] = {}
    for name, prog in mechanics.items():
        kbr, root, single = _reg_keys(prog)
        if single:
            keyed[name] = (kbr, root)

    # count in how many mechanics each subexpression key appears
    occur: Dict[str, set] = {}
    for name, (kbr, root) in keyed.items():
        for k in set(kbr.values()):
            if _op_count(k) >= min_ops:
                occur.setdefault(k, set()).add(name)
    shared = {k: names for k, names in occur.items() if len(names) >= 2}
    # prefer the largest (most ops) subexpressions first; deterministic order
    ordered = sorted(shared.keys(), key=lambda k: (-_op_count(k), k))

    library = MacroLibrary()
    result: Dict[str, Program] = {n: p for n, p in mechanics.items()}
    extracted: List[dict] = []

    for k in ordered:
        names = sorted(shared[k])
        macro_name = f"shared_{len(library) + 1}"
        made_macro: Optional[Macro] = None
        applied: List[str] = []
        for name in names:
            prog = result[name]
            kbr, root = _reg_keys(prog)[0], _reg_keys(prog)[1]
            if k not in root:
                continue                     # already factored away in this prog
            root_reg = None
            # find the register whose value-key == k
            for reg, kk in kbr.items():
                if kk == k:
                    root_reg = reg
                    break
            if root_reg is None:
                continue
            cone_idx = _cone(prog, root_reg)
            if not cone_idx or len(cone_idx) < min_ops:
                continue
            if _internal_reused_outside(prog, cone_idx, root_reg):
                continue
            ext = _external_inputs(prog, cone_idx)
            if made_macro is None:
                made_macro = _build_macro(prog, cone_idx, root_reg, ext, macro_name)
            rewritten = _rewrite_with_call(prog, cone_idx, root_reg, ext, macro_name)
            lib = MacroLibrary(list(library.macros()) + [made_macro])
            flat = expand_macros(rewritten, lib)
            if programs_equivalent(prog, flat, n_probes=n_probes, seed=seed + 1):
                result[name] = rewritten
                applied.append(name)
        if made_macro is not None and len(applied) >= 2:
            library.add(made_macro)
            extracted.append({"macro": macro_name, "expr": k,
                              "ops": _op_count(k), "mechanics": applied})
        elif applied:
            # only one mechanic could safely take it → not worth a shared macro;
            # revert those rewrites so we never ship a call to an unregistered macro
            for name in applied:
                result[name] = mechanics[name]

    return ExtractionResult(library=library, mechanics=result, extracted=extracted)


# ---------------------------------------------------------------------------
# Reference promotion.
# ---------------------------------------------------------------------------
_REFERENCE_KEY = "reference"


def promote_modulation(store, name: str) -> None:
    """Stamp a validated modulation as the new reference implementation."""
    rec = store.get(name)
    rec.metrics[_REFERENCE_KEY] = 1.0
    store.save(rec)


def is_reference(rec) -> bool:
    return float(rec.metrics.get(_REFERENCE_KEY, 0.0)) >= 1.0
