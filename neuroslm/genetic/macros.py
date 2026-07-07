# -*- coding: utf-8 -*-
"""Macros / ADFs — reusable NGL sub-programs the search composes.

A ``Macro`` is a named sub-program with formal input registers ``i0, i1, …`` and
an output register. A parent program invokes it with a single ``call``
instruction; ``expand_macros`` inlines the body — formal inputs remapped to the
call's actual registers, every internal register renamed to a fresh one, the
macro output routed to the call's ``out``. Nested macros expand recursively with
a cycle guard.

This is the abstraction lever: instead of re-deriving primitives every
generation, the GA can drop a whole ``call adam_update`` or ``call divisive_gain``
as one gene and build higher-order algorithms from chunks.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List

from neuroslm.genetic.language import Instruction, Program


@dataclass
class Macro:
    name: str
    body: Program
    n_inputs: int
    doc: str = ""


class MacroLibrary:
    def __init__(self, macros: List[Macro] = None):
        self._by_name: Dict[str, Macro] = {}
        for m in (macros or []):
            self._by_name[m.name] = m

    def add(self, macro: Macro) -> None:
        self._by_name[macro.name] = macro

    def get(self, name: str) -> Macro:
        if name not in self._by_name:
            raise KeyError(f"unknown macro {name!r}")
        return self._by_name[name]

    def has(self, name: str) -> bool:
        return name in self._by_name

    def names(self) -> List[str]:
        return sorted(self._by_name)

    def macros(self) -> List[Macro]:
        return [self._by_name[n] for n in self.names()]

    def __len__(self) -> int:
        return len(self._by_name)


def _max_index(program: Program, prefix: str) -> int:
    hi = -1
    regs = [program.out_reg]
    for ins in program.instructions:
        regs.append(ins.out)
        regs.extend(ins.ins)
    for r in regs:
        if r and r[0] == prefix:
            try:
                hi = max(hi, int(r[1:]))
            except ValueError:
                pass
    return hi


def expand_macros(program: Program, library: MacroLibrary,
                  _stack: List[str] = None) -> Program:
    """Inline every ``call`` in ``program`` (recursively) into a flat program."""
    if not any(i.op == "call" for i in program.instructions):
        return program
    _stack = _stack or []

    t_ctr = [_max_index(program, "t") + 1]
    s_ctr = [_max_index(program, "s") + 1]

    def fresh_t() -> str:
        r = f"t{t_ctr[0]}"
        t_ctr[0] += 1
        return r

    def fresh_s() -> str:
        r = f"s{s_ctr[0]}"
        s_ctr[0] += 1
        return r

    new_instrs: List[Instruction] = []
    for ins in program.instructions:
        if ins.op != "call":
            new_instrs.append(ins)
            continue
        if ins.macro in _stack:
            raise ValueError(f"macro cycle detected: {' → '.join(_stack + [ins.macro])}")
        macro = library.get(ins.macro)
        body = expand_macros(macro.body, library, _stack + [ins.macro])
        written = {b.out for b in body.instructions}

        remap: Dict[str, str] = {}
        for k in range(macro.n_inputs):
            actual = ins.ins[k] if k < len(ins.ins) else "t0"
            formal = f"i{k}"
            if formal in written:
                # body reassigns this input → isolate it in a fresh local so the
                # caller's actual register is never clobbered (copy-in semantics)
                local = fresh_t()
                new_instrs.append(Instruction("cscale", local, (actual,), const=1.0))
                remap[formal] = local
            else:
                remap[formal] = actual  # read-only input → alias directly

        def rr(reg: str) -> str:
            if reg in remap:
                return remap[reg]
            if reg and reg[0] == "s":
                remap[reg] = fresh_s()
            else:
                remap[reg] = fresh_t()
            return remap[reg]

        out_src = body.out_reg
        if out_src.startswith("i"):
            # macro returns an input directly → emit an identity copy to the call's out
            new_instrs.append(Instruction("cscale", ins.out, (rr(out_src),), const=1.0))
        else:
            remap[out_src] = ins.out

        for b in body.instructions:
            new_ins = tuple(rr(r) for r in b.ins)
            new_out = rr(b.out)
            new_instrs.append(Instruction(b.op, new_out, new_ins, b.const, b.config, b.macro))

    return Program(
        instructions=new_instrs,
        n_scalar=max(program.n_scalar, s_ctr[0] + 1),
        n_tensor=max(program.n_tensor, t_ctr[0] + 4),
        out_reg=program.out_reg,
        meta=dict(program.meta),
    )


# ---------------------------------------------------------------------------
# A default library of reusable building blocks.
# ---------------------------------------------------------------------------
def _macro(name, instrs, n_inputs, out_reg, doc=""):
    body = Program(instrs, n_scalar=6, n_tensor=16, out_reg=out_reg)
    return Macro(name=name, body=body, n_inputs=n_inputs, doc=doc)


def default_macro_library() -> MacroLibrary:
    """Reusable neural/optimizer building blocks the search can graft in."""
    return MacroLibrary([
        # divisive normalization: g = x / (rms(x) + eps)  (Carandini & Heeger)
        _macro("divisive_norm",
               [
                   Instruction("rms", "t2", ("i0",)),
                   Instruction("cscale", "t3", ("t2",), const=1.0),
                   Instruction("div", "o0", ("i0", "t3")),
               ],
               n_inputs=1, out_reg="o0", doc="divisive normalization"),
        # bounded gain: g = tanh(x) — homeostatic saturation
        _macro("bounded_gain",
               [Instruction("tanh", "o0", ("i0",))],
               n_inputs=1, out_reg="o0", doc="saturating gain"),
        # sign-momentum core (Lion-like): sign(0.9*a + 0.1*b)
        _macro("sign_interp",
               [
                   Instruction("cscale", "t2", ("i0",), const=0.9),
                   Instruction("cscale", "t3", ("i1",), const=0.1),
                   Instruction("add", "t4", ("t2", "t3")),
                   Instruction("sign", "o0", ("t4",)),
               ],
               n_inputs=2, out_reg="o0", doc="sign of an EMA interpolation"),
        # rms-scaled gradient (RMSProp-ish normaliser): g / (sqrt(g^2)+eps)
        _macro("rms_scale",
               [
                   Instruction("square", "t2", ("i0",)),
                   Instruction("sqrt", "t3", ("t2",)),
                   Instruction("div", "o0", ("i0", "t3")),
               ],
               n_inputs=1, out_reg="o0", doc="per-coordinate rms scaling"),
    ])
