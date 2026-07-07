# -*- coding: utf-8 -*-
"""Compile a DSL architecture (nn_lang forward graph) into an NGL program.

The architecture DSLs (Layer A `nn_lang`) describe a straight-line SSA tensor
DAG. NGL is a register machine — a DAG lowers to it directly: each SSA value gets
a register, each op/binop becomes one instruction, parameters are pre-bound
tensor registers. Once an arch is an NGL program you can *simplify* it
(`simplify.py`), *mutate* it, and run discovery on it (`evolve.py`).

Equivalence is the contract: `run_compiled(compile_layer_to_ngl(src), params, x)`
equals the compiled `nn_lang` module's forward, bit-for-bit, because both call
the same `nn_ops` atoms (now also registered as NGL ops).

Scalar-config ops (attention's `n_heads`, …) mix ints with tensors and are not
yet lowerable — those raise `UnsupportedLowering` rather than miscompile.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, List

import torch

from neuroslm.dsl.nn_lang import (
    parse_layer,
    Num,
    Name,
    Call,
    BinOp,
    LayerDef,
)
from neuroslm.genetic.language import Instruction, Memory, Program, REGISTRY

_BINOP = {"+": "add", "-": "sub", "*": "mul", "/": "div"}


class UnsupportedLowering(Exception):
    """The forward graph uses a construct NGL can't lower yet (e.g. an op that
    mixes scalar config with tensor args)."""


@dataclass
class CompiledArch:
    program: Program
    param_regs: Dict[str, str]     # param name → register
    input_regs: Dict[str, str]     # forward-arg name → register
    layerdef: LayerDef


def compile_layer_to_ngl(source: str) -> CompiledArch:
    ld = parse_layer(source)
    if ld.sublayers:
        raise UnsupportedLowering(
            f"layer {ld.name!r} has sublayers {list(ld.sublayers)}; "
            "compose them at the nn_lang level before lowering")

    reg_of: Dict[str, str] = {}
    tcount = [0]

    def new_t() -> str:
        r = f"t{tcount[0]}"
        tcount[0] += 1
        return r

    # inputs first, then params — these are the pre-bound registers
    for a in ld.fwd_args:
        reg_of[a] = new_t()
    for p in ld.params:
        reg_of[p.name] = new_t()

    # layer args that are pure scalar config (e.g. n_heads) — remembered so we
    # can give a precise error if a lowered op needs one as a tensor.
    scalar_args = set(ld.args) - set(reg_of)

    instrs: List[Instruction] = []
    out_reg = None

    def lower(node) -> str:
        if isinstance(node, Num):
            r = new_t()
            instrs.append(Instruction("const", r, (), const=float(node.value)))
            return r
        if isinstance(node, Name):
            if node.id in reg_of:
                return reg_of[node.id]
            if node.id in scalar_args:
                raise UnsupportedLowering(
                    f"forward references scalar-config arg {node.id!r} as a value; "
                    "scalar-config lowering is not supported")
            raise UnsupportedLowering(f"unbound name {node.id!r} in forward")
        if isinstance(node, BinOp):
            l = lower(node.left)
            r = lower(node.right)
            op = _BINOP.get(node.op)
            if op is None:
                raise UnsupportedLowering(f"binop {node.op!r} not supported")
            out = new_t()
            instrs.append(Instruction(op, out, (l, r)))
            return out
        if isinstance(node, Call):
            if node.fn not in REGISTRY:
                raise UnsupportedLowering(
                    f"op {node.fn!r} is not an NGL op (add it to the registry "
                    "with a total-semantics impl, or decompose it)")
            spec = REGISTRY[node.fn]
            # any Num / scalar-config arg means this op takes non-tensor config
            for a in node.args:
                if isinstance(a, Name) and a.id in scalar_args:
                    raise UnsupportedLowering(
                        f"op {node.fn!r} takes scalar-config arg {a.id!r}; "
                        "not lowerable yet")
            arg_regs = tuple(lower(a) for a in node.args)
            if len(arg_regs) != spec.n_in:
                raise UnsupportedLowering(
                    f"op {node.fn!r} expects {spec.n_in} args, got {len(arg_regs)}")
            out = new_t()
            instrs.append(Instruction(node.fn, out, arg_regs))
            return out
        raise UnsupportedLowering(f"cannot lower node {type(node).__name__}")

    for stmt in ld.body:
        r = lower(stmt.expr)
        if stmt.is_return:
            out_reg = r
        else:
            reg_of[stmt.target] = r

    if out_reg is None:
        raise UnsupportedLowering(f"layer {ld.name!r} forward has no return")

    program = Program(
        instructions=instrs,
        n_scalar=8,
        n_tensor=tcount[0] + 4,
        out_reg=out_reg,
        meta={"name": ld.name, "source": "arch"},
    )
    return CompiledArch(
        program=program,
        param_regs={p.name: reg_of[p.name] for p in ld.params},
        input_regs={a: reg_of[a] for a in ld.fwd_args},
        layerdef=ld,
    )


def run_compiled(compiled: CompiledArch,
                 params: Dict[str, torch.Tensor],
                 inputs: Dict[str, torch.Tensor]) -> torch.Tensor:
    """Bind params + inputs into a Memory and execute the compiled program."""
    mem = Memory(compiled.program.n_scalar, compiled.program.n_tensor)
    for name, reg in compiled.param_regs.items():
        if name not in params:
            raise KeyError(f"missing param tensor {name!r}")
        mem.write(reg, params[name])
    for name, reg in compiled.input_regs.items():
        if name not in inputs:
            raise KeyError(f"missing input tensor {name!r}")
        mem.write(reg, inputs[name])
    compiled.program.execute(mem)
    return mem.read(compiled.program.out_reg)
