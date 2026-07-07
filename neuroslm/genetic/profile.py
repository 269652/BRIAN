# -*- coding: utf-8 -*-
"""Flow + compute heat profiler for NGL execution.

Runs a program and records, per instruction:
- **flow** — the norm of the output tensor (how much information/activation
  magnitude passes through this node), and
- **compute** — an estimated FLOP count for the op given its input/output shapes.

From these you get the map the evolution search wants: where information
concentrates, where heavy computation sits, and — most usefully — the
**low-hanging fruit** (high flow, low compute), the edges cheapest to reshape for
the most effect. This is also the substrate for visualization and for the
graph-theoretic topology analysis (``topology.py``).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List

import torch

from neuroslm.genetic.language import Memory, Program


@dataclass
class NodeStat:
    index: int
    op: str
    out_reg: str
    ins: tuple
    flow: float          # ||output||
    numel: int           # output element count
    flops: float         # estimated FLOPs
    macro: str = ""


def _est_flops(op: str, args, out) -> float:
    """Rough FLOP estimate — enough to rank ops by compute weight."""
    def numel(t):
        return t.numel() if torch.is_tensor(t) else 1
    o = numel(out)
    if op in ("matmul",):
        a, b = args[0], args[1]
        if torch.is_tensor(a) and torch.is_tensor(b) and a.ndim >= 2 and b.ndim >= 2 \
                and a.shape[-1] == b.shape[-2]:
            batch = a.numel() // (a.shape[-2] * a.shape[-1])
            return 2.0 * batch * a.shape[-2] * a.shape[-1] * b.shape[-1]
        return float(o)
    if op == "linear":
        x, w = args[0], args[1]
        if torch.is_tensor(x) and torch.is_tensor(w) and w.ndim == 2:
            rows = x.numel() // x.shape[-1] if x.ndim >= 1 else 1
            return 2.0 * rows * w.shape[0] * w.shape[1]
        return float(o)
    if op == "outer":
        return float(o)
    if op in ("softmax", "softmax_last"):
        return 3.0 * float(max(numel(args[0]), o))
    if op in ("exp", "log", "sqrt", "tanh", "sigmoid", "gelu", "silu",
              "rms", "norm", "l2norm_last", "rmsnorm", "layernorm"):
        return 2.0 * float(max(numel(args[0]) if args else o, o))
    # plain elementwise / reductions
    return float(max((numel(a) for a in args), default=0) or o)


@dataclass
class ExecutionProfile:
    nodes: List[NodeStat] = field(default_factory=list)
    out_reg: str = ""

    def total_flops(self) -> float:
        return float(sum(n.flops for n in self.nodes))

    def heavy_compute(self, top: int = 5) -> List[NodeStat]:
        return sorted(self.nodes, key=lambda n: -n.flops)[:top]

    def hot_flow(self, top: int = 5) -> List[NodeStat]:
        return sorted(self.nodes, key=lambda n: -n.flow)[:top]

    def low_hanging(self, top: int = 5) -> List[NodeStat]:
        """High information flow per unit compute — cheap edges with big effect."""
        return sorted(self.nodes, key=lambda n: -(n.flow / (n.flops + 1.0)))[:top]

    def to_dict(self) -> dict:
        producer: Dict[str, int] = {}
        edges = []
        for n in self.nodes:
            for r in n.ins:
                if r in producer:
                    edges.append({"src": producer[r], "dst": n.index, "reg": r})
            producer[n.out_reg] = n.index
        return {
            "out_reg": self.out_reg,
            "total_flops": self.total_flops(),
            "nodes": [
                {"index": n.index, "op": n.op, "out": n.out_reg, "ins": list(n.ins),
                 "flow": n.flow, "numel": n.numel, "flops": n.flops, "macro": n.macro}
                for n in self.nodes
            ],
            "edges": edges,
        }


def profile_program(program: Program, inputs: Dict[str, torch.Tensor],
                    device=None) -> ExecutionProfile:
    """Execute ``program`` on ``inputs`` and return its flow+compute profile."""
    prof = ExecutionProfile(out_reg=program.out_reg)
    counter = [0]

    def rec(ins, args, out):
        flow = float(out.norm()) if torch.is_tensor(out) and out.numel() else 0.0
        prof.nodes.append(NodeStat(
            index=counter[0], op=ins.op, out_reg=ins.out, ins=tuple(ins.ins),
            flow=flow, numel=int(out.numel()) if torch.is_tensor(out) else 1,
            flops=_est_flops(ins.op, args, out), macro=ins.macro,
        ))
        counter[0] += 1

    mem = Memory(program.n_scalar, program.n_tensor,
                 device=device if device is not None else None)
    for reg, val in inputs.items():
        mem.write(reg, val)
    program.execute(mem, recorder=rec)
    return prof
