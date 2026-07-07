# -*- coding: utf-8 -*-
"""NGL — the Neuro-Genetic Language: a typed register machine for ML algorithms.

NGL is the fourth DSL layer (see ``docs/dsl_subsystem_roadmap.md`` §NGL). Unlike
the straight-line SSA DAG of ``nn_lang.py``, an NGL ``Program`` has *persistent
state* and *control*, so it can encode update rules (SGD, Adam, Lion), learning
rules and gradient/flow-modulation policies — the algorithms that architecture
DSLs cannot express. It is the substrate on which "search the language space to
discover a novel ML mechanism" is tractable (AutoML-Zero / the Lion discovery
search exactly this program space).

Design invariants
-----------------
1. **Total execution.** Every op is total: division is eps-guarded, ``sqrt``/
   ``log`` fold through ``abs``, ``matmul`` on incompatible shapes falls back to
   an elementwise/identity result. A random program *never* raises — that is
   what makes blind mutation/crossover safe.
2. **Closed-form semantics.** Each op has one meaning, listed in ``REGISTRY``.
   The registry *is* the spec.
3. **Deterministic.** Execution and ``semantic_vector`` are pure functions of
   the program + inputs.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Dict, Optional, Tuple

import numpy as np
import torch

_EPS = 1e-12
# Hard cap on tensor size produced by any op. Keeps execution not just total
# (never raises) but memory-safe: a random program that chains `outer`/`matmul`
# would otherwise blow up geometrically (144-vec → 144×144 → 20736² ≈ 1.7 GB and
# OOM). Ops that would exceed the cap fall back to an identity/elementwise result.
_MAX_ELEMS = 1 << 16  # 65536


# ---------------------------------------------------------------------------
# Op registry — each op is a total function over torch tensors / 0-d scalars.
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class OpSpec:
    name: str
    n_in: int
    family: str
    fn: Callable[..., torch.Tensor]
    uses_const: bool = False
    uses_config: bool = False       # takes scalar-config kwargs (e.g. attention n_heads)
    config_names: tuple = ()        # ordered names of the config kwargs


def _t(x) -> torch.Tensor:
    if isinstance(x, torch.Tensor):
        return x
    return torch.as_tensor(float(x))


def _safe_div(a, b):
    a, b = _t(a), _t(b)
    return a / (b + torch.sign(b) * _EPS + _EPS)


def _matmul(a, b):
    a, b = _t(a), _t(b)
    if a.ndim >= 2 and b.ndim >= 2 and a.shape[-1] == b.shape[-2]:
        out_elems = a.numel() // a.shape[-1] * b.shape[-1]
        if out_elems <= _MAX_ELEMS:
            return a @ b
        return a  # too large → identity fallback
    # incompatible → total fallback: broadcast-multiply where possible, else a
    try:
        return a * b
    except RuntimeError:
        return a


def _outer(a, b):
    a, b = _t(a).reshape(-1), _t(b).reshape(-1)
    if a.numel() * b.numel() > _MAX_ELEMS:
        return a  # too large → identity fallback (keeps execution memory-safe)
    return torch.outer(a, b)


def _transpose(a):
    a = _t(a)
    if a.ndim < 2:
        return a
    return a.transpose(-1, -2)


def _select(cond, a, b):
    cond, a, b = _t(cond), _t(a), _t(b)
    try:
        return torch.where(cond > 0, a, b)
    except RuntimeError:
        return torch.where(cond > 0, a, b * torch.ones_like(a))


def _broadcast_binary(f):
    def g(a, b):
        a, b = _t(a), _t(b)
        try:
            return f(a, b)
        except RuntimeError:
            # shape-incompatible → reduce b to a scalar and retry
            return f(a, b.mean())
    return g


def _make_registry() -> Dict[str, OpSpec]:
    R: Dict[str, OpSpec] = {}

    def reg(name, n_in, family, fn, uses_const=False, uses_config=False,
            config_names=()):
        R[name] = OpSpec(name, n_in, family, fn, uses_const, uses_config,
                         tuple(config_names))

    # arithmetic
    reg("add", 2, "arith", _broadcast_binary(lambda a, b: a + b))
    reg("sub", 2, "arith", _broadcast_binary(lambda a, b: a - b))
    reg("mul", 2, "arith", _broadcast_binary(lambda a, b: a * b))
    reg("div", 2, "arith", _safe_div)
    reg("neg", 1, "arith", lambda a: -_t(a))
    reg("abs", 1, "arith", lambda a: _t(a).abs())
    reg("sign", 1, "arith", lambda a: _t(a).sign())
    reg("square", 1, "arith", lambda a: _t(a) * _t(a))
    reg("sqrt", 1, "arith", lambda a: _t(a).abs().add(_EPS).sqrt())
    reg("exp", 1, "arith", lambda a: _t(a).clamp(-30, 30).exp())
    reg("log", 1, "arith", lambda a: _t(a).abs().add(_EPS).log())
    reg("clip", 1, "arith", lambda a, c: _t(a).clamp(-abs(c) - _EPS, abs(c) + _EPS), True)
    reg("cscale", 1, "arith", lambda a, c: _t(a) * c, True)

    # reductions / norms → 0-d scalars
    reg("mean", 1, "reduce", lambda a: _t(a).mean())
    reg("sum", 1, "reduce", lambda a: _t(a).sum())
    reg("norm", 1, "reduce", lambda a: _t(a).norm())
    reg("rms", 1, "reduce", lambda a: _t(a).pow(2).mean().add(_EPS).sqrt())
    reg("max_r", 1, "reduce", lambda a: _t(a).max())
    reg("min_r", 1, "reduce", lambda a: _t(a).min())

    # control / compare
    reg("gt", 2, "control", _broadcast_binary(lambda a, b: (a > b).float()))
    reg("select", 3, "control", _select)
    reg("min", 2, "control", _broadcast_binary(torch.minimum))
    reg("max", 2, "control", _broadcast_binary(torch.maximum))

    # nonlinear
    reg("tanh", 1, "nonlin", lambda a: _t(a).tanh())
    reg("sigmoid", 1, "nonlin", lambda a: _t(a).sigmoid())
    reg("relu", 1, "nonlin", lambda a: _t(a).relu())
    reg("silu", 1, "nonlin", lambda a: torch.nn.functional.silu(_t(a)))
    reg("softmax", 1, "nonlin", lambda a: torch.softmax(_t(a).reshape(-1), dim=0).reshape(_t(a).shape))

    # linear algebra
    reg("matmul", 2, "linalg", _matmul)
    reg("transpose", 1, "linalg", _transpose)
    reg("outer", 2, "linalg", _outer)

    # composite neural-network ops — delegate to the canonical nn_ops atoms so
    # NGL spans the *architecture* grammar too, not just the update-rule grammar.
    # (These make `arch → NGL` a near 1:1 lowering; see genetic/compile_arch.py.)
    from neuroslm.dsl import nn_ops as _nnops

    reg("linear", 2, "nn", lambda x, w: _nnops.linear(x, w))
    reg("rmsnorm", 2, "nn", lambda x, g: _nnops.rmsnorm(x, g))
    reg("layernorm", 3, "nn", lambda x, g, b: _nnops.layernorm(x, g, b))
    reg("swiglu", 4, "nn", lambda x, w1, w2, w3: _nnops.swiglu(x, w1, w2, w3))
    reg("gelu", 1, "nn", lambda x: _nnops.gelu(x))
    reg("embedding", 2, "nn", lambda ids, table: _nnops.embedding(ids.long(), table))

    # config-carrying mechanics: attention mixes tensor args with scalar config.
    def _attn(x, q, kv, o, n_heads=1, n_kv_heads=1, max_ctx=1, rope_base=10000.0):
        return _nnops.causal_self_attention(
            x, q, kv, o, int(n_heads), int(n_kv_heads), int(max_ctx), float(rope_base))

    reg("causal_self_attention", 4, "nn", _attn, uses_config=True,
        config_names=("n_heads", "n_kv_heads", "max_ctx", "rope_base"))

    # constant loader
    reg("const", 0, "const", lambda c: _t(c), True)

    return R


REGISTRY: Dict[str, OpSpec] = _make_registry()
OP_FAMILIES: Tuple[str, ...] = tuple(sorted({s.family for s in REGISTRY.values()}))


# ---------------------------------------------------------------------------
# Memory — the typed register file.
# ---------------------------------------------------------------------------
class Memory:
    """Register file with scalar bank ``s*`` and tensor bank ``t*``.

    Reading an unwritten register returns a zero (0-d scalar tensor) so
    execution is total.
    """

    __slots__ = ("n_scalar", "n_tensor", "_regs", "device")

    def __init__(self, n_scalar: int, n_tensor: int, device=None):
        self.n_scalar = n_scalar
        self.n_tensor = n_tensor
        self._regs: Dict[str, torch.Tensor] = {}
        self.device = torch.device(device) if device is not None else torch.device("cpu")

    def _valid(self, reg: str) -> bool:
        if not reg or reg[0] not in ("s", "t"):
            return False
        try:
            idx = int(reg[1:])
        except ValueError:
            return False
        limit = self.n_scalar if reg[0] == "s" else self.n_tensor
        return 0 <= idx < limit

    def read(self, reg: str) -> torch.Tensor:
        v = self._regs.get(reg)
        if v is None:
            return torch.zeros((), device=self.device)
        return v

    def write(self, reg: str, value) -> None:
        if not self._valid(reg):
            return  # writes to out-of-range registers are silently dropped
        v = _t(value)
        # track the live device: a real (cuda) tensor sets the memory's device so
        # constants + unwritten reads follow it (avoids cuda/cpu op mismatches).
        if isinstance(value, torch.Tensor) and value.device.type != "cpu":
            self.device = value.device
        self._regs[reg] = v

    def clone(self) -> "Memory":
        m = Memory(self.n_scalar, self.n_tensor, device=self.device)
        m._regs = {k: v for k, v in self._regs.items()}
        return m


# ---------------------------------------------------------------------------
# Instruction + Program.
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class Instruction:
    op: str
    out: str
    ins: Tuple[str, ...] = ()
    const: Optional[float] = None
    config: Tuple[Tuple[str, float], ...] = ()   # scalar-config kwargs (attention …)
    macro: str = ""                              # macro name when op == "call"

    def __post_init__(self):
        if self.op == "call":
            if not self.macro:
                raise ValueError("call instruction requires a macro name")
            return
        if self.op not in REGISTRY:
            raise ValueError(f"unknown op {self.op!r}")


@dataclass
class Program:
    instructions: list
    n_scalar: int
    n_tensor: int
    out_reg: str
    meta: dict = field(default_factory=dict)
    library: object = None   # optional MacroLibrary → `call` ops auto-flatten

    def execute(self, memory: Memory) -> Memory:
        prog = self
        if self.library is not None and any(i.op == "call" for i in self.instructions):
            from neuroslm.genetic.macros import expand_macros
            prog = expand_macros(self, self.library)
        return prog._execute_flat(memory)

    def _execute_flat(self, memory: Memory) -> Memory:
        dev = memory.device
        for ins in self.instructions:
            spec = REGISTRY.get(ins.op)
            if spec is None:
                # an unexpanded `call` (no library) — keep execution total
                continue
            args = [memory.read(r) for r in ins.ins]
            # align every tensor arg to the memory device so a `const`/eps scalar
            # (created on cpu) never mismatches a cuda operand — that mismatch
            # would raise inside the op and silently fall back, corrupting math.
            if dev.type != "cpu":
                args = [a.to(dev) if isinstance(a, torch.Tensor) and a.device != dev else a
                        for a in args]
            kwargs = {}
            if spec.uses_config:
                kwargs = {k: v for k, v in ins.config}
            elif spec.uses_const:
                c = 0.0 if ins.const is None else float(ins.const)
                args.append(c)
            try:
                out = spec.fn(*args, **kwargs)
            except Exception:
                # last-resort totality guard: identity of first input or zero
                out = args[0] if args else torch.zeros(())
            if not isinstance(out, torch.Tensor):
                out = _t(out)
            if out.numel() > _MAX_ELEMS:
                # defence-in-depth memory cap: collapse an oversized result to a
                # scalar mean so no register can hold a runaway tensor.
                out = out.mean()
            if not torch.isfinite(out).all():
                out = torch.nan_to_num(out, nan=0.0, posinf=1e6, neginf=-1e6)
            if dev.type != "cpu" and out.device != dev:
                out = out.to(dev)   # keep every register value on the live device
            memory.write(ins.out, out)
        return memory

    # -- semantic embedding -------------------------------------------------
    def semantic_vector(self) -> np.ndarray:
        """Fixed-length embedding: op-family histogram + structural features."""
        fam_counts = {f: 0.0 for f in OP_FAMILIES}
        op_counts: Dict[str, float] = {}
        outs = set()
        reuse = 0
        for ins in self.instructions:
            fam = REGISTRY[ins.op].family if ins.op in REGISTRY else "macro"
            fam_counts[fam] = fam_counts.get(fam, 0.0) + 1.0
            key = ins.op if ins.op != "call" else f"call:{ins.macro}"
            op_counts[key] = op_counts.get(key, 0.0) + 1.0
            for r in ins.ins:
                if r in outs:
                    reuse += 1
            outs.add(ins.out)
        n = max(1, len(self.instructions))
        fam_vec = np.array([fam_counts[f] / n for f in OP_FAMILIES], dtype=np.float64)
        # per-op histogram in a stable op order
        op_order = sorted(REGISTRY.keys())
        op_vec = np.array([op_counts.get(o, 0.0) / n for o in op_order], dtype=np.float64)
        struct = np.array(
            [
                len(self.instructions),
                self.n_scalar,
                self.n_tensor,
                len(outs),
                reuse / n,
            ],
            dtype=np.float64,
        )
        return np.concatenate([fam_vec, op_vec, struct])

    def copy(self) -> "Program":
        return Program(
            instructions=list(self.instructions),
            n_scalar=self.n_scalar,
            n_tensor=self.n_tensor,
            out_reg=self.out_reg,
            meta=dict(self.meta),
            library=self.library,
        )

    def to_source(self) -> str:
        lines = []
        for ins in self.instructions:
            if ins.op == "call":
                lines.append(f"{ins.out} = call {ins.macro}(" + ", ".join(ins.ins) + ")")
                continue
            rhs = ins.op + "(" + ", ".join(ins.ins)
            if REGISTRY[ins.op].uses_const:
                rhs += (", " if ins.ins else "") + f"c={ins.const}"
            if ins.config:
                rhs += (", " if ins.ins else "") + ", ".join(
                    f"{k}={v}" for k, v in ins.config)
            rhs += ")"
            lines.append(f"{ins.out} = {rhs}")
        lines.append(f"return {self.out_reg}")
        return "\n".join(lines)
