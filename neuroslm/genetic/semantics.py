# -*- coding: utf-8 -*-
"""Static semantic analysis / abstract interpretation of NGL programs.

An NGL program is a straight-line register machine; its *meaning* — is the
output bounded? does it normalize? does it carry state across steps? what role
does it play in a model? — is recoverable without ever running it, by
propagating a small lattice of value facts through each op's transfer function.
That is exactly abstract interpretation: the concrete domain (tensors) is
replaced by an abstract domain (`AbstractValue`) and each op gets an abstract
transfer function that over-approximates its concrete behaviour.

Why this earns its place: the discovery / CSE loop needs to know *when two
mechanics are interchangeable* before it dares swap one for a cheaper
equivalent. Two programs that occupy the same semantic role with compatible
abstract outputs (both bounded, both normalizing, same mixing structure) are
substitution candidates. `interchangeable(a, b)` is that gate; `describe()` is
the human-readable projection of the same summary.

The abstract lattice per register (all booleans, ``False`` = "unknown/top"):

* ``bounded``    — magnitude is provably in a bounded range
* ``nonneg``     — provably ``>= 0``
* ``normalized`` — unit-scaled (post row-softmax / L2 / RMS normalization)
* ``sign_only``  — value in ``{-1, 0, +1}``
* ``mixes``      — derived via cross-element mixing (matmul / reduce / softmax …)
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Dict, List, Tuple

from neuroslm.genetic.language import REGISTRY, Program


# ---------------------------------------------------------------------------
# Abstract domain.
# ---------------------------------------------------------------------------
@dataclass
class AbstractValue:
    bounded: bool = False
    nonneg: bool = False
    normalized: bool = False
    sign_only: bool = False
    mixes: bool = False

    def to_dict(self) -> dict:
        return asdict(self)


# A register read before it is ever written is an *input* (the default abstract
# value — magnitude unknown, no mixing has happened yet).
def _input_value() -> AbstractValue:
    return AbstractValue()


# Ops that combine information across elements (a matmul row touches a whole
# column; a reduction touches the whole tensor; a row-softmax touches the axis).
# `elementwise` is the negation of "uses any of these".
_MIXING_OPS = frozenset({
    "matmul", "outer", "linear", "rmsnorm", "layernorm", "swiglu",
    "softmax", "softmax_last", "softpick_last", "l2norm_last",
    "causal_self_attention", "embedding", "mean", "sum", "norm", "rms",
    "max_r", "min_r",
})

# Ops whose *output* is a normalized (unit-scale) quantity.
_NORMALIZING_OPS = frozenset({
    "rmsnorm", "layernorm", "l2norm_last", "softmax", "softmax_last",
    "softpick_last",
})

_OPT_NAMES = frozenset({"sgd", "momentum", "rmsprop", "adam", "lion"})


# ---------------------------------------------------------------------------
# Per-op abstract transfer functions.  Each takes the abstract values of its
# inputs and returns the abstract value of its output.
# ---------------------------------------------------------------------------
def _transfer(op: str, ins: List[AbstractValue]) -> AbstractValue:
    a = ins[0] if ins else _input_value()
    b = ins[1] if len(ins) > 1 else _input_value()
    any_mix = any(v.mixes for v in ins)

    # --- bounded nonlinearities -----------------------------------------
    if op == "tanh":
        return AbstractValue(bounded=True, mixes=any_mix)
    if op == "sigmoid":
        return AbstractValue(bounded=True, nonneg=True, mixes=any_mix)
    if op in ("softmax", "softmax_last"):
        # rows sum to 1 → bounded, non-negative, normalized; mixes over the axis
        return AbstractValue(bounded=True, nonneg=True, normalized=True, mixes=True)
    if op == "softpick_last":
        # like softmax but rectified/not-sum-to-one: still in [0,1], non-negative,
        # a normalizer over the axis; permits true zeros
        return AbstractValue(bounded=True, nonneg=True, normalized=True, mixes=True)
    if op == "relu":
        return AbstractValue(nonneg=True, mixes=any_mix)
    if op in ("silu", "gelu"):
        return AbstractValue(mixes=any_mix)
    if op == "sign":
        return AbstractValue(bounded=True, sign_only=True, mixes=a.mixes)
    if op in ("clip",):
        return AbstractValue(bounded=True, mixes=a.mixes)

    # --- normalizers -----------------------------------------------------
    if op == "l2norm_last":
        # each component of a unit vector is in [-1, 1] → bounded + normalized
        return AbstractValue(bounded=True, normalized=True, mixes=True)
    if op in ("rmsnorm", "layernorm"):
        # unit-variance features scaled by a (learned, unbounded) gain
        return AbstractValue(normalized=True, mixes=True)

    # --- reductions → scalars -------------------------------------------
    if op in ("norm", "rms"):
        return AbstractValue(nonneg=True, mixes=True)
    if op in ("mean", "sum", "max_r", "min_r"):
        return AbstractValue(mixes=True)

    # --- pure reshape (propagates its input's facts) --------------------
    if op == "transpose":
        return AbstractValue(bounded=a.bounded, nonneg=a.nonneg,
                             normalized=a.normalized, sign_only=a.sign_only,
                             mixes=a.mixes)
    if op == "causal_mask":
        # masks future positions with a large negative → no longer bounded
        return AbstractValue(mixes=a.mixes)

    # --- arithmetic ------------------------------------------------------
    if op == "add":
        return AbstractValue(bounded=a.bounded and b.bounded,
                             nonneg=a.nonneg and b.nonneg, mixes=any_mix)
    if op == "sub":
        return AbstractValue(bounded=a.bounded and b.bounded, mixes=any_mix)
    if op == "mul":
        return AbstractValue(bounded=a.bounded and b.bounded,
                             nonneg=a.nonneg and b.nonneg, mixes=any_mix)
    if op == "div":
        return AbstractValue(nonneg=a.nonneg and b.nonneg, mixes=any_mix)
    if op == "neg":
        return AbstractValue(bounded=a.bounded, mixes=a.mixes)
    if op == "abs":
        return AbstractValue(bounded=a.bounded, nonneg=True, mixes=a.mixes)
    if op == "square":
        return AbstractValue(bounded=a.bounded, nonneg=True, mixes=a.mixes)
    if op == "sqrt":
        return AbstractValue(bounded=a.bounded, nonneg=True, mixes=a.mixes)
    if op in ("exp", "log"):
        return AbstractValue(nonneg=(op == "exp"), mixes=a.mixes)
    if op == "cscale":
        return AbstractValue(bounded=a.bounded, mixes=a.mixes)
    if op == "const":
        return AbstractValue(bounded=True)

    # --- control / compare ----------------------------------------------
    if op == "gt":
        return AbstractValue(bounded=True, nonneg=True, mixes=any_mix)
    if op == "select":
        c = ins[2] if len(ins) > 2 else _input_value()
        return AbstractValue(bounded=b.bounded and c.bounded,
                             nonneg=b.nonneg and c.nonneg, mixes=any_mix)
    if op in ("min", "max"):
        return AbstractValue(bounded=a.bounded and b.bounded, mixes=any_mix)

    # --- linear algebra + composite nn (cross-element mixing) -----------
    if op in ("matmul", "outer", "linear", "swiglu", "embedding"):
        return AbstractValue(mixes=True)

    # unknown op → top (conservative: unbounded, assume it mixes)
    return AbstractValue(mixes=True)


# ---------------------------------------------------------------------------
# The summary.
# ---------------------------------------------------------------------------
@dataclass
class SemanticSummary:
    role: str
    bounded: bool
    normalizing: bool
    elementwise: bool
    sign_based: bool
    stateful: bool
    op_families: Tuple[str, ...]
    n_instructions: int
    output: AbstractValue
    inputs: Tuple[str, ...] = ()
    state: Tuple[str, ...] = ()
    notes: Tuple[str, ...] = ()

    def to_dict(self) -> dict:
        d = asdict(self)
        d["output"] = self.output.to_dict()
        return d

    def describe(self) -> str:
        return _describe_summary(self)


# ---------------------------------------------------------------------------
# Analysis driver.
# ---------------------------------------------------------------------------
def analyze(program: Program) -> SemanticSummary:
    prog = program
    if getattr(program, "library", None) is not None and any(
            i.op == "call" for i in program.instructions):
        from neuroslm.genetic.macros import expand_macros
        prog = expand_macros(program, program.library)

    env: Dict[str, AbstractValue] = {}
    written: set = set()
    inputs: List[str] = []       # regs read before first write (order-preserving)
    ever_written: set = set()

    for ins in prog.instructions:
        if ins.op == "call":     # unexpanded macro (no library) — treat as top
            env[ins.out] = AbstractValue(mixes=True)
            written.add(ins.out)
            ever_written.add(ins.out)
            continue
        in_vals: List[AbstractValue] = []
        for r in ins.ins:
            if r not in written and r not in inputs:
                inputs.append(r)
            in_vals.append(env.get(r, _input_value()))
        env[ins.out] = _transfer(ins.op, in_vals)
        written.add(ins.out)
        ever_written.add(ins.out)

    ops = [i.op for i in prog.instructions]
    families = tuple(sorted({REGISTRY[o].family for o in ops if o in REGISTRY}))
    out_val = env.get(prog.out_reg, _input_value())

    # a register read-before-write AND written somewhere = a persistent buffer
    state = tuple(r for r in inputs if r in ever_written)
    stateful = len(state) > 0
    sign_based = "sign" in ops
    elementwise = not any(o in _MIXING_OPS for o in ops)
    normalizing = out_val.normalized or any(o in _NORMALIZING_OPS for o in ops)

    role = _classify_role(prog, ops, out_val, stateful, normalizing, elementwise)
    notes = _notes(role, out_val, stateful, normalizing, sign_based, elementwise)

    return SemanticSummary(
        role=role,
        bounded=out_val.bounded,
        normalizing=normalizing,
        elementwise=elementwise,
        sign_based=sign_based,
        stateful=stateful,
        op_families=families,
        n_instructions=len(prog.instructions),
        output=out_val,
        inputs=tuple(inputs),
        state=state,
        notes=notes,
    )


def _has_attention(ops: List[str]) -> bool:
    if "causal_self_attention" in ops:
        return True
    # a row-normalizer (softmax or softpick) over a score matmul = attention
    has_row_norm = "softmax_last" in ops or "softpick_last" in ops
    return has_row_norm and "matmul" in ops


def _classify_role(prog, ops, out_val, stateful, normalizing, elementwise) -> str:
    name = str(prog.meta.get("name", "")) if prog.meta else ""
    meta_role = prog.meta.get("role") if prog.meta else None
    if meta_role:
        return str(meta_role)
    if name in _OPT_NAMES:
        return "optimizer_update"
    if _has_attention(ops):
        return "attention"
    if stateful:
        # a read-modify-write buffer over a grad/param input is an update rule
        return "optimizer_update"
    if normalizing and not any(o in ("matmul", "outer", "linear") for o in ops):
        return "normalization"
    has_nonlin = any(o in ("tanh", "sigmoid", "relu", "silu", "gelu") for o in ops)
    if has_nonlin and ("mul" in ops or "select" in ops) and elementwise:
        return "gating"
    if elementwise and has_nonlin:
        return "activation"
    if any(o in ("matmul", "linear", "outer", "swiglu") for o in ops):
        return "projection"
    return "generic"


def _notes(role, out_val, stateful, normalizing, sign_based, elementwise) -> Tuple[str, ...]:
    n = []
    if out_val.bounded:
        n.append("output is magnitude-bounded (safe to feed a saturating stage)")
    if normalizing:
        n.append("normalizes its input (unit-scale output)")
    if sign_based:
        n.append("sign-based update (scale-free, Lion-like)")
    if stateful:
        n.append("carries state across steps (persistent buffer)")
    if elementwise:
        n.append("pointwise — no cross-element mixing")
    else:
        n.append("mixes information across elements")
    return tuple(n)


# ---------------------------------------------------------------------------
# Human-readable projection.
# ---------------------------------------------------------------------------
_ROLE_USE = {
    "activation": "Use as a pointwise nonlinearity between linear stages.",
    "normalization": "Use to stabilize scale before attention/MLP (pre-norm).",
    "attention": "Use to mix tokens along the sequence axis.",
    "projection": "Use as a linear feature map / channel mixer.",
    "gating": "Use to modulate a signal by a learned gate.",
    "optimizer_update": "Use as a per-parameter update rule in the training loop.",
    "generic": "General-purpose computation.",
}


def _describe_summary(s: SemanticSummary) -> str:
    bits = [f"Role: {s.role}. {_ROLE_USE.get(s.role, '')}"]
    trait = []
    trait.append("bounded output" if s.bounded else "unbounded output")
    if s.normalizing:
        trait.append("normalizing")
    trait.append("elementwise" if s.elementwise else "element-mixing")
    if s.sign_based:
        trait.append("sign-based")
    if s.stateful:
        trait.append(f"stateful (buffers: {', '.join(s.state) or '—'})")
    bits.append("Traits: " + ", ".join(trait) + ".")
    bits.append(f"Families: {', '.join(s.op_families) or '—'}; "
                f"{s.n_instructions} instruction(s).")
    if s.notes:
        bits.append("Notes: " + "; ".join(s.notes) + ".")
    return " ".join(bits)


def describe(program_or_summary) -> str:
    if isinstance(program_or_summary, SemanticSummary):
        return _describe_summary(program_or_summary)
    return _describe_summary(analyze(program_or_summary))


# ---------------------------------------------------------------------------
# Interchangeability — the substitution gate for CSE / mechanic reuse.
# ---------------------------------------------------------------------------
def interchangeable(a: Program, b: Program) -> bool:
    """Two mechanics may substitute if they play the same role and expose the
    same abstract output contract (boundedness, normalization, mixing).

    This is deliberately stricter than "same role": swapping a bounded
    activation for an unbounded one changes what a downstream saturating stage
    sees, so they are *not* interchangeable even though both are activations.
    """
    sa, sb = analyze(a), analyze(b)
    if sa.role != sb.role:
        return False
    oa, ob = sa.output, sb.output
    return (oa.bounded == ob.bounded
            and oa.normalized == ob.normalized
            and oa.mixes == ob.mixes
            and sa.sign_based == sb.sign_based
            and sa.stateful == sb.stateful)
