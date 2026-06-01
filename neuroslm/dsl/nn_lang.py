# -*- coding: utf-8 -*-
"""NeuroTensor language — `layer`/`model` parser → nn.Module (Phase N3).

Compiles a tensor-graph layer written in DSL text into an executable
nn.Module whose forward calls the exact-match op atoms in `nn_ops`.

Grammar (minimal but sufficient for TransformerBlock / LanguageCortex):

    layer <Name>(<arg>, ...) {
        param <name>: (<shape expr>) init=<spec>
        ...
        forward(<arg>, ...) {
            <var> = <expr>
            ...
            return <expr>
        }
    }

Expressions: numbers, identifiers, function calls (op atoms or sublayers),
and +, -, *, / with the usual precedence. Identifiers that are forward
locals stay bare; everything else (params, layer args) resolves to
`self.<name>`. Op-atom calls lower to `nn_ops.<fn>(...)`.

Design note (per docs/dsl_nn_language.md): the parse tree is a typed DAG
— this is deliberately the structure the future hypershape compiler
(N10) consumes, so we keep statements in SSA-ish form and avoid opaque
control flow.
"""
from __future__ import annotations
import ast
import re
from dataclasses import dataclass
from typing import Dict, List, Optional

import torch
import torch.nn as nn

from neuroslm.dsl import nn_ops
from neuroslm.dsl.novel_topology import (
    make_grid_positions, make_episodic_memory, make_surprise_head,
)


# Op atoms callable from the DSL (introspected from nn_ops)
_OP_NAMES = {
    name for name in dir(nn_ops)
    if callable(getattr(nn_ops, name)) and not name.startswith("_")
}


# ── Parameter init ─────────────────────────────────────────────────────

def _alloc(init: str, shape) -> torch.Tensor:
    """Allocate a parameter tensor. Init only needs the right *shape* for
    exact-match tests (weights get synced); the distributions are sensible
    defaults for real training.

    Supports parameterized inits: `normal(σ)`, `constant(v)`, `uniform(a,b)`.
    """
    if not isinstance(shape, tuple):
        shape = (shape,)

    name, args = _parse_init_spec(init)

    if name == "ones":
        return torch.ones(*shape)
    if name == "zeros":
        return torch.zeros(*shape)
    if name == "xavier":
        t = torch.empty(*shape)
        if t.dim() >= 2:
            nn.init.xavier_uniform_(t)
        else:
            nn.init.normal_(t, std=0.02)
        return t
    if name == "kaiming":
        t = torch.empty(*shape)
        if t.dim() >= 2:
            nn.init.kaiming_uniform_(t, a=5 ** 0.5)
        else:
            nn.init.normal_(t, std=0.02)
        return t
    if name == "normal":
        std = args[0] if args else 0.02
        return torch.randn(*shape) * float(std)
    if name == "constant":
        val = args[0] if args else 0.0
        return torch.full(shape, float(val))
    if name == "uniform":
        a, b = (args + [0.0, 1.0])[:2]
        return torch.empty(*shape).uniform_(float(a), float(b))
    raise ValueError(f"unknown init spec {init!r}")


def _parse_init_spec(spec: str):
    """`normal(0.01)` → ('normal', [0.01]); `zeros` → ('zeros', [])."""
    spec = spec.strip()
    if "(" not in spec:
        return spec, []
    name, rest = spec.split("(", 1)
    rest = rest.rstrip(")")
    args = [a.strip() for a in rest.split(",") if a.strip()]
    return name.strip(), args


# ── AST ────────────────────────────────────────────────────────────────

@dataclass
class ParamDecl:
    name: str
    shape_src: str   # raw "(D, D)" — eval'd against kwargs at init
    init: str


@dataclass
class StateDecl:
    """Persistent buffer state — register_buffer'd, not a parameter.

    Read into a local at the forward prelude, written back at the postlude
    so reassignments in the body persist across calls. The state-machine
    semantics needed by NeurotransmitterSystem / trophic / vesicle pools.
    """
    name: str
    shape_src: str
    init: str


@dataclass
class Stmt:
    target: Optional[str]   # None for a return
    expr: "Expr"
    is_return: bool = False


@dataclass
class LayerDef:
    name: str
    args: List[str]
    params: List[ParamDecl]
    states: List[StateDecl]        # persistent buffer state (NT, trophic, ...)
    sublayers: Dict[str, str]      # name → layer-type (Phase N4 composition)
    fwd_args: List[str]
    body: List[Stmt]


# Expr nodes
@dataclass
class Num:    value: str
@dataclass
class Name:   id: str
@dataclass
class Call:   fn: str; args: List["Expr"]
@dataclass
class BinOp:  op: str; left: "Expr"; right: "Expr"


Expr = object  # one of the above


# ── Tokenizer for expressions ──────────────────────────────────────────

_TOK_RE = re.compile(r"\s*(?:(?P<num>\d+\.?\d*)|(?P<id>[A-Za-z_]\w*)|(?P<op>[()+\-*/,]))")


def _tokenize_expr(s: str) -> List[tuple]:
    toks, i = [], 0
    while i < len(s):
        m = _TOK_RE.match(s, i)
        if not m or m.end() == i:
            if s[i].isspace():
                i += 1
                continue
            raise ValueError(f"bad token in expr at {s[i:]!r}")
        i = m.end()
        if m.group("num"):
            toks.append(("num", m.group("num")))
        elif m.group("id"):
            toks.append(("id", m.group("id")))
        else:
            toks.append(("op", m.group("op")))
    toks.append(("end", ""))
    return toks


class _ExprParser:
    """Recursive-descent: arith → term (+,-) ; term → factor (*,/) ;
    factor → NUM | IDENT['(' args ')'] | '(' arith ')'."""
    def __init__(self, toks):
        self.toks = toks
        self.pos = 0

    def peek(self): return self.toks[self.pos]
    def advance(self):
        t = self.toks[self.pos]; self.pos += 1; return t

    def parse(self) -> Expr:
        e = self.arith()
        if self.peek()[0] != "end":
            raise ValueError(f"trailing tokens: {self.toks[self.pos:]}")
        return e

    def arith(self) -> Expr:
        node = self.term()
        while self.peek() == ("op", "+") or self.peek() == ("op", "-"):
            op = self.advance()[1]
            node = BinOp(op, node, self.term())
        return node

    def term(self) -> Expr:
        node = self.factor()
        while self.peek() == ("op", "*") or self.peek() == ("op", "/"):
            op = self.advance()[1]
            node = BinOp(op, node, self.factor())
        return node

    def factor(self) -> Expr:
        kind, val = self.advance()
        if kind == "num":
            return Num(val)
        if kind == "op" and val == "(":
            inner = self.arith()
            assert self.advance() == ("op", ")"), "expected )"
            return inner
        if kind == "id":
            if self.peek() == ("op", "("):
                self.advance()  # consume (
                args = []
                if self.peek() != ("op", ")"):
                    args.append(self.arith())
                    while self.peek() == ("op", ","):
                        self.advance()
                        args.append(self.arith())
                assert self.advance() == ("op", ")"), "expected ) after args"
                return Call(val, args)
            return Name(val)
        raise ValueError(f"unexpected token {(kind, val)}")


def _parse_expr(s: str) -> Expr:
    return _ExprParser(_tokenize_expr(s)).parse()


# ── Layer parser ───────────────────────────────────────────────────────

_LAYER_RE = re.compile(r"\b(?:layer|model)\s+(\w+)\s*\(([^)]*)\)\s*\{", re.S)
# init=name or init=name(args)
_PARAM_RE = re.compile(r"\bparam\s+(\w+)\s*:\s*(\([^)]*\))\s+init=(\w+(?:\([^)]*\))?)")
# state mirrors param syntax: `state name: (shape) init=spec`. Read at the
# forward prelude, written back at the postlude — persistent buffer.
_STATE_RE = re.compile(r"\bstate\s+(\w+)\s*:\s*(\([^)]*\))\s+init=(\w+(?:\([^)]*\))?)")
_SUBLAYER_RE = re.compile(r"\bsublayer\s+(\w+)\s*:\s*(\w+)")
_FORWARD_RE = re.compile(r"\bforward\s*\(([^)]*)\)\s*\{", re.S)


def _find_matching_brace(s: str, open_pos: int) -> int:
    depth, i = 1, open_pos + 1
    while i < len(s) and depth:
        if s[i] == "{": depth += 1
        elif s[i] == "}": depth -= 1
        i += 1
    return i - 1


def parse_layer(source: str) -> LayerDef:
    m = _LAYER_RE.search(source)
    if not m:
        raise ValueError("no `layer`/`model` declaration found")
    name = m.group(1)
    args = [a.strip() for a in m.group(2).split(",") if a.strip()]
    body_start = m.end() - 1
    body_end = _find_matching_brace(source, body_start)
    body_text = source[body_start + 1:body_end]

    params = [ParamDecl(pm.group(1), pm.group(2), pm.group(3))
              for pm in _PARAM_RE.finditer(body_text)]
    states = [StateDecl(sm.group(1), sm.group(2), sm.group(3))
              for sm in _STATE_RE.finditer(body_text)]
    sublayers = {sm.group(1): sm.group(2) for sm in _SUBLAYER_RE.finditer(body_text)}

    fm = _FORWARD_RE.search(body_text)
    if not fm:
        raise ValueError(f"layer {name!r} has no forward block")
    fwd_args = [a.strip() for a in fm.group(1).split(",") if a.strip()]
    f_start = fm.end() - 1
    f_end = _find_matching_brace(body_text, f_start)
    fwd_text = body_text[f_start + 1:f_end]

    body = _parse_forward_body(fwd_text)
    return LayerDef(name, args, params, states, sublayers, fwd_args, body)


def _parse_forward_body(text: str) -> List[Stmt]:
    """Parse the forward body into Stmts. Lines may continue across
    newlines as long as parentheses are open (enables multi-line calls)."""
    # Coalesce continuation lines by paren depth, then split on real ends.
    raw_lines = text.split("\n")
    logical: List[str] = []
    buf, depth = "", 0
    for raw in raw_lines:
        line = raw.strip().rstrip(";")
        if not line:
            if depth == 0 and buf:
                logical.append(buf); buf = ""
            continue
        buf = (buf + " " + line) if buf else line
        for ch in line:
            if ch == "(":
                depth += 1
            elif ch == ")":
                depth = max(0, depth - 1)
        if depth == 0:
            logical.append(buf); buf = ""
    if buf:
        logical.append(buf)

    stmts = []
    for line in logical:
        line = line.strip()
        if not line:
            continue
        if line.startswith("return "):
            stmts.append(Stmt(None, _parse_expr(line[len("return "):]), is_return=True))
        elif "=" in line:
            target, expr_src = line.split("=", 1)
            stmts.append(Stmt(target.strip(), _parse_expr(expr_src.strip())))
        else:
            raise ValueError(f"unparseable forward statement: {line!r}")
    return stmts


# ── Codegen → Python source → nn.Module ────────────────────────────────

def _lower_expr(node: Expr, locals_: set, sublayers: Dict[str, str]) -> str:
    if isinstance(node, Num):
        return node.value
    if isinstance(node, Name):
        return node.id if node.id in locals_ else f"self.{node.id}"
    if isinstance(node, BinOp):
        return f"({_lower_expr(node.left, locals_, sublayers)} {node.op} "\
               f"{_lower_expr(node.right, locals_, sublayers)})"
    if isinstance(node, Call):
        args = ", ".join(_lower_expr(a, locals_, sublayers) for a in node.args)
        if node.fn in _OP_NAMES:
            return f"nn_ops.{node.fn}({args})"
        if node.fn in sublayers:
            return f"self.{node.fn}({args})"
        raise ValueError(f"unknown op/sublayer {node.fn!r}")
    raise TypeError(f"cannot lower {type(node)}")


def generate_layer_source(ld: LayerDef) -> str:
    lines = [f"class {ld.name}(nn.Module):",
             "    def __init__(self, **kw):",
             "        super().__init__()",
             "        for _k, _v in kw.items():",
             "            setattr(self, _k, _v)"]
    for p in ld.params:
        # shape eval'd against kwargs; tuple-ify single dims
        lines.append(
            f"        _sh = {p.shape_src} if isinstance({p.shape_src}, tuple) "
            f"else ({p.shape_src},)"
        )
        # Re-eval shape with kw names in scope: emit a literal eval helper
        lines.append(
            f"        self.{p.name} = nn.Parameter("
            f"_alloc({p.init!r}, _evalshape({p.shape_src!r}, kw)))"
        )
    # State buffers — persistent, not learnable. register_buffer puts them
    # in named_buffers() (not named_parameters()), moves them across .to(),
    # and survives state_dict round-trips.
    for s in ld.states:
        lines.append(
            f"        self.register_buffer({s.name!r}, "
            f"_alloc({s.init!r}, _evalshape({s.shape_src!r}, kw)))"
        )
    # forward
    lines.append("")
    lines.append(f"    def forward(self, {', '.join(ld.fwd_args)}):")
    locals_ = set(ld.fwd_args)
    # State prelude — read each state buffer into a local so the body can
    # treat it like an ordinary variable (and rebind it).
    for s in ld.states:
        lines.append(f"        {s.name} = self.{s.name}")
        locals_.add(s.name)
    for st in ld.body:
        rhs = _lower_expr(st.expr, locals_, ld.sublayers)
        if st.is_return:
            # State postlude — copy locals back into the buffers *before*
            # returning. .detach() so we don't pin a grad graph to the
            # buffer across calls; torch.no_grad() so the in-place copy
            # doesn't error under autograd version-tracking.
            if ld.states:
                lines.append(f"        _ret = {rhs}")
                lines.append("        with torch.no_grad():")
                for s in ld.states:
                    lines.append(
                        f"            self.{s.name}.copy_({s.name}.detach())"
                    )
                lines.append("        return _ret")
            else:
                lines.append(f"        return {rhs}")
        else:
            lines.append(f"        {st.target} = {rhs}")
            locals_.add(st.target)
    return "\n".join(lines)


def _evalshape(shape_src: str, kw: Dict) -> tuple:
    """Eval a shape expression like '(D, D)' or '(Dkv, D)' against kwargs."""
    val = eval(shape_src, {"__builtins__": {}}, dict(kw))
    return val if isinstance(val, tuple) else (val,)


def compile_layer(source: str) -> type:
    """Parse + compile a layer/model DSL block to an nn.Module subclass."""
    ld = parse_layer(source)
    src = generate_layer_source(ld)
    # Strip the dead `_sh` scratch line we don't actually use
    src = "\n".join(l for l in src.split("\n") if not l.strip().startswith("_sh ="))
    ast.parse(src)  # validate
    ns = {"nn": nn, "torch": torch, "nn_ops": nn_ops,
          "_alloc": _alloc, "_evalshape": _evalshape}
    exec(compile(src, f"<nn_lang:{ld.name}>", "exec"), ns)
    return ns[ld.name]


# ── N4: stacked language model from DSL blocks ─────────────────────────

# The canonical transformer block, in DSL text. The whole LM is composed
# from this — stacking it, plus embedding + final-norm + lm_head — into a
# model whose forward is exact-match to a reference built from the same
# common.py primitives.
TRANSFORMER_BLOCK_DSL = '''
layer TransformerBlock(D, n_heads, n_kv_heads, max_ctx, H, Dkv) {
    param gamma1: (D,) init=ones
    param Wq: (D, D) init=xavier
    param Wkv: (Dkv, D) init=xavier
    param Wo: (D, D) init=xavier
    param gamma2: (D,) init=ones
    param w1: (H, D) init=xavier
    param w2: (H, D) init=xavier
    param w3: (D, H) init=xavier

    forward(x) {
        a = causal_self_attention(rmsnorm(x, gamma1), Wq, Wkv, Wo, n_heads, n_kv_heads, max_ctx)
        x = x + a
        m = swiglu(rmsnorm(x, gamma2), w1, w2, w3)
        return x + m
    }
}
'''


class DSLLanguageModel(nn.Module):
    """Embedding → N DSL TransformerBlocks → final RMSNorm → lm_head.

    The blocks are compiled from `TRANSFORMER_BLOCK_DSL` (the N3 path, so
    each block is provably exact-match to common.TransformerBlock). This
    class supplies only the composition: token embedding, the block stack,
    the final norm, and the output projection — all using the same N1 op
    atoms, so the whole model is bit-identical to a reference assembled
    from common.py.
    """
    def __init__(self, vocab: int, d_model: int, depth: int,
                 n_heads: int, max_ctx: int, n_kv_heads: Optional[int] = None):
        super().__init__()
        n_kv_heads = n_kv_heads or n_heads
        H = nn_ops.swiglu_hidden_dim(d_model)
        Dkv = 2 * n_kv_heads * (d_model // n_heads)

        BlockCls = compile_layer(TRANSFORMER_BLOCK_DSL)
        self.embed = nn.Parameter(_alloc("normal", (vocab, d_model)))
        self.blocks = nn.ModuleList([
            BlockCls(D=d_model, n_heads=n_heads, n_kv_heads=n_kv_heads,
                     max_ctx=max_ctx, H=H, Dkv=Dkv)
            for _ in range(depth)
        ])
        self.gamma_f = nn.Parameter(torch.ones(d_model))
        self.lm_head = nn.Parameter(_alloc("xavier", (vocab, d_model)))

    def forward(self, ids: torch.Tensor) -> torch.Tensor:
        h = nn_ops.embedding(ids, self.embed)
        # Capture per-block activations for the metric observers (Φ, λ₁,
        # ignition, oscillations, trophic). Detached so metrics never
        # perturb the training graph.
        self._layer_acts = []
        for blk in self.blocks:
            h = blk(h)
            self._layer_acts.append(h.detach())
        h = nn_ops.rmsnorm(h, self.gamma_f)
        return nn_ops.linear(h, self.lm_head)


def build_language_model(vocab: int, d_model: int, depth: int,
                         n_heads: int, max_ctx: int,
                         n_kv_heads: Optional[int] = None) -> DSLLanguageModel:
    """Construct a DSL-composed language model (embedding + stacked blocks
    + final norm + head)."""
    return DSLLanguageModel(vocab, d_model, depth, n_heads, max_ctx, n_kv_heads)


# ─── N8: full DSL LanguageCortex (interleaved pattern + adapters) ──────

# The Brain LanguageCortex (non-baseline) interleaves three block types:
#   Layer i%3 == 0  →  Standard (causal_self_attention)
#   Layer i%3 == 1  →  Differential attention
#   Layer i%3 == 2  →  MoD with differential attention
# A NeuralGeometryAdapter follows every block. Each block type is the
# DSL form proven bit-identical in test_dsl_blocks_equivalence.py.

_STD_BLOCK_DSL = '''
layer StandardBlock(D, n_heads, n_kv_heads, max_ctx, H, Dkv) {
    param gamma1: (D,) init=ones
    param Wq:     (D, D) init=xavier
    param Wkv:    (Dkv, D) init=xavier
    param Wo:     (D, D) init=xavier
    param gamma2: (D,) init=ones
    param w1:     (H, D) init=xavier
    param w2:     (H, D) init=xavier
    param w3:     (D, H) init=xavier
    forward(x) {
        a = causal_self_attention(rmsnorm(x, gamma1), Wq, Wkv, Wo, n_heads, n_kv_heads, max_ctx)
        x = x + a
        m = swiglu(rmsnorm(x, gamma2), w1, w2, w3)
        return x + m
    }
}
'''

# Stage 2 OOD push — Tonnetz toroidal attention. Same params as
# StandardBlock so weights are bit-compatible (lets you toggle the
# Tonnetz mask on/off without retraining from scratch). Only the
# attention CALL changes — additive toroidal mask composes with the
# causal constraint; non-harmonic distant positions are exponentially
# suppressed (bounds the convex hull of attention mass, cited as a
# hallucination biomarker).
_STD_TONNETZ_BLOCK_DSL = '''
layer StandardTonnetzBlock(D, n_heads, n_kv_heads, max_ctx, H, Dkv, tonnetz_period) {
    param gamma1: (D,) init=ones
    param Wq:     (D, D) init=xavier
    param Wkv:    (Dkv, D) init=xavier
    param Wo:     (D, D) init=xavier
    param gamma2: (D,) init=ones
    param w1:     (H, D) init=xavier
    param w2:     (H, D) init=xavier
    param w3:     (D, H) init=xavier
    forward(x) {
        a = causal_self_attention_tonnetz(rmsnorm(x, gamma1), Wq, Wkv, Wo, n_heads, n_kv_heads, max_ctx, tonnetz_period)
        x = x + a
        m = swiglu(rmsnorm(x, gamma2), w1, w2, w3)
        return x + m
    }
}
'''

_DIFF_BLOCK_DSL = '''
layer DiffBlock(D, n_heads, n_kv_heads, max_ctx, H, Dkv, head_dim) {
    param gamma1:      (D,) init=ones
    param Wq:          (D, D) init=xavier
    param Wkv:         (Dkv, D) init=xavier
    param Wo:          (D, D) init=xavier
    param lambda_init: (n_heads,) init=zeros
    param sub_norm:    (head_dim,) init=ones
    param gamma2:      (D,) init=ones
    param w1:          (H, D) init=xavier
    param w2:          (H, D) init=xavier
    param w3:          (D, H) init=xavier
    forward(x) {
        a = differential_attention(rmsnorm(x, gamma1), Wq, Wkv, Wo, lambda_init, sub_norm, n_heads, n_kv_heads, max_ctx)
        x = x + a
        m = swiglu(rmsnorm(x, gamma2), w1, w2, w3)
        return x + m
    }
}
'''

_MOD_BLOCK_DSL = '''
layer ModBlock(D, n_heads, n_kv_heads, max_ctx, H, Dkv, head_dim, R_hidden, capacity) {
    param router_w1:   (R_hidden, D)   init=zeros
    param router_b1:   (R_hidden,)     init=zeros
    param router_w2:   (1, R_hidden)   init=zeros
    param router_b2:   (1,)            init=zeros
    param gamma1:      (D,)            init=ones
    param Wq:          (D, D)          init=xavier
    param Wkv:         (Dkv, D)        init=xavier
    param Wo:          (D, D)          init=xavier
    param lambda_init: (n_heads,)      init=zeros
    param sub_norm:    (head_dim,)     init=ones
    param gamma2:      (D,)            init=ones
    param w1:          (H, D)          init=xavier
    param w2:          (H, D)          init=xavier
    param w3:          (D, H)          init=xavier
    forward(x) {
        return mod_block(x, router_w1, router_b1, router_w2, router_b2,
                          gamma1, Wq, Wkv, Wo, lambda_init, sub_norm,
                          gamma2, w1, w2, w3,
                          n_heads, n_kv_heads, max_ctx, capacity)
    }
}
'''

_ADAPTER_DSL = '''
layer NeuralGeometryAdapter(D, Dhyper, R) {
    param gamma:   (D,) init=ones
    param Wup:     (Dhyper, D) init=xavier
    param kern_a:  (Dhyper, R) init=normal(0.01)
    param kern_b:  (R, Dhyper) init=normal(0.01)
    param Wgate:   (Dhyper, Dhyper) init=xavier
    param bgate:   (Dhyper,) init=constant(-2.0)
    param Wdown:   (D, Dhyper) init=zeros
    forward(x) {
        h     = rmsnorm(x, gamma)
        z     = linear(h, Wup)
        k     = matmul(matmul(z, kern_a), kern_b)
        g     = sigmoid(linear(z, Wgate, bgate))
        z_new = silu(k) * g
        out   = linear(z_new, Wdown)
        return x + out
    }
}
'''


class DSLLanguageCortex(nn.Module):
    """Full LanguageCortex assembled from pure-DSL blocks (N8 keystone).

    Matches neuroslm.modules.language.LanguageCortex(baseline=False) on the
    LM-logits path: token embedding → for each layer (Standard/Diff/MoD
    interleaved by i%3) followed by a NeuralGeometryAdapter →
    final RMSNorm → lm_head. PCT / PredictiveDropout / CALM / attention
    pool / mid-trunk tap / motor_bias / NT / memory_kv all OFF (the
    baseline LM-logits path).

    All four block types are bit-identical to their Brain references
    (test_dsl_blocks_equivalence.py + test_nn_attention_equivalence.py).
    """
    def __init__(self, vocab: int, d_model: int, depth: int,
                 n_heads: int, max_ctx: int,
                 n_kv_heads: Optional[int] = None,
                 geometry_expansion: float = 2.0,
                 mod_capacity: float = 0.5,
                 dropout: float = 0.0,
                 pct_trunk: float = 0.0,
                 tonnetz_period: int = 0,
                 stochastic_depth: float = 0.0,
                 grid_positions=None,
                 episodic_memory=None,
                 surprise_head=None):
        super().__init__()
        n_kv_heads = n_kv_heads or n_heads
        head_dim = d_model // n_heads
        H = nn_ops.swiglu_hidden_dim(d_model)
        Dkv = 2 * n_kv_heads * head_dim
        Dhyper = int(d_model * geometry_expansion)
        R = max(8, Dhyper // 8)
        R_hidden = max(32, d_model // 8)
        # Stochastic depth: linearly increasing drop probability per block.
        # At training time, block i is skipped (identity) with prob
        # stochastic_depth * (i+1) / depth. At eval time, always active.
        self.stochastic_depth = stochastic_depth
        self._depth = depth
        # Embed + residual dropout for OOD regularization. Applied post-
        # embed and after each block's output so it touches every layer's
        # output without changing the bit-identical-to-Brain DSL block
        # internals (Brain itself uses dropout=0 on the baseline trunk;
        # this path is an OOD-targeted addition controlled by arch.neuro's
        # `training.dropout`).
        self._dropout_init = dropout
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

        # Pick the standard-block template: Tonnetz variant when the
        # arch enables a toroidal attention period, else the original.
        self.tonnetz_period = tonnetz_period
        if tonnetz_period > 0:
            Std = compile_layer(_STD_TONNETZ_BLOCK_DSL)
        else:
            Std = compile_layer(_STD_BLOCK_DSL)
        Diff = compile_layer(_DIFF_BLOCK_DSL)
        Mod = compile_layer(_MOD_BLOCK_DSL)
        Adp = compile_layer(_ADAPTER_DSL)

        self.embed = nn.Parameter(_alloc("normal", (vocab, d_model)))
        self.blocks = nn.ModuleList()
        self.adapters = nn.ModuleList()
        for i in range(depth):
            pattern = i % 3
            if pattern == 0:
                std_kwargs = dict(D=d_model, n_heads=n_heads,
                                  n_kv_heads=n_kv_heads, max_ctx=max_ctx,
                                  H=H, Dkv=Dkv)
                if tonnetz_period > 0:
                    std_kwargs["tonnetz_period"] = tonnetz_period
                self.blocks.append(Std(**std_kwargs))
            elif pattern == 1:
                self.blocks.append(Diff(D=d_model, n_heads=n_heads,
                                        n_kv_heads=n_kv_heads, max_ctx=max_ctx,
                                        H=H, Dkv=Dkv, head_dim=head_dim))
            else:
                self.blocks.append(Mod(D=d_model, n_heads=n_heads,
                                       n_kv_heads=n_kv_heads, max_ctx=max_ctx,
                                       H=H, Dkv=Dkv, head_dim=head_dim,
                                       R_hidden=R_hidden, capacity=mod_capacity))
            self.adapters.append(Adp(D=d_model, Dhyper=Dhyper, R=R))

        self.gamma_f = nn.Parameter(torch.ones(d_model))
        self.lm_head = nn.Parameter(_alloc("xavier", (vocab, d_model)))

        # Predictive-coding heads — Brain's per-layer-pair aux loss. Each
        # head: pred = Linear(d_model, hidden) → SiLU → Linear(hidden, d_model)
        # with both weights zero-init (residual identity start), bias on
        # first linear only. n_layers-1 heads total.
        pch_hidden = max(d_model // 4, 32)
        n_pch = max(0, depth - 1)
        self.pch_w1 = nn.ParameterList(
            [nn.Parameter(torch.zeros(pch_hidden, d_model)) for _ in range(n_pch)])
        self.pch_b1 = nn.ParameterList(
            [nn.Parameter(torch.zeros(pch_hidden)) for _ in range(n_pch)])
        self.pch_w2 = nn.ParameterList(
            [nn.Parameter(torch.zeros(d_model, pch_hidden)) for _ in range(n_pch)])
        self._last_pred_coding_loss = None

        # ── Predictive Coding Trunk (PCT — forward-path version) ──
        # When pct_trunk > 0, each block's residual update is shaped by
        # the top-down prediction error from the NEXT layer:
        #     h_{i+1} = h_i + Block(h_i) - α * pct_trunk * (h_i - TopDown(h_{i+1}))
        # The trunk's hidden state is pulled toward what the next layer
        # predicts it should be. This is the "deeper layers generate
        # shallower layers" inverse the cited research relies on for the
        # ≥2x OOD-gap reduction — different from the aux-loss-only PCH
        # above, which only adds a regularizer without reshaping forward.
        #
        # TopDown layers: Linear(d_model -> d_model), zero-init so the
        # residual identity is preserved at start (no shock to baseline).
        self.pct_trunk = pct_trunk
        if pct_trunk > 0 and n_pch > 0:
            self.topdown_w = nn.ParameterList(
                [nn.Parameter(torch.zeros(d_model, d_model))
                 for _ in range(n_pch)])
        else:
            self.topdown_w = None

        # ── NT modulation hook (GeneticOrchestrator → trunk) ──
        # Each block reads a per-module NT offset (B, N_NT) and applies
        # a learned per-channel gain on top of its output:
        #     h = h * (1 + alpha_nt * nt_proj(nt_offset))
        # `alpha_nt` is a ReZero scalar (init 0) — at step 0 the cortex
        # is identical to the no-NT-modulation baseline. As alpha lifts
        # off zero under the LM loss, gene → NT → trunk feedback closes.
        # `nt_proj` is Linear(N_NT, d_model); init small-random so payload
        # gradients flow on the first backward.
        N_NT_LOCAL = 7   # matches neurochem.transmitters.N_NT (avoid import)
        self.nt_proj = nn.Linear(N_NT_LOCAL, d_model, bias=False)
        nn.init.normal_(self.nt_proj.weight, std=0.02)
        self.alpha_nt = nn.Parameter(torch.zeros(1))
        # Per-block module index (block_i → module_i). The harness writes
        # the actual mapping via set_block_module_map(); default round-robin.
        self._block_module_idx: Optional[torch.Tensor] = None
        # Current per-module NT offset, set by set_nt_modulation each step.
        # Shape (B, n_modules, N_NT). None → no modulation applied (bit-
        # identical baseline path).
        self._nt_module_offset: Optional[torch.Tensor] = None

        # ── Novel-topology mechanisms (H15 / H16 / H19) — all zero-init ──
        # Each factory returns None when the spec is False/None, so the
        # legacy DSLLanguageCortex remains bit-identical when none enabled.
        # See neuroslm/dsl/novel_topology.py and docs/findings.md H15/H16/H19.
        self._grid_positions = make_grid_positions(grid_positions, d_model)
        self._episodic_memory = make_episodic_memory(episodic_memory, d_model)
        self._surprise_head = make_surprise_head(surprise_head, d_model, vocab)
        # Diagnostic: per-token surprise (B, T) from the last forward.
        # Read by the harness for loss reweighting; None when off.
        self.last_token_surprise: Optional[torch.Tensor] = None

    def set_block_module_map(self, mapping):
        """Tell the cortex which module each block represents.

        `mapping` is an iterable of length `len(self.blocks)` of integer
        indices into the orchestrator's module_names list. Falls back to
        round-robin if unset.
        """
        import torch as _t
        self._block_module_idx = _t.tensor(list(mapping), dtype=_t.long)

    def set_nt_modulation(self, per_module_offset) -> None:
        """Install the current per-module NT offset tensor.

        Args:
            per_module_offset: (B, n_modules, N_NT). Set by the harness
                each step from the GeneticOrchestrator's `baseline_offsets`.
                Pass `None` to disable NT modulation for this step.
        """
        self._nt_module_offset = per_module_offset

    def set_mat_multipliers(self, dropout_p: float, pct_trunk: float) -> None:
        """Update the MAT-phase-gated multipliers for dropout + PCT.

        Called by BRIANHarness.compute_loss before each forward when the
        arch.neuro `mechanisms { ... }` block declares phase-gated
        versions. Zero values disable the mechanism for this step.
        """
        if dropout_p > 0:
            if not isinstance(self.dropout, nn.Dropout):
                self.dropout = nn.Dropout(dropout_p)
            else:
                self.dropout.p = float(dropout_p)
        else:
            self.dropout = nn.Identity()
        self.pct_trunk = float(pct_trunk)

    def forward(self, ids: torch.Tensor) -> torch.Tensor:
        h = nn_ops.embedding(ids, self.embed)
        h = self.dropout(h)
        # ── H16: Grid-cell positional bias (additive on embedding) ──
        # Zero-init proj at construction ⇒ first forward bit-identical
        # to baseline. As proj learns, multi-scale position code shapes
        # the residual. Cheap O(L·K).
        if self._grid_positions is not None:
            pos_bias = self._grid_positions(h.shape[1],
                                            device=h.device, dtype=h.dtype)
            h = h + pos_bias.unsqueeze(0)
        self._layer_acts = []
        pred_coding_loss = torch.zeros((), device=h.device, dtype=h.dtype)

        # Stash each block's output to enable PCT top-down feedback (the
        # next layer's output predicts this layer's input).
        block_outs: List[torch.Tensor] = []
        # Precompute the per-block NT gain (B, depth, d_model) if the
        # GeneticOrchestrator has handed us a per-module offset.
        block_gain = None
        if self._nt_module_offset is not None:
            offs = self._nt_module_offset            # (B_orch, M, N_NT)
            # Batch alignment: the orchestrator caches its outputs from
            # the train step (B_orch = train batch size). When eval/OOD
            # passes a different batch (often B=1), broadcasting `(B_orch,
            # T, D)` against `(B_in, T, D)` produces (max(B_in, B_orch),
            # T, D) — silently turning a B=1 OOD batch into B=4, then
            # the LM head produces too many logit rows and CE shape
            # mismatches. Fix: collapse offs to the input's batch size.
            B_in = h.shape[0]
            if offs.shape[0] != B_in:
                if B_in == 1:
                    offs = offs.mean(dim=0, keepdim=True)
                else:
                    # Repeat or slice to match — last resort for unusual
                    # batch sizes. Mean-then-repeat keeps semantics intact.
                    offs = offs.mean(dim=0, keepdim=True).expand(B_in, -1, -1)
            n_blocks = len(self.blocks)
            if self._block_module_idx is None or \
                    self._block_module_idx.numel() != n_blocks:
                bm = torch.arange(n_blocks, device=offs.device) % offs.shape[1]
            else:
                bm = self._block_module_idx.to(offs.device)
            per_block = offs.index_select(dim=1, index=bm)  # (B_in, depth, N_NT)
            block_gain = self.alpha_nt * self.nt_proj(per_block)
        # Track which blocks were dropped so PCT can skip them (skip-aware
        # pairing). Without this, dropped blocks pollute PCT pairs with
        # zero residuals → silent regularizer-disable when stochastic_depth
        # is active. With it, PCT forms pairs over only the surviving
        # subsequence, so dropout and predictive coding compose cleanly.
        dropped_mask: List[bool] = []
        for bi, (blk, adapter) in enumerate(zip(self.blocks, self.adapters)):
            # ── Stochastic depth: skip block with linearly increasing prob ──
            is_dropped = False
            if self.training and self.stochastic_depth > 0:
                drop_prob = self.stochastic_depth * (bi + 1) / self._depth
                if torch.rand(1).item() < drop_prob:
                    is_dropped = True

            if not is_dropped:
                h = blk(h)
                h = adapter(h)
                h = self.dropout(h)
                if block_gain is not None:
                    # Broadcast (B, 1, d_model) over the (B, T, d_model) hidden
                    gain = block_gain[:, bi, :].unsqueeze(1)
                    h = h * (1.0 + gain)
            block_outs.append(h)
            dropped_mask.append(is_dropped)
            self._layer_acts.append(h.detach())

        # ── PCT trunk pass (skip-aware forward-path predictive coding) ──
        # For each adjacent pair of NON-DROPPED blocks (i, j):
        #   pred  = TopDown(h_j)              # what j expects h_i to be
        #   err   = h_i - pred                # prediction error at layer i
        # When stochastic depth dropped some blocks, we form pairs over
        # only the surviving subsequence so PCT residuals are always
        # meaningful (not just `0 - 0 = 0` for dropped pairs).
        # Zero-init topdown_w means pct_correction starts at exactly
        # block_outs[i], so err = 0 and h_final is unperturbed at init.
        if self.pct_trunk > 0 and self.topdown_w is not None:
            alpha = 0.5      # safety damping so PCT can't dominate
            depth_t = len(block_outs)
            pct_correction = torch.zeros_like(block_outs[-1])
            # Active (non-dropped) block indices, in order
            active = [i for i, d in enumerate(dropped_mask) if not d]
            # Fall back to all blocks if nothing was dropped (common case
            # at eval time + when stochastic_depth=0)
            if not active:
                active = list(range(depth_t))
            for k in range(len(active) - 1):
                i_lo = active[k]
                i_hi = active[k + 1]
                # Per-layer-pair residual difference. Use topdown_w[i_lo]
                # — the head is indexed by the LOWER block of the pair.
                if i_lo >= len(self.topdown_w):
                    break
                residual_diff = block_outs[i_lo] - block_outs[i_hi]
                err = nn_ops.linear(residual_diff, self.topdown_w[i_lo])
                # Weight by 1/(depth - i_lo) so the pair closest to the
                # output (largest i_lo) dominates the correction.
                pct_correction = (
                    pct_correction - alpha * self.pct_trunk * err / (depth_t - i_lo)
                )
            # Apply the aggregated correction to the final hidden state
            # that feeds the LM head.
            block_outs[-1] = block_outs[-1] + pct_correction

        # PCH aux loss (Brain-compatible): per-pair MSE between layers.
        # Computed on the *post-PCT* block outputs so the regularizer
        # measures what's left after top-down explanation.
        for i in range(len(block_outs) - 1):
            if i >= len(self.pch_w1):
                break
            pred_coding_loss = pred_coding_loss + nn_ops.predictive_coding_head(
                block_outs[i], block_outs[i + 1],
                self.pch_w1[i], self.pch_b1[i], self.pch_w2[i])
        if len(self.pch_w1) > 0:
            pred_coding_loss = pred_coding_loss / len(self.pch_w1)

        # Stash aux loss so the harness can aggregate it into total loss
        # without changing the forward signature (still returns logits).
        self._last_pred_coding_loss = pred_coding_loss
        h_final = block_outs[-1] if block_outs else h
        h_final = nn_ops.rmsnorm(h_final, self.gamma_f)
        logits = nn_ops.linear(h_final, self.lm_head)

        # ── H19: Surprise head — local-context NLL vs global NLL ──
        # Uses ids as next-token labels (NLL of token t given <t under
        # causal mask). Surprise (B, T) is exposed for episodic write
        # gating and downstream loss-reweighting hooks.
        if self._surprise_head is not None and self.training:
            self._surprise_head.set_labels(ids)
            self.last_token_surprise = self._surprise_head(h_final, logits)
        else:
            self.last_token_surprise = None

        # ── H15: Episodic kNN memory — read (gated) + write (surprise) ──
        # alpha=0 at init ⇒ delta is zero ⇒ logits unchanged. As alpha
        # lifts off zero under LM loss, blended episodes enter the
        # residual and we re-project to logits. Keeps the no-op path
        # free of extra lm_head work.
        if self._episodic_memory is not None:
            delta = self._episodic_memory(h_final,
                                          surprise=self.last_token_surprise)
            if self._episodic_memory.alpha.detach().abs().item() > 0:
                h_final_blended = h_final + delta
                logits = nn_ops.linear(h_final_blended, self.lm_head)
        return logits


def build_dsl_language_cortex(vocab: int, d_model: int, depth: int,
                               n_heads: int, max_ctx: int, dropout: float = 0.0,
                               n_kv_heads: Optional[int] = None,
                               geometry_expansion: float = 2.0,
                               mod_capacity: float = 0.5,
                               pct_trunk: float = 0.0,
                               tonnetz_period: int = 0,
                               stochastic_depth: float = 0.0,
                               grid_positions=None,
                               episodic_memory=None,
                               surprise_head=None) -> DSLLanguageCortex:
    """Assemble Brain's full LanguageCortex from pure-DSL blocks.

    `pct_trunk > 0` enables forward-path predictive coding: each layer
    is pulled toward what the next layer predicts it should be. Zero-init
    so the residual identity is preserved at start. Targets the >=2x OOD
    gap reduction claimed for PCT vs. standard backprop training.

    Novel-topology kwargs (all default OFF, see neuroslm/dsl/novel_topology.py):
        grid_positions     — H16, multi-scale grid-cell positional bias.
        episodic_memory    — H15, kNN memory blended via ReZero gate.
        surprise_head      — H19, local-NLL surprise (gates H15 writes).
    Each accepts True / False / None / dict; bit-identical to baseline
    when all are False/None.
    """
    return DSLLanguageCortex(vocab, d_model, depth, n_heads, max_ctx,
                              n_kv_heads, geometry_expansion, mod_capacity,
                              dropout=dropout, pct_trunk=pct_trunk,
                              tonnetz_period=tonnetz_period,
                              stochastic_depth=stochastic_depth,
                              grid_positions=grid_positions,
                              episodic_memory=episodic_memory,
                              surprise_head=surprise_head)
