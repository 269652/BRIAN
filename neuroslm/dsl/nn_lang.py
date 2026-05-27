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


# Op atoms callable from the DSL (introspected from nn_ops)
_OP_NAMES = {
    name for name in dir(nn_ops)
    if callable(getattr(nn_ops, name)) and not name.startswith("_")
}


# ── Parameter init ─────────────────────────────────────────────────────

def _alloc(init: str, shape) -> torch.Tensor:
    """Allocate a parameter tensor. Init only needs the right *shape* for
    exact-match tests (weights get synced); the distributions are sensible
    defaults for real training."""
    if not isinstance(shape, tuple):
        shape = (shape,)
    if init == "ones":
        return torch.ones(*shape)
    if init == "zeros":
        return torch.zeros(*shape)
    if init == "xavier":
        t = torch.empty(*shape)
        if t.dim() >= 2:
            nn.init.xavier_uniform_(t)
        else:
            nn.init.normal_(t, std=0.02)
        return t
    if init == "kaiming":
        t = torch.empty(*shape)
        if t.dim() >= 2:
            nn.init.kaiming_uniform_(t, a=5 ** 0.5)
        else:
            nn.init.normal_(t, std=0.02)
        return t
    if init == "normal":
        return torch.randn(*shape) * 0.02
    raise ValueError(f"unknown init spec {init!r}")


# ── AST ────────────────────────────────────────────────────────────────

@dataclass
class ParamDecl:
    name: str
    shape_src: str   # raw "(D, D)" — eval'd against kwargs at init
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
_PARAM_RE = re.compile(r"\bparam\s+(\w+)\s*:\s*(\([^)]*\))\s+init=(\w+)")
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
    sublayers = {sm.group(1): sm.group(2) for sm in _SUBLAYER_RE.finditer(body_text)}

    fm = _FORWARD_RE.search(body_text)
    if not fm:
        raise ValueError(f"layer {name!r} has no forward block")
    fwd_args = [a.strip() for a in fm.group(1).split(",") if a.strip()]
    f_start = fm.end() - 1
    f_end = _find_matching_brace(body_text, f_start)
    fwd_text = body_text[f_start + 1:f_end]

    body = _parse_forward_body(fwd_text)
    return LayerDef(name, args, params, sublayers, fwd_args, body)


def _parse_forward_body(text: str) -> List[Stmt]:
    stmts = []
    # Statements separated by newlines; ignore blanks/braces.
    for raw in text.split("\n"):
        line = raw.strip().rstrip(";")
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
    # forward
    lines.append("")
    lines.append(f"    def forward(self, {', '.join(ld.fwd_args)}):")
    locals_ = set(ld.fwd_args)
    for st in ld.body:
        rhs = _lower_expr(st.expr, locals_, ld.sublayers)
        if st.is_return:
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
