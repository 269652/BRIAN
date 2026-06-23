# -*- coding: utf-8 -*-
"""DSL v2: `model { ... }` block parser — THSD-grounded universal LM spec.

Every LM is a cellular sheaf F with stalks = token hidden states and
coboundary δ implementing attention + FFN.  The `kind` field identifies
which sheaf configuration to use:

  kind: gpt2   — trivial H¹, learned pos, LayerNorm, fused QKV, GELU
  kind: llama  — trivial H¹, RoPE, RMSNorm, GQA, SwiGLU (LLaMA / SmolLM2 family)
  kind: qwen   — same as llama (Qwen2 / Qwen2.5 family, same architecture)
  kind: brian  — non-trivial sheaf with THSD mechanisms (KJPLA, Noether, etc.)

The `model { }` block is parsed from arch.neuro files independently of the
existing `training { }` block — backward-compatible with all existing files.
"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Optional

from .training_config import _split_top_level_kv, _strip_braces, _strip_quotes, _parse_bool

_VALID_KINDS = {"gpt2", "llama", "qwen", "mistral", "brian"}
_VALID_POS = {"none", "learned", "rope", "alibi"}
_VALID_FF_ACT = {"gelu", "swiglu", "geglu", "relu", "silu"}
_VALID_NORM = {"layernorm", "rmsnorm"}


@dataclass
class SheafConfig:
    """Sheaf topology configuration — the architectural parameters of one LM.

    In THSD terms: d_model is the stalk dimension, depth is the number of
    coboundary operators, heads/kv_heads define the interaction pattern in C¹.
    """
    dim: int = 768
    depth: int = 12
    heads: int = 12
    kv_heads: int = 12        # GQA: < heads enables grouped-query attention
    context: int = 1024       # max sequence length
    vocab: int = 50257

    # Positional encoding
    pos: str = "rope"         # "none" | "learned" | "rope" | "alibi"
    rope_base: int = 10000    # RoPE theta (ignored when pos != "rope")

    # Feed-forward
    ff_mult: float = 4.0     # intermediate_size = dim * ff_mult
    ff_act: str = "swiglu"   # "gelu" | "swiglu" | "geglu" | "relu" | "silu"

    # Normalization
    norm: str = "rmsnorm"    # "layernorm" | "rmsnorm"
    norm_eps: float = 1e-5

    # Output head
    tie_embed: bool = True
    bias: bool = False


@dataclass
class ModelSpec:
    """Top-level DSL v2 model specification.

    Maps to a `build_model(spec)` call that returns an nn.Module with:
      - `forward(input_ids: LongTensor[B, T]) -> FloatTensor[B, T, vocab]`
      - `load_state_dict(...)` with our canonical parameter naming

    The `weights` field is an optional HF model ID for weight loading:
      `weights: "hf:openai-community/gpt2"`
    """
    kind: str = "llama"
    weights: Optional[str] = None
    sheaf: SheafConfig = field(default_factory=SheafConfig)


def _normalize_block_syntax(text: str) -> str:
    """Convert bare `key {` block syntax to `key: {` colon syntax.

    This allows the simplified v2 DSL surface:
      sheaf {
          dim: 768
      }
    to be parsed by the same _split_top_level_kv machinery as:
      sheaf: { dim: 768 }
    """
    import re
    # Replace `identifier {` (not already followed by or preceded by `:`)
    # with `identifier: {`.
    # Pattern: word char(s), optional whitespace, `{` — but only when the
    # word is not already preceded by a colon.
    normalized = re.sub(r'(?<![:\w])(\b[a-zA-Z_]\w*)\s*\{', r'\1: {', text)
    return normalized


def parse_model_block(text: str) -> ModelSpec:
    """Parse a `model { ... }` DSL block (or the full arch text).

    Accepts either:
      - The full arch.neuro text (finds the `model { ... }` block)
      - Just the body text (braces included or stripped)

    Both `sheaf: { ... }` (colon) and `sheaf { ... }` (block) syntax are
    supported; the latter is normalized to the former before parsing.
    """
    text = _normalize_block_syntax(text.strip())

    # If the text contains `model {`, extract the body.
    if "model" in text and "{" in text:
        body = _extract_block(text, "model")
    else:
        body = _strip_braces(text)

    spec = ModelSpec()
    if not body.strip():
        return spec

    props = _split_top_level_kv(body)

    if "kind" in props:
        kind = _strip_quotes(props["kind"]).lower()
        if kind not in _VALID_KINDS:
            raise ValueError(
                f"model.kind={kind!r} is not valid; expected one of "
                f"{sorted(_VALID_KINDS)}"
            )
        spec.kind = kind

    if "weights" in props:
        spec.weights = _strip_quotes(props["weights"])

    if "sheaf" in props:
        spec.sheaf = _parse_sheaf(props["sheaf"])

    return spec


def _extract_block(text: str, keyword: str) -> str:
    """Extract the body of the first `keyword { ... }` top-level block."""
    idx = text.find(keyword)
    if idx == -1:
        return ""
    start = text.find("{", idx)
    if start == -1:
        return ""
    depth = 0
    for i in range(start, len(text)):
        if text[i] == "{":
            depth += 1
        elif text[i] == "}":
            depth -= 1
            if depth == 0:
                return text[start + 1:i]
    return text[start + 1:]


def _parse_sheaf(raw: str) -> SheafConfig:
    p = _split_top_level_kv(_strip_braces(raw))
    out = SheafConfig()

    if "dim" in p:
        v = int(p["dim"])
        if v <= 0:
            raise ValueError(f"sheaf.dim={v} must be > 0")
        out.dim = v
    if "depth" in p:
        out.depth = int(p["depth"])
    if "heads" in p:
        out.heads = int(p["heads"])
    if "kv_heads" in p:
        out.kv_heads = int(p["kv_heads"])
    if "context" in p:
        out.context = int(p["context"])
    if "vocab" in p:
        out.vocab = int(p["vocab"])

    if "pos" in p:
        v = _strip_quotes(p["pos"]).lower()
        if v not in _VALID_POS:
            raise ValueError(f"sheaf.pos={v!r}; expected one of {sorted(_VALID_POS)}")
        out.pos = v
    if "rope_base" in p:
        out.rope_base = int(p["rope_base"])

    if "ff_mult" in p:
        out.ff_mult = float(p["ff_mult"])
    if "ff_act" in p:
        v = _strip_quotes(p["ff_act"]).lower()
        if v not in _VALID_FF_ACT:
            raise ValueError(f"sheaf.ff_act={v!r}; expected one of {sorted(_VALID_FF_ACT)}")
        out.ff_act = v

    if "norm" in p:
        v = _strip_quotes(p["norm"]).lower()
        if v not in _VALID_NORM:
            raise ValueError(f"sheaf.norm={v!r}; expected one of {sorted(_VALID_NORM)}")
        out.norm = v
    if "norm_eps" in p:
        out.norm_eps = float(p["norm_eps"])

    if "tie_embed" in p:
        out.tie_embed = _parse_bool(p["tie_embed"])
    if "bias" in p:
        out.bias = _parse_bool(p["bias"])

    return out
