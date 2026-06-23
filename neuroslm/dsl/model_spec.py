# -*- coding: utf-8 -*-
"""DSL v3: `model { ... }` block parser — unified THSD grammar for all LMs.

Every LM is a cellular sheaf F:
  C⁰(F) — token stalks of dimension `dim`
  C¹(F) — interaction co-chains (attention)
  δ: C⁰ → C¹  — coboundary (attention operator)
  τ: C¹ → C⁰  — transition (FFN)

All architectural dynamics are expressed in nested sub-blocks:
  embed      — token + positional embedding
  coboundary — δ: attention operator (mha/gqa/mla/kjpla/swa)
  transition — τ: FFN (mlp/swiglu/geglu/moe/liouville_symplectic)
  norm       — normalisation (layernorm/rmsnorm), pre/post
  output     — lm_head weight tying
  dropout    — all dropout coefficients
  init       — weight initialisation scheme
  layers     — per-layer overrides (MoE dense prefix, SWA window)
  diagnostic — auxiliary diagnostic modules (topo_charge)
  optimizer  — AdamW/SGD + scheduler (in model-level block)

`kind: gpt2|llama|qwen|deepseek|mistral|brian` is macro sugar that
pre-fills sub-block defaults; explicit blocks override macro defaults.
"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Optional, Tuple

from .training_config import _split_top_level_kv, _strip_braces, _strip_quotes, _parse_bool

# ── Valid enum sets ────────────────────────────────────────────────────────────

_VALID_COBOUNDARY_TYPES = {"mha", "gqa", "mla", "swa", "kjpla", "linear"}
_VALID_TRANSITION_TYPES = {"mlp", "swiglu", "geglu", "moe", "liouville_symplectic"}
_VALID_NORM_TYPES       = {"layernorm", "rmsnorm"}
_VALID_NORM_PLACEMENTS  = {"pre", "post"}
_VALID_EMBED_POSITIONS  = {"none", "learned", "sinusoidal", "alibi"}
_VALID_ROPE_SCALINGS    = {"none", "yarn", "dynamic", "longrope"}
_VALID_DIAGNOSTIC_TYPES = {"topo_charge"}
_VALID_OPTIMIZER_TYPES  = {"adamw", "adam", "sgd", "muon", "lion"}
_VALID_SCHEDULER_TYPES  = {"cosine", "linear", "constant", "wsd", "warmup_stable_decay"}
_VALID_ROUTING_TYPES    = {"topk", "expert_choice", "softmax"}
_VALID_INIT_METHODS     = {"normal", "scaled", "xavier", "trunc_normal"}
_VALID_KINDS            = {"gpt2", "llama", "qwen", "deepseek", "mistral", "brian"}

# ── DSL equation name → impl type string ──────────────────────────────────────
# These allow arch.neuro files to write `equation: grouped_query_attention`
# instead of `type: gqa`. The equation name is the canonical DSL identifier
# defined in @lib/primitives/; the type string is the internal impl key.

_EQUATION_TO_COBOUNDARY_TYPE: dict = {
    "multi_head_attention":          "mha",
    "grouped_query_attention":       "gqa",
    "sliding_window_attention":      "swa",
    "multi_head_latent_attention":   "mla",
    "kjpla_phase_lattice_attention": "kjpla",
    "linear_attention":              "linear",
}

_EQUATION_TO_TRANSITION_TYPE: dict = {
    "standard_ffn":              "mlp",
    "gelu_ffn":                  "mlp",
    "swish_ffn":                 "mlp",
    "squared_relu_ffn":          "mlp",
    "swiglu_ffn":                "swiglu",
    "geglu_ffn":                 "geglu",
    "moe_ffn":                   "moe",
    "liouville_symplectic_ffn":  "liouville_symplectic",
}

_EQUATION_TO_NORM_TYPE: dict = {
    "layer_norm": "layernorm",
    "rms_norm":   "rmsnorm",
}


# ── Sub-block dataclasses ──────────────────────────────────────────────────────

@dataclass
class RopeConfig:
    """Rotary positional encoding applied within the coboundary (C¹ space).

    YaRN scaling (Peng et al. 2023) extends context with:
      original_max_position — training length before extension
      factor                — extension factor (new_len / original_len)
      beta_fast / beta_slow — frequency band thresholds
    """
    base: int = 10000
    scaling: str = "none"                # "none" | "yarn" | "dynamic" | "longrope"
    original_max_position: int = 4096    # YaRN: original training context
    factor: float = 1.0                  # YaRN: extension factor
    beta_fast: float = 32.0             # YaRN: high-freq band threshold
    beta_slow: float = 1.0              # YaRN: low-freq band threshold


@dataclass
class MLAConfig:
    """DeepSeek Multi-Head Latent Attention (MLA) KV-compression parameters.

    Compresses KV cache via LoRA-style projection:
      KV: C⁰ → ℝ^{kv_lora_rank} → (K, V) keys
      Q:  C⁰ → ℝ^{q_lora_rank} → Q queries

    qk_nope_dim — dims with no positional encoding (shared key)
    qk_rope_dim — dims where RoPE is applied
    """
    kv_lora_rank: int = 512
    q_lora_rank: int = 1536
    qk_nope_dim: int = 128
    qk_rope_dim: int = 64
    v_dim: int = 128


@dataclass
class CoboundaryConfig:
    """C⁰→C¹ coboundary operator (attention mechanism).

    type:
      mha   — multi-head attention, fused QKV Conv1D (GPT-2)
      gqa   — grouped-query attention, separate Q/K/V (LLaMA/Qwen/Mistral)
      swa   — sliding-window attention, GQA with local window (Mistral/Phi)
      mla   — multi-head latent attention, LoRA KV (DeepSeek-V2/V3)
      kjpla — Kuramoto-Josephson phase-lattice attention (BRIAN)

    head_dim: explicit head dim; derived as dim//heads when None
    softmax_scale: attention scale; 1/sqrt(head_dim) when None
    window_size: SWA window; None means full causal attention
    dropout: attention-softmax dropout (usually 0 in modern LMs)
    resid_dropout: dropout applied after o_proj output
    """
    type: str = "gqa"
    qkv: str = "separate"              # "fused" (GPT-2) | "separate" (LLaMA/Qwen)
    bias: bool = False                 # o_proj bias
    qkv_bias: bool = False             # q/k/v proj bias (Qwen2)
    head_dim: Optional[int] = None     # explicit; derived from dim/heads if None
    softmax_scale: Optional[float] = None  # custom scale; 1/√head_dim if None
    window_size: Optional[int] = None  # SWA local window; None = full
    dropout: float = 0.0              # softmax attention dropout
    resid_dropout: float = 0.0        # post-output residual dropout
    rope: Optional[RopeConfig] = None
    mla: Optional[MLAConfig] = None
    # KJPLA-specific
    josephson_strength: float = 0.1
    entropy_eps: float = 0.01


@dataclass
class MoEConfig:
    """Mixture-of-Experts configuration (DeepSeek-V2/V3, Mixtral).

    routing:
      topk          — top-k expert selection (standard)
      expert_choice — each expert picks its own top-k tokens
      softmax       — soft routing (all experts, weighted)

    load_balance_coef — auxiliary load-balancing loss coefficient
    capacity_factor   — router capacity relative to n_active/n_experts
    """
    n_experts: int = 64
    n_active: int = 8
    shared_experts: int = 1           # always-active dense experts
    routing: str = "topk"
    load_balance_coef: float = 0.001  # aux loss weight for load balancing
    capacity_factor: float = 1.5      # expert token capacity factor


@dataclass
class TransitionConfig:
    """C¹→C⁰ transition operator (FFN / MLP).

    type:
      mlp                  — dense FFN: Linear + act + Linear
      swiglu               — SwiGLU: silu(gate_proj(x)) × up_proj(x) → down_proj (LLaMA)
      geglu                — GELU-gated variant of SwiGLU
      moe                  — mixture-of-experts (DeepSeek / Mixtral)
      liouville_symplectic — leapfrog MLP preserving det(J)=1 (BRIAN)

    dropout — post-FFN residual dropout
    """
    type: str = "swiglu"
    ff_mult: float = 4.0
    activation: str = "gelu"           # for type="mlp"
    bias: bool = False
    dropout: float = 0.0               # post-FFN residual dropout
    moe: Optional[MoEConfig] = None
    noether_strength: float = 0.01     # liouville_symplectic: conservation loss


@dataclass
class EmbedConfig:
    """Token and positional embedding configuration."""
    tokens: str = "learned"            # always "learned"
    position: str = "none"             # "none" | "learned" | "sinusoidal" | "alibi"
    dropout: float = 0.0               # post-embed dropout (GPT-2: 0.1)
    scale_by_dim: bool = False         # some models scale embed by √dim


@dataclass
class NormConfig:
    """Normalisation applied around each coboundary/transition operator."""
    type: str = "rmsnorm"              # "layernorm" | "rmsnorm"
    placement: str = "pre"             # "pre" | "post"
    eps: float = 1e-5


@dataclass
class OutputConfig:
    """LM head / output projection configuration."""
    tie_embed: bool = True
    bias: bool = False


@dataclass
class DropoutConfig:
    """All dropout coefficients in one place.

    embed  — applied to the combined token+position embedding
    attn   — attention softmax dropout (alias for coboundary.dropout)
    resid  — residual stream dropout after attention and FFN
    ffn    — dropout inside the FFN (rare; applies between FF layers)
    path   — stochastic depth / layer drop probability
    """
    embed: float = 0.0
    attn: float = 0.0
    resid: float = 0.0
    ffn: float = 0.0
    path: float = 0.0


@dataclass
class InitConfig:
    """Weight initialisation scheme.

    method:
      normal      — N(0, std) for all weights (GPT-2 default)
      scaled      — N(0, std/√(2L)) for output projections (GPT-2 paper §2.3)
      xavier      — Xavier uniform for attention projections
      trunc_normal — truncated N(0, std) clipped at ±2σ

    output_scale — whether to scale output projections by 1/√(2·depth)
    """
    std: float = 0.02
    method: str = "normal"
    output_scale: bool = False         # GPT-2: scale c_proj/down_proj by 1/√(2·depth)


@dataclass
class SchedulerConfig:
    """Learning-rate scheduler specification.

    type:
      cosine                — cosine annealing to min_lr
      linear                — linear decay to min_lr
      constant              — flat LR after warmup
      wsd / warmup_stable_decay — 3-phase: warmup → stable → cosine decay
    """
    type: str = "cosine"
    warmup_steps: int = 100
    min_lr_ratio: float = 0.1
    decay_steps: Optional[int] = None  # total decay steps; derived from training if None
    stable_ratio: float = 0.9         # WSD: fraction of steps in stable phase


@dataclass
class OptimizerConfig:
    """Optimizer configuration for this model.

    betas — (β₁, β₂) for Adam-family optimizers
    grad_clip — global gradient norm clip; 0 = disabled
    """
    type: str = "adamw"
    lr: float = 3e-4
    betas: Tuple[float, float] = (0.9, 0.95)
    weight_decay: float = 0.1
    eps: float = 1e-8
    grad_clip: float = 1.0
    scheduler: Optional[SchedulerConfig] = None


@dataclass
class LayersConfig:
    """Per-layer overrides for heterogeneous architectures.

    first_k_dense — first N layers use a dense FFN instead of MoE
                    (DeepSeek: first 3 layers are always dense)
    window_size   — default SWA window for all layers (can be overridden
                    per-layer; None = full causal for all layers)
    """
    first_k_dense: int = 0
    window_size: Optional[int] = None


@dataclass
class DiagnosticConfig:
    """Auxiliary diagnostic module attached to the sheaf.

    type:
      topo_charge — Berg-Lüscher discrete Pontryagin charge per attention head
                    penalty = α·(Q_h - Q_target)² + γ·ε_ortho
    """
    type: str = "topo_charge"
    alpha: float = 0.01
    gamma: float = 0.005


@dataclass
class SheafConfig:
    """Complete THSD cellular sheaf specification.

    Scale parameters (stalk dim, depth, heads) plus nested sub-blocks for
    every architectural dynamic, dropout, init, and layer-level config.
    """
    # ── Scale ──────────────────────────────────────────────────────────────
    dim: int = 768
    depth: int = 12
    heads: int = 12
    kv_heads: int = 12
    context: int = 1024
    vocab: int = 50257

    # ── Sub-block configs ──────────────────────────────────────────────────
    embed:       EmbedConfig       = field(default_factory=EmbedConfig)
    coboundary:  CoboundaryConfig  = field(default_factory=CoboundaryConfig)
    transition:  TransitionConfig  = field(default_factory=TransitionConfig)
    norm:        NormConfig        = field(default_factory=NormConfig)
    output:      OutputConfig      = field(default_factory=OutputConfig)
    dropout:     DropoutConfig     = field(default_factory=DropoutConfig)
    init:        InitConfig        = field(default_factory=InitConfig)
    layers:      LayersConfig      = field(default_factory=LayersConfig)
    diagnostic:  Optional[DiagnosticConfig] = None


@dataclass
class ModelSpec:
    """Top-level DSL v3 model specification.

    `kind` is optional macro sugar — if present, pre-fills sheaf sub-block
    defaults before any explicit blocks are applied.

    `optimizer` is model-level (not inside sheaf) because it is not an
    architectural property — it is a training-time decision.
    """
    kind:      Optional[str]       = None
    weights:   Optional[str]       = None
    sheaf:     SheafConfig         = field(default_factory=SheafConfig)
    optimizer: Optional[OptimizerConfig] = None


# ── Kind macro defaults ────────────────────────────────────────────────────────

_KIND_DEFAULTS: dict = {
    "gpt2": {
        "embed":      {"position": "learned", "dropout": 0.0},
        "coboundary": {"type": "mha", "qkv": "fused", "bias": True},
        "transition": {"type": "mlp", "ff_mult": 4.0, "activation": "gelu", "bias": True},
        "norm":       {"type": "layernorm"},
        "output":     {"tie_embed": True},
        "init":       {"std": 0.02, "output_scale": True, "method": "normal"},
    },
    "llama": {
        "embed":      {"position": "none"},
        "coboundary": {"type": "gqa", "qkv": "separate"},
        "transition": {"type": "swiglu"},
        "norm":       {"type": "rmsnorm"},
        "output":     {"tie_embed": True},
        "init":       {"std": 0.02},
    },
    "qwen": {
        "embed":      {"position": "none"},
        "coboundary": {"type": "gqa", "qkv": "separate", "qkv_bias": True},
        "transition": {"type": "swiglu"},
        "norm":       {"type": "rmsnorm"},
        "output":     {"tie_embed": True},
        "init":       {"std": 0.02},
    },
    "mistral": {
        "embed":      {"position": "none"},
        "coboundary": {"type": "gqa", "qkv": "separate"},
        "transition": {"type": "swiglu"},
        "norm":       {"type": "rmsnorm"},
        "output":     {"tie_embed": False},
        "init":       {"std": 0.02},
    },
    "deepseek": {
        "embed":      {"position": "none"},
        "coboundary": {"type": "mla"},
        "transition": {"type": "moe"},
        "norm":       {"type": "rmsnorm"},
        "output":     {"tie_embed": False},
        "init":       {"std": 0.006},
    },
    "brian": {
        "embed":      {"position": "none"},
        "coboundary": {"type": "kjpla"},
        "transition": {"type": "liouville_symplectic"},
        "norm":       {"type": "rmsnorm"},
        "output":     {"tie_embed": True},
        "init":       {"std": 0.02},
    },
}


# ── Block-syntax normaliser ────────────────────────────────────────────────────

def _normalize_block_syntax(text: str) -> str:
    """Convert bare `key {` block syntax → `key: {` colon syntax."""
    import re
    return re.sub(r'(?<![:\w])(\b[a-zA-Z_]\w*)\s*\{', r'\1: {', text)


# ── Block extractor ───────────────────────────────────────────────────────────

def _extract_block(text: str, keyword: str) -> str:
    """Extract the body of the first `keyword { ... }` top-level block.

    Handles nested braces correctly.
    """
    idx = text.find(keyword)
    if idx == -1:
        return ""
    # Ensure it's a keyword followed by `: {` or just `{`
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


def _strip_nested_block(text: str, keyword: str) -> str:
    """Remove a nested `keyword: { ... }` block from text (handles nesting)."""
    import re
    pattern = rf'\b{keyword}\s*:\s*\{{'
    result = []
    i = 0
    while i < len(text):
        m = re.search(pattern, text[i:])
        if not m:
            result.append(text[i:])
            break
        result.append(text[i:i + m.start()])
        start = i + m.end() - 1  # position of opening {
        depth = 1
        j = start + 1
        while j < len(text) and depth > 0:
            if text[j] == '{':
                depth += 1
            elif text[j] == '}':
                depth -= 1
            j += 1
        i = j
    return "".join(result)


def _flat_kv_strip_nested(body: str, *keywords) -> dict:
    """Strip named nested blocks then parse flat kv pairs from remainder."""
    flat = body
    for kw in keywords:
        flat = _strip_nested_block(flat, kw)
    return _split_top_level_kv(flat)


def _strip_inline_comments(text: str) -> str:
    """Strip `# ...` comments from DSL text, preserving quoted strings."""
    lines = []
    for line in text.splitlines():
        in_string = False
        result = []
        i = 0
        while i < len(line):
            c = line[i]
            if c == '"':
                in_string = not in_string
                result.append(c)
            elif c == '#' and not in_string:
                break
            else:
                result.append(c)
            i += 1
        lines.append(''.join(result).rstrip())
    return '\n'.join(lines)


def _strip_import_lines(text: str) -> str:
    """Remove DSL `import { ... } from "..."` lines before parsing."""
    import re
    return re.sub(r'^\s*import\s*\{[^}]*\}\s*from\s*"[^"]*"\s*$', '', text, flags=re.MULTILINE)


def _extract_all_equations(body: str) -> list:
    """Return all bare `equation: identifier` values found in a block body."""
    import re
    return re.findall(r'\bequation\s*:\s*([a-zA-Z_]\w*)', body)


# ── Sub-block parsers ─────────────────────────────────────────────────────────

def _parse_rope(raw: str) -> RopeConfig:
    p = _split_top_level_kv(_strip_braces(raw))
    out = RopeConfig()
    if "base" in p:
        out.base = int(p["base"])
    if "scaling" in p:
        v = _strip_quotes(p["scaling"]).lower()
        if v not in _VALID_ROPE_SCALINGS:
            raise ValueError(f"rope.scaling={v!r}; expected {sorted(_VALID_ROPE_SCALINGS)}")
        out.scaling = v
    if "original_max_position" in p:
        out.original_max_position = int(p["original_max_position"])
    if "factor" in p:
        out.factor = float(p["factor"])
    if "beta_fast" in p:
        out.beta_fast = float(p["beta_fast"])
    if "beta_slow" in p:
        out.beta_slow = float(p["beta_slow"])
    return out


def _parse_mla(raw: str) -> MLAConfig:
    p = _split_top_level_kv(_strip_braces(raw))
    out = MLAConfig()
    for fname in ("kv_lora_rank", "q_lora_rank", "qk_nope_dim", "qk_rope_dim", "v_dim"):
        if fname in p:
            setattr(out, fname, int(p[fname]))
    return out


def _parse_moe_block(raw: str) -> MoEConfig:
    p = _split_top_level_kv(_strip_braces(raw))
    out = MoEConfig()
    for fname in ("n_experts", "n_active", "shared_experts"):
        if fname in p:
            setattr(out, fname, int(p[fname]))
    if "routing" in p:
        v = _strip_quotes(p["routing"]).lower()
        if v not in _VALID_ROUTING_TYPES:
            raise ValueError(f"moe.routing={v!r}; expected {sorted(_VALID_ROUTING_TYPES)}")
        out.routing = v
    if "load_balance_coef" in p:
        out.load_balance_coef = float(p["load_balance_coef"])
    if "capacity_factor" in p:
        out.capacity_factor = float(p["capacity_factor"])
    return out


def _parse_coboundary(raw: str) -> CoboundaryConfig:
    body = _strip_braces(raw)
    rope_body = _extract_block(body, "rope")
    mla_body  = _extract_block(body, "mla")
    p = _flat_kv_strip_nested(body, "rope", "mla")

    out = CoboundaryConfig()
    for eq in _extract_all_equations(body):
        if eq in _EQUATION_TO_COBOUNDARY_TYPE:
            out.type = _EQUATION_TO_COBOUNDARY_TYPE[eq]
            if out.type == "mha":
                out.qkv = "fused"
            break
    if "type" in p:
        v = _strip_quotes(p["type"]).lower()
        if v not in _VALID_COBOUNDARY_TYPES:
            raise ValueError(f"coboundary.type={v!r}; expected {sorted(_VALID_COBOUNDARY_TYPES)}")
        out.type = v
        if v == "mha":
            out.qkv = "fused"
    if "qkv" in p:
        out.qkv = _strip_quotes(p["qkv"]).lower()
    if "bias" in p:
        out.bias = _parse_bool(p["bias"])
    if "qkv_bias" in p:
        out.qkv_bias = _parse_bool(p["qkv_bias"])
    if "head_dim" in p:
        out.head_dim = int(p["head_dim"])
    if "softmax_scale" in p:
        out.softmax_scale = float(p["softmax_scale"])
    if "window_size" in p:
        out.window_size = int(p["window_size"])
    if "dropout" in p:
        out.dropout = float(p["dropout"])
    if "resid_dropout" in p:
        out.resid_dropout = float(p["resid_dropout"])
    if "josephson_strength" in p:
        out.josephson_strength = float(p["josephson_strength"])
    if "entropy_eps" in p:
        out.entropy_eps = float(p["entropy_eps"])
    # MLA flat fields (when not in a nested mla {} block)
    mla_flat_keys = {"kv_lora_rank", "q_lora_rank", "qk_nope_dim", "qk_rope_dim", "v_dim"}
    if mla_flat_keys & set(p):
        out.mla = MLAConfig()
        for k in mla_flat_keys:
            if k in p:
                setattr(out.mla, k, int(p[k]))
    if rope_body:
        out.rope = _parse_rope(rope_body)
    if mla_body:
        out.mla = _parse_mla(mla_body)
    return out


def _parse_transition(raw: str) -> TransitionConfig:
    body = _strip_braces(raw)
    moe_body = _extract_block(body, "moe")
    p = _flat_kv_strip_nested(body, "moe")

    out = TransitionConfig()
    for eq in _extract_all_equations(body):
        if eq in _EQUATION_TO_TRANSITION_TYPE:
            out.type = _EQUATION_TO_TRANSITION_TYPE[eq]
            break
    if "type" in p:
        v = _strip_quotes(p["type"]).lower()
        if v not in _VALID_TRANSITION_TYPES:
            raise ValueError(f"transition.type={v!r}; expected {sorted(_VALID_TRANSITION_TYPES)}")
        out.type = v
    if "ff_mult" in p:
        out.ff_mult = float(p["ff_mult"])
    if "activation" in p:
        out.activation = _strip_quotes(p["activation"]).lower()
    if "bias" in p:
        out.bias = _parse_bool(p["bias"])
    if "dropout" in p:
        out.dropout = float(p["dropout"])
    if "noether_strength" in p:
        out.noether_strength = float(p["noether_strength"])
    # MoE flat fields
    moe_flat = {"n_experts", "n_active", "shared_experts"}
    if moe_flat & set(p):
        out.moe = MoEConfig()
        for k in moe_flat:
            if k in p:
                setattr(out.moe, k, int(p[k]))
        if "routing" in p:
            out.moe.routing = _strip_quotes(p["routing"]).lower()
        if "load_balance_coef" in p:
            out.moe.load_balance_coef = float(p["load_balance_coef"])
    if moe_body:
        out.moe = _parse_moe_block(moe_body)
    return out


def _parse_embed(raw: str) -> EmbedConfig:
    p = _split_top_level_kv(_strip_braces(raw))
    out = EmbedConfig()
    if "tokens" in p:
        out.tokens = _strip_quotes(p["tokens"]).lower()
    if "position" in p:
        v = _strip_quotes(p["position"]).lower()
        if v not in _VALID_EMBED_POSITIONS:
            raise ValueError(f"embed.position={v!r}; expected {sorted(_VALID_EMBED_POSITIONS)}")
        out.position = v
    if "dropout" in p:
        out.dropout = float(p["dropout"])
    if "scale_by_dim" in p:
        out.scale_by_dim = _parse_bool(p["scale_by_dim"])
    return out


def _parse_norm(raw: str) -> NormConfig:
    body = _strip_braces(raw)
    p = _split_top_level_kv(body)
    out = NormConfig()
    for eq in _extract_all_equations(body):
        if eq in _EQUATION_TO_NORM_TYPE:
            out.type = _EQUATION_TO_NORM_TYPE[eq]
            break
    if "type" in p:
        v = _strip_quotes(p["type"]).lower()
        if v not in _VALID_NORM_TYPES:
            raise ValueError(f"norm.type={v!r}; expected {sorted(_VALID_NORM_TYPES)}")
        out.type = v
    if "placement" in p:
        v = _strip_quotes(p["placement"]).lower()
        if v not in _VALID_NORM_PLACEMENTS:
            raise ValueError(f"norm.placement={v!r}; expected {sorted(_VALID_NORM_PLACEMENTS)}")
        out.placement = v
    if "eps" in p:
        out.eps = float(p["eps"])
    return out


def _parse_output(raw: str) -> OutputConfig:
    p = _split_top_level_kv(_strip_braces(raw))
    out = OutputConfig()
    if "tie_embed" in p:
        out.tie_embed = _parse_bool(p["tie_embed"])
    if "bias" in p:
        out.bias = _parse_bool(p["bias"])
    return out


def _parse_dropout(raw: str) -> DropoutConfig:
    p = _split_top_level_kv(_strip_braces(raw))
    out = DropoutConfig()
    for fname in ("embed", "attn", "resid", "ffn", "path"):
        if fname in p:
            setattr(out, fname, float(p[fname]))
    return out


def _parse_init(raw: str) -> InitConfig:
    p = _split_top_level_kv(_strip_braces(raw))
    out = InitConfig()
    if "std" in p:
        out.std = float(p["std"])
    if "method" in p:
        v = _strip_quotes(p["method"]).lower()
        if v not in _VALID_INIT_METHODS:
            raise ValueError(f"init.method={v!r}; expected {sorted(_VALID_INIT_METHODS)}")
        out.method = v
    if "output_scale" in p:
        out.output_scale = _parse_bool(p["output_scale"])
    return out


def _parse_layers(raw: str) -> LayersConfig:
    p = _split_top_level_kv(_strip_braces(raw))
    out = LayersConfig()
    if "first_k_dense" in p:
        out.first_k_dense = int(p["first_k_dense"])
    if "window_size" in p:
        out.window_size = int(p["window_size"])
    return out


def _parse_diagnostic(raw: str) -> DiagnosticConfig:
    p = _split_top_level_kv(_strip_braces(raw))
    out = DiagnosticConfig()
    if "type" in p:
        v = _strip_quotes(p["type"]).lower()
        if v not in _VALID_DIAGNOSTIC_TYPES:
            raise ValueError(f"diagnostic.type={v!r}; expected {sorted(_VALID_DIAGNOSTIC_TYPES)}")
        out.type = v
    if "alpha" in p:
        out.alpha = float(p["alpha"])
    if "gamma" in p:
        out.gamma = float(p["gamma"])
    return out


def _parse_scheduler(raw: str) -> SchedulerConfig:
    p = _split_top_level_kv(_strip_braces(raw))
    out = SchedulerConfig()
    if "type" in p:
        v = _strip_quotes(p["type"]).lower()
        # normalise alias
        if v == "warmup_stable_decay":
            v = "wsd"
        if v not in _VALID_SCHEDULER_TYPES:
            raise ValueError(f"scheduler.type={v!r}; expected {sorted(_VALID_SCHEDULER_TYPES)}")
        out.type = v
    if "warmup_steps" in p:
        out.warmup_steps = int(p["warmup_steps"])
    if "min_lr_ratio" in p:
        out.min_lr_ratio = float(p["min_lr_ratio"])
    if "decay_steps" in p:
        out.decay_steps = int(p["decay_steps"])
    if "stable_ratio" in p:
        out.stable_ratio = float(p["stable_ratio"])
    return out


def _parse_optimizer(raw: str) -> OptimizerConfig:
    body = _strip_braces(raw)
    sched_body = _extract_block(body, "scheduler")
    p = _flat_kv_strip_nested(body, "scheduler")

    out = OptimizerConfig()
    if "type" in p:
        v = _strip_quotes(p["type"]).lower()
        if v not in _VALID_OPTIMIZER_TYPES:
            raise ValueError(f"optimizer.type={v!r}; expected {sorted(_VALID_OPTIMIZER_TYPES)}")
        out.type = v
    if "lr" in p:
        out.lr = float(p["lr"])
    if "betas" in p:
        raw_betas = p["betas"].strip().strip("[]")
        b1, b2 = [float(x.strip()) for x in raw_betas.split(",")]
        out.betas = (b1, b2)
    if "weight_decay" in p:
        out.weight_decay = float(p["weight_decay"])
    if "eps" in p:
        out.eps = float(p["eps"])
    if "grad_clip" in p:
        out.grad_clip = float(p["grad_clip"])
    if sched_body:
        out.scheduler = _parse_scheduler(sched_body)
    return out


# ── Sheaf parser ──────────────────────────────────────────────────────────────

_SHEAF_SUB_BLOCKS = (
    "embed", "coboundary", "transition", "norm", "output",
    "dropout", "init", "layers", "diagnostic",
)


def _parse_sheaf(raw: str, kind: Optional[str] = None) -> SheafConfig:
    body = _strip_braces(raw)
    out = SheafConfig()

    # Apply kind macro defaults
    if kind and kind in _KIND_DEFAULTS:
        defaults = _KIND_DEFAULTS[kind]
        for sub, fields in defaults.items():
            sub_obj = getattr(out, sub, None)
            if sub_obj is not None:
                for k, v in fields.items():
                    setattr(sub_obj, k, v)

    # Extract nested sub-blocks
    sub_bodies = {sb: _extract_block(body, sb) for sb in _SHEAF_SUB_BLOCKS}

    # Strip sub-blocks to get flat kv
    flat = body
    for sb in _SHEAF_SUB_BLOCKS:
        flat = _strip_nested_block(flat, sb)
    p = _split_top_level_kv(flat)

    # Scale hyperparameters
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

    # Explicit sub-block overrides (after kind macro defaults)
    if sub_bodies["embed"]:
        out.embed = _parse_embed(sub_bodies["embed"])
    if sub_bodies["coboundary"]:
        out.coboundary = _parse_coboundary(sub_bodies["coboundary"])
    if sub_bodies["transition"]:
        out.transition = _parse_transition(sub_bodies["transition"])
    if sub_bodies["norm"]:
        out.norm = _parse_norm(sub_bodies["norm"])
    if sub_bodies["output"]:
        out.output = _parse_output(sub_bodies["output"])
    if sub_bodies["dropout"]:
        out.dropout = _parse_dropout(sub_bodies["dropout"])
    if sub_bodies["init"]:
        out.init = _parse_init(sub_bodies["init"])
    if sub_bodies["layers"]:
        out.layers = _parse_layers(sub_bodies["layers"])
    if sub_bodies["diagnostic"]:
        out.diagnostic = _parse_diagnostic(sub_bodies["diagnostic"])

    return out


# ── Top-level parser ──────────────────────────────────────────────────────────

def parse_model_block(text: str) -> ModelSpec:
    """Parse a `model { ... }` DSL block (or full arch text).

    Supports both `block { }` and `block: { }` syntax (normalised first).
    """
    text = _strip_inline_comments(text.strip())
    text = _strip_import_lines(text)
    text = _normalize_block_syntax(text.strip())

    body = _extract_block(text, "model") if "model" in text and "{" in text \
        else _strip_braces(text)

    spec = ModelSpec()
    if not body.strip():
        return spec

    sheaf_body = _extract_block(body, "sheaf")
    optimizer_body = _extract_block(body, "optimizer")
    flat = _flat_kv_strip_nested(body, "sheaf", "optimizer")

    if "kind" in flat:
        kind = _strip_quotes(flat["kind"]).lower()
        if kind not in _VALID_KINDS:
            raise ValueError(
                f"model.kind={kind!r} is not valid; expected one of {sorted(_VALID_KINDS)}"
            )
        spec.kind = kind

    if "weights" in flat:
        spec.weights = _strip_quotes(flat["weights"])

    if sheaf_body:
        spec.sheaf = _parse_sheaf(sheaf_body, kind=spec.kind)
    elif spec.kind:
        spec.sheaf = _parse_sheaf("{}", kind=spec.kind)

    if optimizer_body:
        spec.optimizer = _parse_optimizer(optimizer_body)

    return spec
