# -*- coding: utf-8 -*-
"""`training { ... }` block parser — pipeline config from arch.neuro.

The training block lets an architecture declare *how it expects to be
trained* alongside *what it is*. The BRIAN harness reads the result and
applies the requested transformations on top of the bare DSL circuit.

Supported fields:
    loss_clipping  — per-sample Huber-style robust loss aggregation
    quantization   — int8/int4 post-training or QAT (PyTorch dynamic)
    grad_accum     — gradient accumulation steps
    optimizer      — adamw | adafactor (matches existing presets)
    learning_rate  — base LR
    weight_decay   — AdamW weight decay
    grad_clip      — global gradient norm clip
    label_smoothing — cross-entropy label smoothing
"""
from __future__ import annotations
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Optional


# ── Sub-config dataclasses ─────────────────────────────────────────────

@dataclass
class LossClippingConfig:
    """Per-sample loss clipping (p4 fix, declarative version).

    method = "per_sample":   each sequence's per-token loss is clipped at
                              `factor * batch_median` before averaging,
                              so one outlier sequence can't dominate the
                              gradient.
    factor = 3.0:             3× median is the GPT-3 / Cerebras default.
    """
    enabled: bool = False
    method: str = "per_sample"
    factor: float = 3.0


@dataclass
class QuantizationConfig:
    """Optional post-training or training-time quantization."""
    enabled: bool = False
    bits: int = 8


@dataclass
class TrainingConfig:
    """Pipeline-level config the BRIAN harness consumes.

    Defaults match Brain's `rcc_bowtie_30m_p4` so an arch.neuro with no
    `training {}` block trains under the same conditions as the reference
    Brain run (matching loss/ppl/lm trajectory bit-for-bit when the trunk
    is the bit-identical port). The five new fields (batch_size, seq_len,
    steps, warmup_steps, min_lr_ratio) move runtime hyperparameters into
    the architecture spec instead of leaving them in shell-script defaults.
    """
    loss_clipping: LossClippingConfig = field(default_factory=LossClippingConfig)
    quantization: QuantizationConfig = field(default_factory=QuantizationConfig)
    grad_accum: int = 1
    optimizer: str = "adamw"
    learning_rate: float = 3e-4
    weight_decay: float = 0.01
    grad_clip: float = 1.0
    label_smoothing: float = 0.0
    # Runtime hyperparameters — declarative form of the deploy script flags.
    # batch_size / seq_len affect tokens-per-step (Brain p4 uses 32 / 1024 →
    # 32k tokens/step; DSL must match for trajectory parity).
    batch_size: int = 4
    seq_len: int = 256
    steps: int = 10000
    warmup_steps: int = 300
    min_lr_ratio: float = 0.1
    # OOD-generalization knobs (added 2026-05-30 after first 10k run hit
    # gap_ratio 7.04). All default to 0 = off so prior runs reproduce.
    dropout: float = 0.0
    pct_strength: float = 0.0   # multiplies the PCH-aux weight in the trunk path
    # Forward-path PCT (Stage 1 of the OOD architecture push).
    # > 0 inserts a top-down prediction loop at every block boundary so
    # each layer is pulled toward what the next layer predicts. The
    # cited literature claims >=2x OOD-gap reduction vs the aux-only
    # PCH variant (`pct_strength`). Recommended start: 0.5.
    pct_trunk: float = 0.0
    # Stage 2 OOD push: Tonnetz toroidal attention mask. >0 enables the
    # mask with the given period (12 = musical octave is the standard
    # choice). The mask suppresses attention to positions outside a
    # circular-distance bandwidth + local window. 0 = standard causal
    # attention (no toroidal constraint).
    tonnetz_period: int = 0


# ── Constants for validation ───────────────────────────────────────────

_VALID_LOSS_METHODS = {"per_sample"}
_VALID_QUANT_BITS = {4, 8, 16}
_VALID_OPTIMIZERS = {"adamw", "adafactor"}


# ── Parser ─────────────────────────────────────────────────────────────

def parse_training_config(body: str) -> TrainingConfig:
    """Parse the body of a `training { ... }` block (without the braces).

    Empty body → all defaults. Unknown top-level keys are silently ignored
    (forward-compat with future versions of this schema). Invalid values
    for known keys raise ValueError.
    """
    cfg = TrainingConfig()

    props = _split_top_level_kv(body)

    # Sub-blocks
    if "loss_clipping" in props:
        cfg.loss_clipping = _parse_loss_clipping(props["loss_clipping"])
    if "quantization" in props:
        cfg.quantization = _parse_quantization(props["quantization"])

    # Scalars
    if "grad_accum" in props:
        cfg.grad_accum = int(props["grad_accum"])
    if "optimizer" in props:
        opt = _strip_quotes(props["optimizer"])
        if opt not in _VALID_OPTIMIZERS:
            raise ValueError(
                f"unknown optimizer {opt!r}; expected one of {sorted(_VALID_OPTIMIZERS)}"
            )
        cfg.optimizer = opt
    if "learning_rate" in props:
        cfg.learning_rate = float(props["learning_rate"])
    if "weight_decay" in props:
        cfg.weight_decay = float(props["weight_decay"])
    if "grad_clip" in props:
        cfg.grad_clip = float(props["grad_clip"])
    if "label_smoothing" in props:
        cfg.label_smoothing = float(props["label_smoothing"])
    # Runtime hyperparameters (deploy-script flags moved into the arch spec)
    if "batch_size" in props:
        cfg.batch_size = int(props["batch_size"])
    if "seq_len" in props:
        cfg.seq_len = int(props["seq_len"])
    if "steps" in props:
        cfg.steps = int(props["steps"])
    if "warmup_steps" in props:
        cfg.warmup_steps = int(props["warmup_steps"])
    if "min_lr_ratio" in props:
        cfg.min_lr_ratio = float(props["min_lr_ratio"])
    if "dropout" in props:
        cfg.dropout = float(props["dropout"])
    if "pct_strength" in props:
        cfg.pct_strength = float(props["pct_strength"])
    if "pct_trunk" in props:
        cfg.pct_trunk = float(props["pct_trunk"])
    if "tonnetz_period" in props:
        cfg.tonnetz_period = int(props["tonnetz_period"])

    return cfg


def _parse_loss_clipping(body: str) -> LossClippingConfig:
    body = _strip_braces(body)
    props = _split_top_level_kv(body)
    cfg = LossClippingConfig()
    if "enabled" in props:
        cfg.enabled = _parse_bool(props["enabled"])
    if "method" in props:
        method = _strip_quotes(props["method"])
        if method not in _VALID_LOSS_METHODS:
            raise ValueError(
                f"loss_clipping method {method!r}: expected one of {sorted(_VALID_LOSS_METHODS)}"
            )
        cfg.method = method
    if "factor" in props:
        cfg.factor = float(props["factor"])
    return cfg


def _parse_quantization(body: str) -> QuantizationConfig:
    body = _strip_braces(body)
    props = _split_top_level_kv(body)
    cfg = QuantizationConfig()
    if "enabled" in props:
        cfg.enabled = _parse_bool(props["enabled"])
    if "bits" in props:
        bits = int(props["bits"])
        if bits not in _VALID_QUANT_BITS:
            raise ValueError(
                f"quantization bits={bits}: expected one of {sorted(_VALID_QUANT_BITS)}"
            )
        cfg.bits = bits
    return cfg


# ── Loader: pull training config out of an architecture folder ─────────

def load_training_config_from_arch(arch_root) -> TrainingConfig:
    """Read arch.neuro and parse the `training { ... }` block if present.

    Returns defaults when:
      * no `training` block exists (most common)
      * arch.neuro doesn't exist (returns defaults so callers can degrade
        gracefully rather than crashing on a missing folder)
    """
    arch_path = Path(arch_root) / "arch.neuro"
    if not arch_path.is_file():
        return TrainingConfig()

    source = arch_path.read_text(encoding="utf-8")
    # Find `training { ... }` block, brace-aware
    body = _extract_block(source, "training")
    if body is None:
        return TrainingConfig()
    return parse_training_config(body)


# ── Helpers (mirror multifile.py's small parsers) ─────────────────────

def _strip_comments(source: str) -> str:
    """Replace `# ...` to end-of-line with spaces.

    Preserves line numbers and positions so other parsers see the same
    offsets, but removes comment characters that would otherwise confuse
    the brace/string walker (e.g. apostrophes in `Brain's reference run`
    were entering string mode and swallowing `{` / `}` counts).
    """
    out = []
    i, n = 0, len(source)
    in_str = None
    while i < n:
        ch = source[i]
        if in_str:
            out.append(ch)
            if ch == in_str and source[i - 1] != '\\':
                in_str = None
            i += 1
            continue
        if ch in ('"', "'"):
            in_str = ch
            out.append(ch)
            i += 1
            continue
        if ch == '#':
            # Skip to end of line, replacing with spaces (preserve newline)
            while i < n and source[i] != '\n':
                out.append(' ')
                i += 1
            continue
        out.append(ch)
        i += 1
    return ''.join(out)


def _extract_block(source: str, keyword: str) -> Optional[str]:
    """Find `<keyword> { ... }` at top level; return brace body or None."""
    # Strip comments first so apostrophes/braces inside `# ...` text can't
    # confuse the brace/string walker below.
    source = _strip_comments(source)
    pattern = re.compile(rf'\b{re.escape(keyword)}\s*\{{', re.MULTILINE)
    m = pattern.search(source)
    if not m:
        return None
    start = m.end() - 1  # position of `{`
    depth = 1
    i = start + 1
    in_str = None
    while i < len(source) and depth > 0:
        ch = source[i]
        if in_str:
            if ch == in_str:
                in_str = None
        elif ch in ('"', "'"):
            in_str = ch
        elif ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
        i += 1
    if depth != 0:
        raise ValueError(f"unbalanced braces in `{keyword}` block")
    return source[start + 1 : i - 1]


def _split_top_level_kv(body: str) -> Dict[str, str]:
    """Quote/paren/brace-aware key:value splitter."""
    out: Dict[str, str] = {}
    buf, depth, in_str = [], 0, None

    def flush() -> None:
        piece = "".join(buf).strip()
        if not piece or ":" not in piece:
            return
        k, v = piece.split(":", 1)
        out[k.strip()] = v.strip()

    for ch in body:
        if in_str:
            buf.append(ch)
            if ch == in_str:
                in_str = None
        elif ch in ('"', "'"):
            in_str = ch
            buf.append(ch)
        elif ch in "([{":
            depth += 1
            buf.append(ch)
        elif ch in ")]}":
            depth = max(0, depth - 1)
            buf.append(ch)
        elif (ch == "," or ch == "\n") and depth == 0:
            flush()
            buf = []
        else:
            buf.append(ch)
    flush()
    return out


def _strip_quotes(s: str) -> str:
    s = s.strip()
    if len(s) >= 2 and s[0] in ('"', "'") and s[-1] == s[0]:
        return s[1:-1]
    return s


def _strip_braces(s: str) -> str:
    s = s.strip()
    if s.startswith("{") and s.endswith("}"):
        return s[1:-1]
    return s


def _parse_bool(s: str) -> bool:
    return _strip_quotes(s).lower() in ("true", "yes", "1")
