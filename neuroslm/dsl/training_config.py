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
    """Pipeline-level config the BRIAN harness consumes."""
    loss_clipping: LossClippingConfig = field(default_factory=LossClippingConfig)
    quantization: QuantizationConfig = field(default_factory=QuantizationConfig)
    grad_accum: int = 1
    optimizer: str = "adamw"
    learning_rate: float = 3e-4
    weight_decay: float = 0.01
    grad_clip: float = 1.0
    label_smoothing: float = 0.0


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

def _extract_block(source: str, keyword: str) -> Optional[str]:
    """Find `<keyword> { ... }` at top level; return brace body or None."""
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
