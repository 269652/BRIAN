# -*- coding: utf-8 -*-
"""Lightweight .neuro parser for Brian Studio.

Extracts visual structure (nodes + edges) from .neuro architecture files
without needing the full DSL compiler. Handles the model{}/sheaf{} format
used by GPT-2, LLaMA, Qwen and custom architectures.
"""

from __future__ import annotations

import re
from typing import Any

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_BLOCK_RE = re.compile(r"(\w+)\s*\{([^{}]*(?:\{[^{}]*\}[^{}]*)*)\}", re.DOTALL)
_KV_RE = re.compile(r"(\w+)\s*:\s*(\"[^\"]*\"|true|false|\d+(?:\.\d+)?(?:e[+-]?\d+)?|\w+)", re.IGNORECASE)


def _extract_block_content(src: str, block_name: str) -> str | None:
    """Brace-counting extractor — handles arbitrary nesting depth."""
    pattern = re.compile(r"\b" + re.escape(block_name) + r"\s*\{", re.DOTALL)
    m = pattern.search(src)
    if not m:
        return None
    start = m.end()  # position right after the opening '{'
    depth = 1
    i = start
    while i < len(src) and depth > 0:
        if src[i] == "{":
            depth += 1
        elif src[i] == "}":
            depth -= 1
        i += 1
    return src[start : i - 1] if depth == 0 else None


def _parse_block(src: str) -> dict[str, Any]:
    """Parse flat key: value pairs (and nested blocks) from a { ... } body."""
    out: dict[str, Any] = {}
    # Nested blocks first (handles 1 level of sub-nesting)
    for m in _BLOCK_RE.finditer(src):
        out[m.group(1)] = _parse_block(m.group(2))
    # Then scalar key:value pairs (skip already-consumed block regions)
    cleaned = _BLOCK_RE.sub("", src)
    for m in _KV_RE.finditer(cleaned):
        key, val_raw = m.group(1), m.group(2)
        if val_raw.startswith('"') and val_raw.endswith('"'):
            out[key] = val_raw[1:-1]
        elif val_raw.lower() == "true":
            out[key] = True
        elif val_raw.lower() == "false":
            out[key] = False
        else:
            try:
                out[key] = int(val_raw)
            except ValueError:
                try:
                    out[key] = float(val_raw)
                except ValueError:
                    out[key] = val_raw
    return out


def _infer_kind(name: str, weights: str) -> str:
    """Derive model kind from arch name or weights string."""
    name_lower = name.lower()
    weights_lower = weights.lower()
    for k in _KIND_MECHANICS:
        if k in name_lower or k in weights_lower:
            return k
    return name  # use arch folder name as best guess


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def parse_arch(source: str, name: str) -> dict[str, Any]:
    """Parse a .neuro architecture source into React Flow nodes + edges.

    Returns::

        {
          "name": str,
          "source": str,
          "nodes": [{"id", "type", "data", "position"}, ...],
          "edges": [{"id", "source", "target"}, ...],
        }
    """
    nodes: list[dict] = []
    edges: list[dict] = []

    # Strip line comments so they don't confuse the block parser
    stripped = re.sub(r"#[^\n]*", "", source)

    # ------------------------------------------------------------------
    # Model block (may have 3+ levels: model → sheaf → embed/coboundary)
    # Use brace-counting so nesting depth doesn't limit us.
    # ------------------------------------------------------------------
    model_body = _extract_block_content(stripped, "model")
    if model_body is not None:
        data = _parse_block(model_body)
        weights = data.get("weights", "")
        kind = data.get("kind") or _infer_kind(name, weights)

        nodes.append({
            "id": "model",
            "type": "model",
            "data": {
                "kind": kind,
                "weights": weights,
                "label": kind,
                **{k: v for k, v in data.items() if k not in ("kind", "weights", "sheaf")},
            },
            "position": {"x": 80, "y": 120},
        })

        # Sheaf nested inside model block
        if "sheaf" in data and isinstance(data["sheaf"], dict):
            sheaf_data = data["sheaf"]
            _add_sheaf_node(nodes, edges, sheaf_data, connect_from="model")

    # ------------------------------------------------------------------
    # Top-level sheaf block (outside model, e.g. population-based archs)
    # ------------------------------------------------------------------
    if not any(n["id"] == "sheaf" for n in nodes):
        sheaf_body = _extract_block_content(stripped, "sheaf")
        if sheaf_body is not None:
            sheaf_data = _parse_block(sheaf_body)
            connect = "model" if any(n["id"] == "model" for n in nodes) else None
            _add_sheaf_node(nodes, edges, sheaf_data, connect_from=connect)

    # ------------------------------------------------------------------
    # Training block
    # ------------------------------------------------------------------
    training_body = _extract_block_content(stripped, "training")
    if training_body is not None:
        tdata = _parse_block(training_body)
        nodes.append({
            "id": "training",
            "type": "dynamic",
            "data": {"label": "training", **tdata},
            "position": {"x": 560, "y": 200},
        })
        connect_src = "sheaf" if any(n["id"] == "sheaf" for n in nodes) else "model"
        if any(n["id"] == connect_src for n in nodes):
            edges.append({"id": f"{connect_src}-training", "source": connect_src, "target": "training", "animated": False})

    # ------------------------------------------------------------------
    # Infer mechanics from model kind
    # ------------------------------------------------------------------
    _infer_kind_mechanics(nodes, edges)

    # Fallback model node if nothing was parsed
    if not any(n["id"] == "model" for n in nodes):
        nodes.insert(0, {
            "id": "model",
            "type": "model",
            "data": {"kind": name, "weights": "", "label": name},
            "position": {"x": 80, "y": 120},
        })

    return {"name": name, "source": source, "nodes": nodes, "edges": edges}


def _add_sheaf_node(
    nodes: list[dict],
    edges: list[dict],
    sheaf_data: dict[str, Any],
    connect_from: str | None,
) -> None:
    norm = sheaf_data.get("norm", {})
    if isinstance(norm, dict):
        # Accept both `type:` (legacy) and `equation:` (current DSL)
        norm_name = norm.get("type") or norm.get("equation", "transformer")
    else:
        norm_name = str(norm)
    nodes.append({
        "id": "sheaf",
        "type": "sheaf",
        "data": {
            "label": f"sheaf  dim={sheaf_data.get('dim', '?')}  depth={sheaf_data.get('depth', '?')}",
            "norm_name": norm_name,
            **sheaf_data,
        },
        "position": {"x": 320, "y": 120},
    })
    if connect_from and any(n["id"] == connect_from for n in nodes):
        edges.append({
            "id": f"{connect_from}-sheaf",
            "source": connect_from,
            "target": "sheaf",
            "animated": False,
        })


_KIND_MECHANICS: dict[str, list[tuple[str, str, dict]]] = {
    "gpt2": [
        ("norm_layernorm", "structure", {"label": "LayerNorm x12", "category": "norm", "impl": "layernorm"}),
        ("ffn_gelu", "mechanic", {"label": "FFN + GELU", "category": "ffn", "impl": "gelu"}),
        ("pos_learned", "mechanic", {"label": "Learned Pos Emb", "category": "position", "impl": "learned_pos"}),
        ("attn_mha", "mechanic", {"label": "Multi-Head Attn x12", "category": "attention", "impl": "mha"}),
    ],
    "llama": [
        ("norm_rms", "structure", {"label": "RMSNorm", "category": "norm", "impl": "rmsnorm"}),
        ("ffn_swiglu", "mechanic", {"label": "SwiGLU FFN", "category": "ffn", "impl": "swiglu"}),
        ("pos_rope", "mechanic", {"label": "RoPE", "category": "position", "impl": "rope"}),
        ("attn_gqa", "mechanic", {"label": "GQA", "category": "attention", "impl": "gqa"}),
    ],
    "qwen2": [
        ("norm_rms", "structure", {"label": "RMSNorm", "category": "norm", "impl": "rmsnorm"}),
        ("ffn_swiglu", "mechanic", {"label": "SwiGLU FFN", "category": "ffn", "impl": "swiglu"}),
        ("pos_rope", "mechanic", {"label": "RoPE", "category": "position", "impl": "rope"}),
        ("attn_gqa", "mechanic", {"label": "GQA", "category": "attention", "impl": "gqa"}),
    ],
}


def _infer_kind_mechanics(nodes: list[dict], edges: list[dict]) -> None:
    """Add mechanic nodes inferred from model kind (GPT-2, LLaMA, etc.)."""
    model_node = next((n for n in nodes if n["id"] == "model"), None)
    if not model_node:
        return
    kind = model_node["data"].get("kind", "")
    mechanics = _KIND_MECHANICS.get(kind, [])
    sheaf_exists = any(n["id"] == "sheaf" for n in nodes)
    connect_from = "sheaf" if sheaf_exists else "model"

    x_start = 560
    y_start = 60
    y_step = 100

    for i, (mid, mtype, mdata) in enumerate(mechanics):
        if any(n["id"] == mid for n in nodes):
            continue
        nodes.append({
            "id": mid,
            "type": mtype,
            "data": mdata,
            "position": {"x": x_start, "y": y_start + i * y_step},
        })
        edges.append({
            "id": f"{