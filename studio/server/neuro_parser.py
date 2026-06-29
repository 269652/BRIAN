# -*- coding: utf-8 -*-
"""Lightweight .neuro parser for Brian Studio.

Extracts visual structure (nodes + edges) from .neuro architecture files
without needing the full DSL compiler. Handles the model{}/sheaf{} format
used by GPT-2, LLaMA, Qwen and custom architectures.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

_LIB_ROOT = Path(__file__).parent.parent.parent / "neuroslm" / "dsl" / "lib"

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_KV_RE = re.compile(r"(\w+)\s*:\s*(\"[^\"]*\"|true|false|\d+(?:\.\d+)?(?:e[+-]?\d+)?|\w+)", re.IGNORECASE)


def _extract_block_content(src: str, block_name: str) -> str | None:
    """Brace-counting extractor — handles arbitrary nesting depth.

    Matches both ``name {`` and ``name: {`` syntax.
    """
    pattern = re.compile(r"\b" + re.escape(block_name) + r"\s*:?\s*\{", re.DOTALL)
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


_BLOCK_OPEN_RE = re.compile(r"(\w+)\s*:?\s*\{")


def _split_blocks(src: str) -> tuple[dict[str, str], str]:
    """Brace-aware split: return ({block_name: inner_body}, scalar_text).

    Walks the source character by character so nested blocks of *any* depth are
    extracted correctly (the old `_BLOCK_RE` only handled one level, which lost
    e.g. `model { sheaf { coboundary { rope {} } } }`).
    """
    blocks: dict[str, str] = {}
    scalar_parts: list[str] = []
    i = 0
    n = len(src)
    while i < n:
        m = _BLOCK_OPEN_RE.search(src, i)
        if not m:
            scalar_parts.append(src[i:])
            break
        # Text before this block contributes scalar key:value pairs.
        scalar_parts.append(src[i : m.start()])
        name = m.group(1)
        depth = 1
        j = m.end()
        while j < n and depth > 0:
            if src[j] == "{":
                depth += 1
            elif src[j] == "}":
                depth -= 1
            j += 1
        inner = src[m.end() : j - 1] if depth == 0 else src[m.end() : j]
        # Last block of a given name wins for dict storage; duplicates are rare
        # for blocks (scalars handle the equation-list case separately).
        blocks[name] = inner
        i = j
    return blocks, "".join(scalar_parts)


def _parse_block(src: str) -> dict[str, Any]:
    """Parse flat key: value pairs (and nested blocks) from a { ... } body."""
    out: dict[str, Any] = {}
    blocks, cleaned = _split_blocks(src)
    for name, inner in blocks.items():
        out[name] = _parse_block(inner)
    for m in _KV_RE.finditer(cleaned):
        key, val_raw = m.group(1), m.group(2)
        if val_raw.startswith('"') and val_raw.endswith('"'):
            val: Any = val_raw[1:-1]
        elif val_raw.lower() == "true":
            val = True
        elif val_raw.lower() == "false":
            val = False
        else:
            try:
                val = int(val_raw)
            except ValueError:
                try:
                    val = float(val_raw)
                except ValueError:
                    val = val_raw
        # Duplicate `equation:` keys → collect as list (handles coboundary with GQA+RoPE)
        if key == "equation" and key in out:
            existing = out[key]
            out[key] = (existing if isinstance(existing, list) else [existing]) + [val]
        else:
            out[key] = val
    return out


def _infer_kind(name: str, weights: str) -> str:
    """Derive model kind from arch name or weights string."""
    name_lower = name.lower()
    weights_lower = weights.lower()
    for k in _KIND_MECHANICS:
        if k in name_lower or k in weights_lower:
            return k
    return name  # use arch folder name as best guess


_IMPORT_RE = re.compile(
    r'import\s*\{([^}]+)\}\s*from\s*"@lib/([^"]+)"', re.MULTILINE
)
_TRIPLE_EQ_RE = re.compile(r'equation\s*:\s*"""(.*?)"""', re.DOTALL)
_SINGLE_EQ_RE = re.compile(r'(?<!\w)equation(?:_\w+)?\s*:\s*"([^"]+)"')


def _parse_equation_block(body: str) -> dict[str, Any]:
    """Extract math, where, properties from an exported equation body."""
    result: dict[str, Any] = {}
    m = _TRIPLE_EQ_RE.search(body)
    if m:
        result["equation_math"] = m.group(1).strip()
    else:
        s = _SINGLE_EQ_RE.search(body)
        if s:
            result["equation_math"] = s.group(1)
    where_body = _extract_block_content(body, "where")
    if where_body:
        result["where"] = _parse_block(where_body)
    props_body = _extract_block_content(body, "properties")
    if props_body:
        result["primitive_properties"] = _parse_block(props_body)
    return result


def _resolve_imports(source: str) -> dict[str, dict[str, Any]]:
    """Parse `import { X } from "@lib/..."` lines and return resolved equation defs.

    Returns: {equation_name: {equation_math, where, primitive_properties, lib_path}}
    """
    resolved: dict[str, dict[str, Any]] = {}
    for m in _IMPORT_RE.finditer(source):
        names = [n.strip() for n in m.group(1).split(",") if n.strip()]
        lib_rel = m.group(2)  # e.g. "primitives/attention"
        lib_file = _LIB_ROOT / (lib_rel + ".neuro")
        if not lib_file.exists():
            continue
        lib_src = re.sub(r"#[^\n]*", "", lib_file.read_text(encoding="utf-8"))
        for name in names:
            # Find: export equation NAME { ... }
            eq_body = _extract_block_content(lib_src, f"equation {name}")
            if eq_body:
                data = _parse_equation_block(eq_body)
                data["lib_path"] = f"@lib/{lib_rel}"
                resolved[name] = data
    return resolved


# ---------------------------------------------------------------------------
# Full transformer-graph expansion
# ---------------------------------------------------------------------------
#
# A declarative `model { sheaf { ... } }` block names the equations of a
# transformer but hides its actual forward-pass wiring (residual streams,
# the repeated block, the embedding sum, the tied head). This expansion
# makes every one of those mechanics a visible node, grouped into three
# colored panels: Embeddings → Transformer Block ×depth → Output Head.

def _norm_label(equation: str) -> str:
    return (equation or "norm").replace("_", " ")


def _eq_list(value: Any) -> list[str]:
    """Coerce an `equation:` field (str or list) into a list of names."""
    if isinstance(value, list):
        return [str(v) for v in value]
    if value:
        return [str(value)]
    return []


def _build_transformer_graph(
    nodes: list[dict],
    edges: list[dict],
    sheaf: dict[str, Any],
    resolved_imports: dict[str, dict[str, Any]],
) -> bool:
    """Expand a transformer sheaf into its full computation graph.

    Returns True when the graph was built (always, once called with a sheaf
    that has coboundary/transition).
    """
    ri = resolved_imports or {}
    depth = sheaf.get("depth", "?")

    embed = sheaf.get("embed") if isinstance(sheaf.get("embed"), dict) else {}
    cob = sheaf.get("coboundary") if isinstance(sheaf.get("coboundary"), dict) else {}
    trans = sheaf.get("transition") if isinstance(sheaf.get("transition"), dict) else {}
    norm = sheaf.get("norm") if isinstance(sheaf.get("norm"), dict) else {}
    out = sheaf.get("output") if isinstance(sheaf.get("output"), dict) else {}

    norm_eq = norm.get("equation") or norm.get("type") or "layer_norm"
    placement = norm.get("placement", "pre")
    pos_kind = str(embed.get("position", "learned"))

    def _primitive(eq: str) -> dict[str, Any]:
        return dict(ri.get(eq, {}))

    def child(nid, parent, x, y, label, category, sub, *, eqs=None):
        extra = {k: v for k, v in (sub or {}).items() if k != "equation"}
        prim = _primitive(eqs[0]) if eqs else {}
        d = {"label": label, "category": category, **extra, **prim}
        if eqs:
            d["impl"] = " + ".join(eqs)
            if len(eqs) > 1:
                d["equations"] = eqs
        nodes.append({
            "id": nid, "type": "mechanic", "parentId": parent, "extent": "parent",
            "data": d, "position": {"x": x, "y": y},
        })

    def group(nid, x, y, w, h, label, kind):
        nodes.append({
            "id": nid, "type": "group",
            "data": {"label": label, "kind": kind},
            "position": {"x": x, "y": y},
            "style": {"width": w, "height": h},
        })

    def edge(src, tgt, *, residual=False, label=""):
        e = {"id": f"{src}->{tgt}", "source": src, "target": tgt, "animated": False}
        if label:
            e["label"] = label
        if residual:
            e["style"] = {"stroke": "var(--text-muted)", "strokeDasharray": "5 4"}
            e["data"] = {"residual": True}
        edges.append(e)

    CW = 175  # child width slot
    # ── Embeddings panel ────────────────────────────────────────────────
    group("grp_embed", 320, 40, 210, 250, "Embeddings", "embed")
    child("tok_embed", "grp_embed", 15, 45, "token embedding", "embed",
          {"vocab": sheaf.get("vocab"), "dim": sheaf.get("dim")})
    edge("model", "tok_embed")
    has_pos = pos_kind == "learned"
    if has_pos:
        child("pos_embed", "grp_embed", 15, 110, "learned position", "embed",
              embed, eqs=_eq_list(embed.get("equation")) or ["learned_position_encoding"])
        edge("model", "pos_embed")
        child("embed_add", "grp_embed", 15, 175, "embed sum  ⊕", "residual", {})
        edge("tok_embed", "embed_add")
        edge("pos_embed", "embed_add")
        block_in = "embed_add"
    else:
        # RoPE / no absolute position: token embedding flows straight in;
        # position is injected inside attention (see attn node).
        child("pos_note", "grp_embed", 15, 110, f"position: {pos_kind}", "embed", {})
        block_in = "tok_embed"

    # ── Transformer block panel (repeated ×depth) ────────────────────────
    group("grp_block", 600, 20, 230, 470, f"Transformer Block  ×{depth}", "block")
    cob_eqs = _eq_list(cob.get("equation"))
    attn_label = " + ".join(e.replace("_", " ") for e in cob_eqs) or "attention"
    ff_eqs = _eq_list(trans.get("equation"))
    ff_label = " + ".join(e.replace("_", " ") for e in ff_eqs) or "feed-forward"

    child("ln_1", "grp_block", 15, 45, _norm_label(norm_eq), "norm", norm, eqs=[norm_eq])
    child("attn", "grp_block", 15, 115, attn_label, "attention", cob, eqs=cob_eqs)
    child("attn_resid", "grp_block", 15, 185, "residual  ⊕", "residual", {})
    child("ln_2", "grp_block", 15, 255, _norm_label(norm_eq), "norm", norm, eqs=[norm_eq])
    child("ffn", "grp_block", 15, 325, ff_label, "ffn", trans, eqs=ff_eqs)
    child("ffn_resid", "grp_block", 15, 395, "residual  ⊕", "residual", {})

    edge(block_in, "ln_1", label="pre-norm" if placement == "pre" else "")
    edge("ln_1", "attn")
    edge("attn", "attn_resid")
    edge(block_in, "attn_resid", residual=True)  # skip connection
    edge("attn_resid", "ln_2")
    edge("ln_2", "ffn")
    edge("ffn", "ffn_resid")
    edge("attn_resid", "ffn_resid", residual=True)  # skip connection

    # ── Output panel ─────────────────────────────────────────────────────
    group("grp_output", 900, 120, 210, 200, "Output Head", "output")
    final_norm = bool(norm.get("final")) or placement == "pre"
    if final_norm:
        child("final_ln", "grp_output", 15, 45, f"final {_norm_label(norm_eq)}", "norm",
              norm, eqs=[norm_eq])
        edge("ffn_resid", "final_ln")
        head_src = "final_ln"
    else:
        head_src = "ffn_resid"
    tie = bool(out.get("tie_embed"))
    head_label = "lm head (tied)" if tie else "lm head"
    child("lm_head", "grp_output", 15, 115, head_label, "output", out)
    edge(head_src, "lm_head")
    if tie:
        edge("tok_embed", "lm_head", residual=True, label="weight tie")

    return True


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

    # Resolve @lib/... imports so mechanics can show full equation definitions
    resolved_imports = _resolve_imports(source)

    # ------------------------------------------------------------------
    # Model block (may have 3+ levels: model → sheaf → embed/coboundary)
    # Use brace-counting so nesting depth doesn't limit us.
    # ------------------------------------------------------------------
    expanded = False
    model_body = _extract_block_content(stripped, "model")
    if model_body is not None:
        data = _parse_block(model_body)
        weights = data.get("weights", "")
        kind = data.get("kind") or _infer_kind(name, weights)
        sheaf_data = data.get("sheaf") if isinstance(data.get("sheaf"), dict) else {}

        # Model node carries the sheaf scalars (dim/depth/heads/...) so the
        # full graph below doesn't need a separate summary node.
        model_scalars = {k: v for k, v in (sheaf_data or {}).items()
                         if not isinstance(v, dict) and not isinstance(v, list)}
        nodes.append({
            "id": "model",
            "type": "model",
            "data": {
                "kind": kind,
                "weights": weights,
                "label": kind,
                **{k: v for k, v in data.items() if k not in ("kind", "weights", "sheaf")},
                **model_scalars,
            },
            "position": {"x": 40, "y": 220},
        })

        # Expand the declarative sheaf into the full transformer forward-pass
        # graph (embeddings → block ×depth → output head), grouped into panels.
        if sheaf_data and ("coboundary" in sheaf_data or "transition" in sheaf_data):
            expanded = _build_transformer_graph(nodes, edges, sheaf_data, resolved_imports)

    # ------------------------------------------------------------------
    # Top-level sheaf block (outside model, e.g. population-based archs)
    # ------------------------------------------------------------------
    if not expanded and not any(n["id"] == "sheaf" for n in nodes):
        sheaf_body = _extract_block_content(stripped, "sheaf")
        if sheaf_body is not None:
            sheaf_data = _parse_block(sheaf_body)
            connect = "model" if any(n["id"] == "model" for n in nodes) else None
            _add_sheaf_node(nodes, edges, sheaf_data, connect_from=connect)

    # ------------------------------------------------------------------
    # Training block
    # ------------------------------------------------------------------
    training_body = _extract_block_content(stripped, "training")
    if training_body is not None and not expanded:
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
    # Legacy fallback: flat mechanics (only when no full graph was built)
    # ------------------------------------------------------------------
    if not expanded:
        if not _extract_equation_mechanics(nodes, edges, resolved_imports):
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


def _extract_equation_mechanics(
    nodes: list[dict],
    edges: list[dict],
    resolved_imports: dict[str, dict[str, Any]] | None = None,
) -> bool:
    """Build mechanic nodes from equation: fields inside the sheaf sub-blocks.

    Returns True when at least one mechanic node was added this way.
    """
    sheaf_node = next((n for n in nodes if n["id"] == "sheaf"), None)
    if not sheaf_node:
        return False

    _SUB_CATEGORY = {
        "coboundary": "attention",
        "transition": "ffn",
        "norm": "norm",
        "embed": "position",
        "output": "output",
    }

    added = False
    x = 560
    y = 60
    y_step = 100
    i = 0
    ri = resolved_imports or {}

    sheaf_data = sheaf_node["data"]
    for sub_name, category in _SUB_CATEGORY.items():
        sub = sheaf_data.get(sub_name)
        if not isinstance(sub, dict):
            continue
        eq_raw = sub.get("equation", "")
        # May be a list when multiple `equation:` lines exist (e.g. GQA + RoPE in coboundary)
        equations: list[str] = eq_raw if isinstance(eq_raw, list) else ([eq_raw] if eq_raw else [])
        if not equations:
            continue
        extra = {k: v for k, v in sub.items() if k != "equation"}
        for eq_idx, equation in enumerate(equations):
            mid = f"eq_{sub_name}" if eq_idx == 0 else f"eq_{sub_name}_{eq_idx}"
            if any(n["id"] == mid for n in nodes):
                continue
            label = equation.replace("_", " ")
            primitive = ri.get(equation, {})
            nodes.append({
                "id": mid,
                "type": "mechanic",
                "data": {
                    "label": label,
                    "category": category,
                    "impl": equation,
                    **extra,
                    **primitive,
                },
                "position": {"x": x, "y": y + i * y_step},
            })
            edges.append({
                "id": f"sheaf-{mid}",
                "source": "sheaf",
                "target": mid,
                "animated": False,
            })
            added = True
            i += 1

    # Output block — show even without an equation field
    output = sheaf_data.get("output")
    if isinstance(output, dict) and output.get("tie_embed") and not any(n["id"] == "eq_output" for n in nodes):
        nodes.append({
            "id": "eq_output",
            "type": "structure",
            "data": {"label": "LM Head (tied)", "category": "output", "tie_embed": True},
            "position": {"x": x, "y": y + i * y_step},
        })
        edges.append({"id": "sheaf-eq_output", "source": "sheaf", "target": "eq_output", "animated": False})

    return added


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
            "id": f"{connect_from}-{mid}",
            "source": connect_from,
            "target": mid,
            "animated": False,
        })
