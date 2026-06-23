# -*- coding: utf-8 -*-
"""Parser for `mechanic NAME { ... }` blocks in .neuro DSL files.

A `mechanic` block is the canonical specification for a reusable neural
mechanism. It declares WHAT a mechanism computes (equation), HOW it is
implemented (impl), WHAT can be configured (params), and WHEN to use it
(when_to_use). The block is documentation-first but machine-readable.

Grammar::

    [export] mechanic <name> {
        category: "<string>"           # attention|regularizer|routing|...
        summary:  "<string>"           # one-line description
        equation: \"\"\"...\"\"\"      # formal math (multiline OK)
        impl:     "<dotted.path>"      # Python implementation class
        loss_fn:  "<dotted.path>"      # optional — auxiliary loss function
        zero_init: true|false          # ReZero convention
        params: {
            <name>: {
                default: <value>
                type: "<float|int|bool|str>"
                min: <number>          # optional
                max: <number>          # optional
                doc: "<string>"
            }
            ...
        }
        properties: { <name>: "<string>", ... }
        when_to_use: \"\"\"...\"\"\"   # guidance (multiline OK)
        not_for:    \"\"\"...\"\"\"    # counter-indications
        empirical_evidence: { source: "...", result: "..." }
        formal_proof: "<citation or file>"
        references: ["...", ...]
    }

Unknown fields are silently preserved in `extra` for forward-compat.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


# ── Helpers (subset of training_config utilities, reproduced here to avoid
#    circular import) ──────────────────────────────────────────────────────


def _strip_quotes(s: str) -> str:
    s = s.strip()
    if (s.startswith('"') and s.endswith('"')) or (
        s.startswith("'") and s.endswith("'")
    ):
        return s[1:-1]
    return s


def _strip_triple_quotes(s: str) -> str:
    """Strip triple-quote delimiters and normalise leading indent."""
    s = s.strip()
    for delim in ('"""', "'''"):
        if s.startswith(delim) and s.endswith(delim) and len(s) >= 6:
            inner = s[3:-3]
            # Strip uniform leading whitespace from each line.
            lines = inner.split("\n")
            # Remove blank head/tail lines.
            while lines and not lines[0].strip():
                lines.pop(0)
            while lines and not lines[-1].strip():
                lines.pop()
            # Detect minimum indent.
            min_indent = min(
                (len(l) - len(l.lstrip())) for l in lines if l.strip()
            ) if lines else 0
            return "\n".join(l[min_indent:] for l in lines)
    return _strip_quotes(s)


def _parse_bool(s: str) -> bool:
    s = s.strip().lower()
    if s in ("true", "1", "yes"):
        return True
    if s in ("false", "0", "no"):
        return False
    raise ValueError(f"cannot parse bool: {s!r}")


def _parse_list(s: str) -> List[str]:
    """Parse `[ "a", "b", ... ]` into a list of stripped strings.

    Splits on top-level commas only — commas inside a quoted string (e.g.
    a citation like "Ye, Dong, Jia et al.") are preserved.
    """
    s = s.strip()
    if s.startswith("[") and s.endswith("]"):
        s = s[1:-1].strip()
    if not s:
        return []
    parts: List[str] = []
    buf: List[str] = []
    quote = ""
    for ch in s:
        if quote:
            buf.append(ch)
            if ch == quote:
                quote = ""
        elif ch in ('"', "'"):
            quote = ch
            buf.append(ch)
        elif ch == ",":
            parts.append("".join(buf))
            buf = []
        else:
            buf.append(ch)
    parts.append("".join(buf))
    return [_strip_quotes(p.strip()) for p in parts if p.strip()]


def _split_top_level_kv(body: str) -> Dict[str, str]:
    """Split `key: value` pairs at the top brace level.

    Values can be:
     - bare scalars:  ``true``, ``1.0``, ``"text"``
     - braced blocks: ``{ ... }``
     - bracketed lists: ``[ ... ]``
     - triple-quoted strings: ``\"\"\"...\"\"\"``

    Returns a dict mapping key → raw value string (not yet typed).
    """
    result: Dict[str, str] = {}
    i = 0
    body = body.strip()
    n = len(body)

    while i < n:
        # Skip whitespace and commas
        while i < n and body[i] in " \t\n\r,":
            i += 1
        if i >= n:
            break

        # Skip line comments
        if body[i] == "#":
            while i < n and body[i] != "\n":
                i += 1
            continue

        # Read key (up to ':')
        key_start = i
        while i < n and body[i] not in ":\n#":
            i += 1
        if i >= n:
            break
        if body[i] == "#":
            while i < n and body[i] != "\n":
                i += 1
            continue
        key = body[key_start:i].strip()
        if not key:
            i += 1
            continue
        i += 1  # consume ':'

        # Skip whitespace
        while i < n and body[i] in " \t":
            i += 1
        if i >= n:
            break

        # Detect value type
        if i + 2 < n and body[i:i+3] in ('"""', "'''"):
            # Triple-quoted string
            delim = body[i:i+3]
            end = body.find(delim, i + 3)
            if end == -1:
                val = body[i:]
                i = n
            else:
                val = body[i:end+3]
                i = end + 3
        elif body[i] == "{":
            # Nested block — find matching }
            depth = 0
            start = i
            while i < n:
                if body[i] == "{":
                    depth += 1
                elif body[i] == "}":
                    depth -= 1
                    if depth == 0:
                        i += 1
                        break
                elif body[i] == '"':
                    i += 1
                    while i < n and body[i] != '"':
                        if body[i] == "\\" :
                            i += 1
                        i += 1
                i += 1
            val = body[start:i].strip()
        elif body[i] == "[":
            # List — find matching ]
            depth = 0
            start = i
            while i < n:
                if body[i] == "[":
                    depth += 1
                elif body[i] == "]":
                    depth -= 1
                    if depth == 0:
                        i += 1
                        break
                elif body[i] == '"':
                    i += 1
                    while i < n and body[i] != '"':
                        i += 1
                i += 1
            val = body[start:i].strip()
        else:
            # Bare scalar: read to end of line or comma
            start = i
            while i < n and body[i] not in "\n,#":
                i += 1
            val = body[start:i].strip()

        if key:
            result[key] = val

    return result


def _strip_braces(s: str) -> str:
    s = s.strip()
    if s.startswith("{") and s.endswith("}"):
        return s[1:-1]
    return s


# ── Dataclasses ──────────────────────────────────────────────────────────


@dataclass
class ParamSpec:
    """Specification for a single configurable parameter of a mechanic."""
    name: str = ""
    default: Any = None
    type_hint: str = "float"
    min_val: Optional[float] = None
    max_val: Optional[float] = None
    exclusive_min: bool = False
    doc: str = ""


@dataclass
class MechanicSpec:
    """Parsed representation of a `mechanic NAME { ... }` block.

    This is the canonical specification object for a reusable neural
    mechanism. It is returned by `parse_mechanic_file` and can be
    consumed by:
      - Documentation generators (produce docs/mechanics/*.md)
      - Compiler / codegen (wire up impl classes)
      - Validators (check param constraints)
      - Arch analysers (enumerate active mechanics)
    """
    name: str = ""
    category: str = ""
    summary: str = ""
    equation: str = ""
    impl: str = ""
    loss_fn: str = ""
    zero_init: bool = False
    params: Dict[str, ParamSpec] = field(default_factory=dict)
    properties: Dict[str, str] = field(default_factory=dict)
    when_to_use: str = ""
    not_for: str = ""
    empirical_evidence: Dict[str, str] = field(default_factory=dict)
    formal_proof: str = ""
    references: List[str] = field(default_factory=list)
    exported: bool = False
    # Catch-all for forward-compat.
    extra: Dict[str, str] = field(default_factory=dict)


# ── Parser ───────────────────────────────────────────────────────────────

# Matches:  [export ]mechanic NAME {
_MECHANIC_RE = re.compile(
    r"(?:^|\n)\s*(export\s+)?mechanic\s+(\w+)\s*\{",
    re.MULTILINE,
)


def parse_mechanic_file(source: str) -> List[MechanicSpec]:
    """Parse all `mechanic NAME { ... }` blocks from a .neuro source string.

    Returns a list of MechanicSpec objects (one per block). Order matches
    declaration order in the source. Unknown fields are silently stored in
    `MechanicSpec.extra`.
    """
    specs: List[MechanicSpec] = []
    for m in _MECHANIC_RE.finditer(source):
        exported = bool(m.group(1))
        name = m.group(2)
        brace_start = m.end()  # position of char AFTER the opening '{'

        # Find the matching closing '}'
        depth = 1
        i = brace_start
        in_triple = False
        triple_delim = ""
        while i < len(source) and depth > 0:
            c = source[i]
            # Triple-quote detection
            if not in_triple and i + 2 < len(source) and source[i:i+3] in ('"""', "'''"):
                in_triple = True
                triple_delim = source[i:i+3]
                i += 3
                continue
            if in_triple:
                if source[i:i+3] == triple_delim:
                    in_triple = False
                    i += 3
                    continue
                i += 1
                continue
            # Regular depth counting
            if c == "{":
                depth += 1
            elif c == "}":
                depth -= 1
            i += 1

        body = source[brace_start:i-1]  # content between the outer braces
        spec = _parse_mechanic_body(name, body, exported)
        specs.append(spec)

    return specs


def _parse_mechanic_body(name: str, body: str, exported: bool) -> MechanicSpec:
    """Parse the body (braces stripped) of a mechanic block."""
    spec = MechanicSpec(name=name, exported=exported)
    props = _split_top_level_kv(body)

    KNOWN = {
        "category", "summary", "equation", "impl", "loss_fn",
        "zero_init", "params", "properties", "when_to_use", "not_for",
        "empirical_evidence", "formal_proof", "references",
    }

    for key, raw in props.items():
        if key == "category":
            spec.category = _strip_quotes(raw)
        elif key == "summary":
            spec.summary = _strip_triple_quotes(raw)
        elif key == "equation":
            spec.equation = _strip_triple_quotes(raw)
        elif key == "impl":
            spec.impl = _strip_quotes(raw)
        elif key == "loss_fn":
            spec.loss_fn = _strip_quotes(raw)
        elif key == "zero_init":
            spec.zero_init = _parse_bool(raw)
        elif key == "params":
            spec.params = _parse_params_block(_strip_braces(raw))
        elif key == "properties":
            spec.properties = _parse_str_dict(_strip_braces(raw))
        elif key == "when_to_use":
            spec.when_to_use = _strip_triple_quotes(raw)
        elif key == "not_for":
            spec.not_for = _strip_triple_quotes(raw)
        elif key == "empirical_evidence":
            spec.empirical_evidence = _parse_str_dict(_strip_braces(raw))
        elif key == "formal_proof":
            spec.formal_proof = _strip_quotes(raw)
        elif key == "references":
            spec.references = _parse_list(raw)
        else:
            spec.extra[key] = raw

    return spec


def _parse_params_block(body: str) -> Dict[str, ParamSpec]:
    """Parse `params: { name: { default: v, type: t, doc: d }, ... }`."""
    result: Dict[str, ParamSpec] = {}
    top = _split_top_level_kv(body)
    for pname, praw in top.items():
        ps = ParamSpec(name=pname)
        inner = _split_top_level_kv(_strip_braces(praw))
        for k, v in inner.items():
            if k == "default":
                # Try to type-coerce
                v = v.strip()
                try:
                    ps.default = int(v) if "." not in v else float(v)
                except ValueError:
                    ps.default = _strip_quotes(v)
            elif k == "type":
                ps.type_hint = _strip_quotes(v)
            elif k == "min":
                ps.min_val = float(v.strip())
            elif k == "max":
                ps.max_val = float(v.strip())
            elif k == "exclusive_min":
                ps.exclusive_min = _parse_bool(v)
            elif k == "doc":
                ps.doc = _strip_quotes(v)
        result[pname] = ps
    return result


def _parse_str_dict(body: str) -> Dict[str, str]:
    """Parse a flat `{ key: "value", ... }` into a str→str dict."""
    d: Dict[str, str] = {}
    top = _split_top_level_kv(body)
    for k, v in top.items():
        d[k] = _strip_triple_quotes(v)
    return d
