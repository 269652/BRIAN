# -*- coding: utf-8 -*-
"""`param_scope { ... }` parser — declarative gradient isolation (p3 fix).

The p3 architectural fix ("parameter closure isolation") stopped bio-side
parameters from being mutated by the main LM loss. In the hand-written
`Brain` this was `partition_trunk_bio_params()`. In the DSL it becomes a
declaration in arch.neuro:

    param_scope trunk {
        populations: [sensory, thalamus, gws, pfc, motor]
    }
    param_scope bio {
        populations: [amygdala, hippo, vta],
        gradient: "detached_from_main_loss"
    }

The BRIANHarness reads these and freezes (`requires_grad=False`) the
parameters of populations in a `detached_from_main_loss` scope, so the
main loss can't update them. Auxiliary objectives (Phase D) re-enable
them selectively.
"""
from __future__ import annotations
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import List


_VALID_GRADIENT_POLICIES = {"normal", "detached_from_main_loss"}


@dataclass
class ParamScope:
    """A named group of populations with a gradient policy.

    gradient:
        "normal"                  — params train from the main loss (default)
        "detached_from_main_loss" — params frozen w.r.t. the main loss
                                     (the p3 isolation)
    """
    name: str
    populations: List[str] = field(default_factory=list)
    gradient: str = "normal"


def parse_param_scopes(source: str) -> List[ParamScope]:
    """Extract all `param_scope name { ... }` blocks from a source string."""
    scopes: List[ParamScope] = []
    for m in re.finditer(r'\bparam_scope\s+(\w+)\s*\{', source):
        name = m.group(1)
        body, _ = _slice_braced(source, m.end() - 1)
        props = _split_top_level_kv(body)

        pops_raw = props.get("populations", "")
        populations = _parse_list(pops_raw)

        gradient = "normal"
        if "gradient" in props:
            gradient = _strip_quotes(props["gradient"])
            if gradient not in _VALID_GRADIENT_POLICIES:
                raise ValueError(
                    f"param_scope {name!r}: invalid gradient policy "
                    f"{gradient!r}; expected one of {sorted(_VALID_GRADIENT_POLICIES)}"
                )

        scopes.append(ParamScope(name=name, populations=populations,
                                 gradient=gradient))
    return scopes


def load_param_scopes_from_arch(arch_root) -> List[ParamScope]:
    """Read arch.neuro and parse all param_scope blocks (empty if none)."""
    arch_path = Path(arch_root) / "arch.neuro"
    if not arch_path.is_file():
        return []
    return parse_param_scopes(arch_path.read_text(encoding="utf-8"))


# ── small parsing helpers (shared style with multifile.py) ────────────

def _slice_braced(body: str, start: int):
    assert body[start] == "{"
    depth, i, in_str = 1, start + 1, None
    while i < len(body) and depth > 0:
        ch = body[i]
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
    return body[start + 1 : i - 1], i


def _split_top_level_kv(body: str):
    out, buf, depth, in_str = {}, [], 0, None

    def flush():
        piece = "".join(buf).strip()
        if piece and ":" in piece:
            k, v = piece.split(":", 1)
            out[k.strip()] = v.strip()

    for ch in body:
        if in_str:
            buf.append(ch)
            if ch == in_str:
                in_str = None
        elif ch in ('"', "'"):
            in_str = ch; buf.append(ch)
        elif ch in "([{":
            depth += 1; buf.append(ch)
        elif ch in ")]}":
            depth = max(0, depth - 1); buf.append(ch)
        elif (ch == "," or ch == "\n") and depth == 0:
            flush(); buf = []
        else:
            buf.append(ch)
    flush()
    return out


def _parse_list(text: str) -> List[str]:
    text = text.strip()
    if text.startswith("[") and text.endswith("]"):
        text = text[1:-1]
    return [t.strip() for t in text.split(",") if t.strip()]


def _strip_quotes(s: str) -> str:
    s = s.strip()
    if len(s) >= 2 and s[0] in ('"', "'") and s[-1] == s[0]:
        return s[1:-1]
    return s
