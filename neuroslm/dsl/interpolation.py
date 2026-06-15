# -*- coding: utf-8 -*-
"""``%key%`` / ``$ENV`` interpolation for the NeuroSLM DSL.

Substitution surface
====================

* ``%key%``  resolves against an explicit ``config`` dict.
* ``$NAME``  resolves against ``os.environ`` (or an explicit ``env``
  mapping, which shadows the OS).

Both forms are recursive: substituted text is rescanned until no
``%...%`` or ``$NAME`` remains, with a hard ``max_depth`` ceiling to
catch cycles.

Why a separate module
=====================

The compiler called ``_parse_properties`` on every block body before
this lifted parameterised values into a first-class concern. Folding
the substitution into ``_parse_properties`` would have made every
existing call site implicitly require a config dict; keeping it as a
standalone pre-pass means callers opt in (``compile_with_lib`` does;
``compile`` does not), and existing tests stay green.

Standalone, no torch, no heavy imports — parse-time cost is one
regex pass per property string.

Pinned by ``tests/dsl/test_interpolation.py``.
"""
from __future__ import annotations

import os
import re
from typing import Any, Mapping, Optional


__all__ = [
    "InterpolationError",
    "resolve_interpolation",
    "_CONFIG_RE",
    "_ENV_RE",
]


class InterpolationError(ValueError):
    """Raised when a ``%key%`` or ``$NAME`` cannot be resolved.

    Subclasses :class:`ValueError` so existing ``except ValueError``
    blocks in the compiler keep catching parse errors uniformly.
    """


# ``%alphanumeric_with_underscores%`` — bounded by ``%`` on both
# sides. The inner regex permits digits and underscores but not
# whitespace, so a stray ``50%`` followed by another ``%`` won't
# accidentally swallow surrounding text. Identifiers must start
# with a letter or underscore (NOT a digit) to avoid lexing
# percentage-style numbers like ``42%foo%``.
_CONFIG_RE = re.compile(r"%([A-Za-z_][A-Za-z0-9_]*)%")

# ``$NAME`` — env variable. Same identifier rules: must start with a
# letter or underscore so ``$1foo`` (regex capture group) is left
# alone. Trailing ``$`` or ``$<digit>...`` are pass-through.
_ENV_RE = re.compile(r"\$([A-Za-z_][A-Za-z0-9_]*)")


def resolve_interpolation(
    text: str,
    *,
    config: Optional[Mapping[str, Any]] = None,
    env: Optional[Mapping[str, str]] = None,
    max_depth: int = 8,
) -> str:
    """Resolve every ``%key%`` and ``$NAME`` in ``text``.

    Args:
      text: source text to scan. Returned unchanged if it contains
        neither form (zero-cost for legacy arch.neuro inputs).
      config: mapping for ``%key%`` lookups. ``None`` ⇒ any ``%key%``
        token raises :class:`InterpolationError`.
      env: mapping for ``$NAME`` lookups. ``None`` ⇒ falls back to
        ``os.environ``. Passing an explicit mapping shadows the OS,
        which lets tests pin env-driven config without ``monkeypatch``.
      max_depth: hard ceiling on recursive expansion passes. Default
        8 is comfortably above any real config-of-configs the user
        will ever write; a cycle hits the ceiling and raises
        :class:`InterpolationError`.

    Returns:
      ``text`` with every ``%key%`` and ``$NAME`` replaced by the
      stringified value from its source. ``None`` config + no
      ``%`` / ``$`` in input ⇒ exact original returned (identity
      pass-through for the back-compat path).

    Raises:
      :class:`InterpolationError` if any token can't be resolved or
      expansion exceeds ``max_depth`` (cycle).
    """
    if not text:
        return text

    # Fast path: nothing to substitute.
    if "%" not in text and "$" not in text:
        return text

    # Use os.environ as the env fallback so callers don't have to
    # spell it out. Tests can pass env={...} to fully shadow it.
    env_map: Mapping[str, str] = (
        env if env is not None else os.environ
    )

    cfg = config or {}
    current = text
    for depth in range(max_depth + 1):
        # Substitute %config% tokens first, then $env. Order matters
        # only when a config value itself contains $env — the next
        # pass picks that up.
        def _cfg_sub(m: re.Match) -> str:
            key = m.group(1)
            if key not in cfg:
                raise InterpolationError(
                    f"unknown config key {key!r}: missing from "
                    f"interpolation config (available: "
                    f"{sorted(cfg) if cfg else '<none>'})"
                )
            return str(cfg[key])

        def _env_sub(m: re.Match) -> str:
            name = m.group(1)
            if name not in env_map:
                raise InterpolationError(
                    f"env var ${name} is not set in the resolution "
                    "environment"
                )
            return str(env_map[name])

        next_pass = _CONFIG_RE.sub(_cfg_sub, current)
        next_pass = _ENV_RE.sub(_env_sub, next_pass)
        if next_pass == current:
            # Fixed point reached — no further substitutions possible.
            return next_pass
        current = next_pass

    # Burned through max_depth passes — give one final substitution
    # pass a chance to converge so callers can spend their depth
    # budget on actual indirection, not on the convergence check.
    def _cfg_sub_final(m: re.Match) -> str:
        key = m.group(1)
        if key not in cfg:
            raise InterpolationError(
                f"unknown config key {key!r}: missing from "
                f"interpolation config (available: "
                f"{sorted(cfg) if cfg else '<none>'})"
            )
        return str(cfg[key])

    def _env_sub_final(m: re.Match) -> str:
        name = m.group(1)
        if name not in env_map:
            raise InterpolationError(
                f"env var ${name} is not set in the resolution "
                "environment"
            )
        return str(env_map[name])

    final = _CONFIG_RE.sub(_cfg_sub_final, current)
    final = _ENV_RE.sub(_env_sub_final, final)
    if final == current:
        return final

    # Still not converged — genuine cycle or chain too deep.
    raise InterpolationError(
        f"interpolation did not converge after {max_depth} passes "
        f"(cycle or chain too deep). Last expansion: {final!r}"
    )
