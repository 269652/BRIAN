# -*- coding: utf-8 -*-
"""Contracts for the ``%key%`` and ``$ENV`` interpolation engine.

Why a separate module
=====================

Before the DSL had any way to parameterise a reusable library file.
``lib/equations.neuro`` could declare ``standard_synapse`` once, but
every arch using it had to inline its own hyperparameters next to the
``feature`` block — no way to write ``temperature: %cfd_temperature%``
in the lib and have the arch.neuro author supply the value.

This module pins the parse-time substitution contract:

* ``%key%``  resolves against an explicit ``config`` dict passed by
  the caller. Missing key ⇒ :class:`InterpolationError`.
* ``$NAME``  resolves against ``os.environ``. Unset ⇒
  :class:`InterpolationError`. Lets a deployment override a value
  without editing the arch.
* Interpolation is **recursive** (max-depth=8): a config value
  containing ``%other_key%`` or ``$ENV`` is re-resolved until a fixed
  point. Cycles raise :class:`InterpolationError` (with the cycle
  trail in the message).
* Interpolation is **non-destructive** for sources that don't use
  either form: callers can pipe every legacy arch.neuro through
  :func:`resolve_interpolation` unchanged.
* The engine is **standalone** — no torch, no heavy imports — so
  parse-time costs stay tiny and the compiler can call it on every
  property string.

These contracts are pinned here so future refactors don't quietly
widen the substitution surface (e.g. into identifiers, which would
break the LL(1) grammar).
"""
from __future__ import annotations

import pytest


# ─────────────────────────────────────────────────────────────────────
# Contract A — literal pass-through
# ─────────────────────────────────────────────────────────────────────


class TestPassThrough:

    def test_plain_text_returns_unchanged(self):
        from neuroslm.dsl.interpolation import resolve_interpolation
        s = 'population p1 { count: 16, dynamics: "rate_code" }'
        assert resolve_interpolation(s) == s, (
            "sources without %key% or $ENV must pass through unchanged "
            "so every legacy arch.neuro still parses bit-for-bit")

    def test_empty_string_returns_empty(self):
        from neuroslm.dsl.interpolation import resolve_interpolation
        assert resolve_interpolation("") == ""

    def test_percent_in_quoted_literal_is_NOT_substituted_when_no_match(self):
        """``%abc%`` with no matching config key must raise — even
        inside what looks like a quoted literal. The substitution
        engine is text-level (it doesn't try to lex strings), so the
        only way to write a literal ``%`` is to omit it entirely or
        ensure no config key matches the surrounding text."""
        from neuroslm.dsl.interpolation import (
            resolve_interpolation, InterpolationError,
        )
        with pytest.raises(InterpolationError, match="unknown"):
            resolve_interpolation('msg = "100%abc%done"')


# ─────────────────────────────────────────────────────────────────────
# Contract B — %key% substitution from config dict
# ─────────────────────────────────────────────────────────────────────


class TestConfigSubstitution:

    def test_single_key_substitution(self):
        from neuroslm.dsl.interpolation import resolve_interpolation
        out = resolve_interpolation(
            "temperature: %temp%",
            config={"temp": "4.0"},
        )
        assert out == "temperature: 4.0"

    def test_multiple_keys_in_one_string(self):
        from neuroslm.dsl.interpolation import resolve_interpolation
        out = resolve_interpolation(
            "T: %temp%, alpha: %alpha%",
            config={"temp": "4.0", "alpha": "0.7"},
        )
        assert out == "T: 4.0, alpha: 0.7"

    def test_int_value_substitutes_as_string(self):
        from neuroslm.dsl.interpolation import resolve_interpolation
        out = resolve_interpolation(
            "warmup: %steps%",
            config={"steps": 10000},
        )
        # config values are coerced to str at substitution time
        assert out == "warmup: 10000"

    def test_missing_key_raises_with_key_name_in_message(self):
        from neuroslm.dsl.interpolation import (
            resolve_interpolation, InterpolationError,
        )
        with pytest.raises(InterpolationError, match="missing_key"):
            resolve_interpolation("x: %missing_key%", config={"other": "v"})

    def test_no_config_provided_AND_uses_key_raises(self):
        """A source that uses ``%key%`` without any config passed in
        must raise (so it can never silently slip through as a literal
        ``%key%`` token in downstream parsers)."""
        from neuroslm.dsl.interpolation import (
            resolve_interpolation, InterpolationError,
        )
        with pytest.raises(InterpolationError, match="missing"):
            resolve_interpolation("x: %missing%")


# ─────────────────────────────────────────────────────────────────────
# Contract C — $ENV substitution from os.environ
# ─────────────────────────────────────────────────────────────────────


class TestEnvSubstitution:

    def test_env_var_substitution(self, monkeypatch):
        from neuroslm.dsl.interpolation import resolve_interpolation
        monkeypatch.setenv("BRIAN_TEST_TEMP", "8.0")
        out = resolve_interpolation("T: $BRIAN_TEST_TEMP")
        assert out == "T: 8.0"

    def test_explicit_env_mapping_overrides_os_environ(self, monkeypatch):
        from neuroslm.dsl.interpolation import resolve_interpolation
        monkeypatch.setenv("BRIAN_OVERRIDE", "from_os")
        out = resolve_interpolation(
            "v: $BRIAN_OVERRIDE",
            env={"BRIAN_OVERRIDE": "explicit"},
        )
        assert out == "v: explicit", (
            "an explicit env mapping must shadow os.environ so tests "
            "can pin env-driven config without touching the real env")

    def test_missing_env_var_raises_with_var_name(self):
        from neuroslm.dsl.interpolation import (
            resolve_interpolation, InterpolationError,
        )
        with pytest.raises(
            InterpolationError, match="BRIAN_DEFINITELY_UNSET",
        ):
            resolve_interpolation(
                "v: $BRIAN_DEFINITELY_UNSET", env={},
            )

    def test_env_name_must_be_uppercase_or_underscore_start(self):
        """``$1foo`` is NOT a valid env name — it must start with a
        letter or underscore. Anything else is left as a literal
        ``$1foo`` so YAML-style ``$1`` capture groups in regex
        literals don't accidentally trip the resolver."""
        from neuroslm.dsl.interpolation import resolve_interpolation
        out = resolve_interpolation("pattern: $1foo$2bar", env={})
        # Both must remain literal — they start with digits, not letters.
        assert out == "pattern: $1foo$2bar"


# ─────────────────────────────────────────────────────────────────────
# Contract D — recursion + cycle detection
# ─────────────────────────────────────────────────────────────────────


class TestRecursion:

    def test_config_value_containing_another_key_resolves_recursively(self):
        from neuroslm.dsl.interpolation import resolve_interpolation
        out = resolve_interpolation(
            "v: %outer%",
            config={"outer": "%inner%", "inner": "42"},
        )
        assert out == "v: 42"

    def test_config_value_containing_env_var_resolves(self, monkeypatch):
        from neuroslm.dsl.interpolation import resolve_interpolation
        monkeypatch.setenv("BRIAN_NESTED", "nested_value")
        out = resolve_interpolation(
            "v: %wrapper%",
            config={"wrapper": "$BRIAN_NESTED"},
        )
        assert out == "v: nested_value"

    def test_cycle_raises_with_cycle_trail(self):
        from neuroslm.dsl.interpolation import (
            resolve_interpolation, InterpolationError,
        )
        with pytest.raises(InterpolationError, match=r"cycle|depth"):
            resolve_interpolation(
                "v: %a%",
                config={"a": "%b%", "b": "%a%"},
            )

    def test_max_depth_bound_prevents_runaway(self):
        from neuroslm.dsl.interpolation import (
            resolve_interpolation, InterpolationError,
        )
        # 9 levels of indirection with max_depth=8 must abort.
        chain = {f"k{i}": f"%k{i + 1}%" for i in range(8)}
        chain["k8"] = "tail"
        out = resolve_interpolation("v: %k0%", config=chain, max_depth=8)
        assert out == "v: tail"
        # Same chain with max_depth=4 must abort.
        with pytest.raises(InterpolationError, match=r"cycle|depth"):
            resolve_interpolation("v: %k0%", config=chain, max_depth=4)
