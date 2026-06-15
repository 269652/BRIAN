"""Contracts for ``_nt_colour`` — the NFG Graphviz renderer's
neurotransmitter colour-lookup function.

Bug discovered 2026-06-15: every ``synapse`` row in the master arch
stores its NT name as ``"glutamate"`` (with literal quotes baked into
the IR attribute string), but ``_nt_colour`` did a raw dict lookup
without stripping the quotes. Result: every glutamate / gaba / etc.
synapse in the canonical NFG diagram fell through to the grey
``_DEFAULT_MOD_COLOUR``. The NT colour grammar — which is the
whole reason the lookup exists — was a no-op across the entire
production render. Modulation edges weren't affected because their
NT comes from the source NODE name (no quotes).

This file pins the quote-tolerant lookup so future refactors of the
IR/attr storage can't silently re-break the colour grammar.
"""
from __future__ import annotations

import pytest


# ──────────────────────────────────────────────────────────────────────
# Per-NT colour resolution
# ──────────────────────────────────────────────────────────────────────


class TestNtColourQuoteTolerance:
    """``_nt_colour`` must return the correct NT colour regardless of
    whether the caller hands it a bare name or a DSL-quoted name."""

    @pytest.mark.parametrize(
        "raw,expected",
        [
            ("glutamate",       "#1f77b4"),  # blue
            ('"glutamate"',     "#1f77b4"),  # DSL-quoted -- main bug
            ("'glutamate'",     "#1f77b4"),  # single-quoted variant
            ("GLUTAMATE",       "#1f77b4"),  # case-insensitive (existing)
            ('"GLUTAMATE"',     "#1f77b4"),  # quoted + upper
            ("gaba",            "#7f7f7f"),
            ('"gaba"',          "#7f7f7f"),
            ('"dopamine"',      None),       # known NT, just check non-default
        ],
    )
    def test_nt_colour_strips_quotes(self, raw: str, expected: str):
        from neuroslm.compiler.nfg_graphviz import (
            _nt_colour, _DEFAULT_MOD_COLOUR,
        )

        got = _nt_colour(raw)
        if expected is None:
            # Just make sure it's NOT the grey fallback (i.e. lookup
            # succeeded). The exact colour for dopamine etc. is the
            # renderer's choice — we don't pin it here.
            assert got != _DEFAULT_MOD_COLOUR, (
                f"_nt_colour({raw!r}) returned the default {got!r}; "
                f"expected a real NT colour (any non-default value)"
            )
        else:
            assert got == expected, (
                f"_nt_colour({raw!r}) returned {got!r}; expected "
                f"{expected!r}. The DSL stores NT names with quotes "
                f"in the attr dict (`neurotransmitter: \"glutamate\"`) "
                f"-- the lookup must strip them before the dict probe."
            )

    def test_unknown_nt_still_falls_back_to_default(self):
        """Negative control: a genuinely unknown NT name (with or
        without quotes) must still hit the grey default — we only
        strip quotes, we don't invent colours."""
        from neuroslm.compiler.nfg_graphviz import (
            _nt_colour, _DEFAULT_MOD_COLOUR,
        )

        assert _nt_colour("nonexistent_nt") == _DEFAULT_MOD_COLOUR
        assert _nt_colour('"nonexistent_nt"') == _DEFAULT_MOD_COLOUR
        assert _nt_colour("") == _DEFAULT_MOD_COLOUR
        assert _nt_colour(None) == _DEFAULT_MOD_COLOUR  # type: ignore[arg-type]

    def test_real_master_arch_glutamate_synapse_is_blue(self, tmp_path):
        """End-to-end: emit DOT from the master arch and assert at
        least one of our new cortex_* -> pfc edges is rendered with
        the glutamate blue, not the grey default. This is the bug
        the audit surfaced visually on nfg_rcc_bowtie.png."""
        from pathlib import Path
        from neuroslm.compiler.hypergraph_ir import lift_arch_to_hypergraph
        from neuroslm.compiler.nfg_graphviz import emit_dot_from_hypergraph

        repo_root = Path(__file__).resolve().parent.parent
        ir = lift_arch_to_hypergraph(repo_root / "architectures" / "master")
        dot_src = emit_dot_from_hypergraph(ir, engine="dot")

        # The blue colour for glutamate, as declared in _NT_COLOURS.
        GLUTAMATE_BLUE = "#1f77b4"

        # Look for an edge line that names a cortex_* source AND uses
        # the glutamate blue. Graphviz writes attrs like
        #   "cortex_code" -> "pfc" [color="#1f77b4" ...]
        cortex_blue_edges = [
            line for line in dot_src.splitlines()
            if "cortex_" in line and GLUTAMATE_BLUE in line
        ]
        assert cortex_blue_edges, (
            "Expected at least one cortex_* edge rendered with "
            f"glutamate blue {GLUTAMATE_BLUE!r}, found none. The IR "
            "stores neurotransmitter as the quoted string "
            "'\"glutamate\"' -- the renderer's colour lookup must "
            "strip the quotes."
        )
