# -*- coding: utf-8 -*-
"""TDD: extended ``${LOG_TAIL:src:selector:N}`` README macros.

The base macro ``${LOG_TAIL:src:N}`` returns the **last N lines** of
``src`` (where ``src`` is ``best``, ``latest``, a TOML key, or a literal
path).  The extended form adds a **selector** segment that picks WHICH
N lines to return, enabling targeted excerpts of the run log:

    ${LOG_TAIL:latest:ood:3}   → 3 lines ending at the last [mid-ood] line
    ${LOG_TAIL:latest:best:3}  → 3 lines ending at the line with the best
                                  combined score (lowest train_ppl + 4×gap_ratio)
    ${LOG_TAIL:best:ood:5}     → in the BEST log, 5 lines ending at last OOD
    ${LOG_TAIL:best:best:5}    → in the BEST log, 5 lines around the
                                  best-metric step

Selector semantics
------------------
* ``ood``  — locate the LAST line matching the ``[mid-ood]`` prefix in
             the log; return that line plus the (N-1) immediately
             preceding lines, in original order.
* ``best`` — parse every ``[mid-ood]`` line in the log, compute combined
             score ``train_ppl + 4 × gap_ratio`` for each, pick the line
             with the lowest score; return that line plus the (N-1)
             immediately preceding lines.

When the selector finds no matching line (no ``[mid-ood]`` in the log),
the macro falls back to the legacy "last N lines" behaviour so the
README still renders something useful.

Contracts pinned
----------------
LT-1   Regex matches both 2-arg (legacy) and 3-arg (extended) forms.
LT-2   Backwards-compat: ``${LOG_TAIL:latest:3}`` keeps last-N semantics.
LT-3   ``${LOG_TAIL:latest:ood:N}`` returns N lines ending at last OOD line.
LT-4   ``${LOG_TAIL:latest:best:N}`` returns N lines ending at min-combined
       OOD line (NOT necessarily the last OOD line).
LT-5   ``${LOG_TAIL:best:ood:N}`` uses the .brian/best_run.ln source.
LT-6   N=1 is supported (just the selector's matched line).
LT-7   N larger than file length returns the entire file.
LT-8   Selector with no OOD lines in source → falls back to last-N lines.
LT-9   Selector with no source file → renders "*(log not available)*".
LT-10  Multiple macros in same template all resolve independently.
LT-11  GitHub-style link prefix is preserved on selector forms.
"""
from __future__ import annotations

import textwrap
from pathlib import Path

import pytest


# ── synthetic log content ──────────────────────────────────────────────

# 12 lines. Three OOD evals at lines 4, 7, 10. Best combined score at line 7.
# Layout (1-indexed for sanity — pythoneers can mentally subtract 1):
#   1  [train_dsl] boot
#   2  step  500 | … ppl 100.0 …                          (combined 108)
#   3  step  600 | … other line
#   4  [mid-ood] step 500: ppl=… gap_ratio=2.00 (train_ppl=100.0)
#   5  step 1000 | … ppl 30.0 …
#   6  step 1100 | … other line
#   7  [mid-ood] step 1000: ppl=… gap_ratio=1.50 (train_ppl=30.0)   ← BEST: 36
#   8  step 1500 | … ppl 25.0 …
#   9  step 1600 | … other line
#  10  [mid-ood] step 1500: ppl=… gap_ratio=3.00 (train_ppl=25.0)   ← LAST: 37
#  11  step 2000 | … ppl 22.0 …
#  12  step 2100 | … other line  ← absolute last line
_LOG_THREE_OODS = textwrap.dedent("""\
    [train_dsl] boot
    step   500 | loss 4.605 | lm 4.605 | ppl 100.0 | gnorm 1.0 | lr 3e-04 | 500 tok/s
    step   600 | loss 4.500 | lm 4.500 | ppl 90.0  | gnorm 1.0 | lr 3e-04 | 500 tok/s
    [mid-ood] step 500: wikitext ppl=200.0 gap_ratio=2.00 (train_ppl=100.0) (50 seq, 6430 tok)
    step  1000 | loss 3.401 | lm 3.401 | ppl 30.0  | gnorm 0.8 | lr 2e-04 | 600 tok/s
    step  1100 | loss 3.350 | lm 3.350 | ppl 28.5  | gnorm 0.8 | lr 2e-04 | 600 tok/s
    [mid-ood] step 1000: wikitext ppl=45.0 gap_ratio=1.50 (train_ppl=30.0) (50 seq, 6430 tok)
    step  1500 | loss 3.219 | lm 3.219 | ppl 25.0  | gnorm 0.7 | lr 1e-04 | 700 tok/s
    step  1600 | loss 3.180 | lm 3.180 | ppl 24.0  | gnorm 0.7 | lr 1e-04 | 700 tok/s
    [mid-ood] step 1500: wikitext ppl=75.0 gap_ratio=3.00 (train_ppl=25.0) (50 seq, 6430 tok)
    step  2000 | loss 3.091 | lm 3.091 | ppl 22.0  | gnorm 0.7 | lr 1e-04 | 720 tok/s
    step  2100 | loss 3.040 | lm 3.040 | ppl 20.9  | gnorm 0.7 | lr 1e-04 | 720 tok/s
""")

# Log with no OOD eval lines at all
_LOG_NO_OOD = textwrap.dedent("""\
    [train_dsl] boot
    step  500  | loss 5.298 | lm 5.298 | ppl 200.0 | gnorm 1.5 | lr 3e-04 | 500 tok/s
    step  1000 | loss 4.787 | lm 4.787 | ppl 120.0 | gnorm 1.2 | lr 3e-04 | 500 tok/s
""")


# ── helpers ────────────────────────────────────────────────────────────

def _write_log(root: Path, rel: str, content: str) -> Path:
    p = root / rel
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(content, encoding="utf-8")
    return p


def _write_best_ln(root: Path, log_path: Path) -> None:
    ln = root / ".brian" / "best_run.ln"
    ln.parent.mkdir(parents=True, exist_ok=True)
    ln.write_text(
        f"# auto-generated for test\n{log_path.relative_to(root).as_posix()}\n",
        encoding="utf-8",
    )


# ── LT-1: regex matches both legacy and extended forms ─────────────────

class TestRegexParse:

    def test_legacy_two_arg_form_still_matches(self):
        from neuroslm.readme_renderer import _LOG_TAIL_RE
        m = _LOG_TAIL_RE.search("${LOG_TAIL:latest:3}")
        assert m is not None

    def test_three_arg_form_matches_with_selector(self):
        from neuroslm.readme_renderer import _LOG_TAIL_RE
        m = _LOG_TAIL_RE.search("${LOG_TAIL:latest:ood:5}")
        assert m is not None

    def test_three_arg_form_with_best_selector(self):
        from neuroslm.readme_renderer import _LOG_TAIL_RE
        m = _LOG_TAIL_RE.search("${LOG_TAIL:best:best:7}")
        assert m is not None


# ── LT-2: legacy behaviour preserved ───────────────────────────────────

class TestLegacyTwoArgForm:

    def test_legacy_latest_n_returns_last_n_lines(self, tmp_path):
        from neuroslm.readme_renderer import resolve_log_macros
        _write_log(tmp_path, "logs/run.log", _LOG_THREE_OODS)
        out = resolve_log_macros("${LOG_TAIL:latest:2}", {}, repo_root=tmp_path)
        # last two lines should be step 2000 and step 2100
        assert "step  2000" in out
        assert "step  2100" in out
        assert "step  1500" not in out


# ── LT-3: latest:ood:N selector ────────────────────────────────────────

class TestOodSelectorLatest:

    def test_ood_selector_returns_last_ood_line(self, tmp_path):
        from neuroslm.readme_renderer import resolve_log_macros
        _write_log(tmp_path, "logs/run.log", _LOG_THREE_OODS)
        out = resolve_log_macros("${LOG_TAIL:latest:ood:1}", {}, repo_root=tmp_path)
        # The last [mid-ood] line is step 1500
        assert "[mid-ood] step 1500" in out
        # And nothing past it (no step 2000)
        assert "step  2000" not in out

    def test_ood_selector_returns_n_lines_ending_at_last_ood(self, tmp_path):
        from neuroslm.readme_renderer import resolve_log_macros
        _write_log(tmp_path, "logs/run.log", _LOG_THREE_OODS)
        out = resolve_log_macros("${LOG_TAIL:latest:ood:3}", {}, repo_root=tmp_path)
        # Returns 3 lines ending at the last OOD line:
        # line  8 step 1500
        # line  9 step 1600
        # line 10 [mid-ood] step 1500   ← last OOD line
        assert "step  1500" in out
        assert "step  1600" in out
        assert "[mid-ood] step 1500" in out
        # No content beyond
        assert "step  2000" not in out
        # No content much before
        assert "[mid-ood] step 500" not in out

    def test_ood_selector_n_equals_one(self, tmp_path):
        from neuroslm.readme_renderer import resolve_log_macros
        _write_log(tmp_path, "logs/run.log", _LOG_THREE_OODS)
        out = resolve_log_macros("${LOG_TAIL:latest:ood:1}", {}, repo_root=tmp_path)
        assert out.count("[mid-ood]") == 1
        assert "[mid-ood] step 1500" in out


# ── LT-4: latest:best:N selector ───────────────────────────────────────

class TestBestSelectorLatest:

    def test_best_selector_picks_lowest_combined_score_ood_line(self, tmp_path):
        from neuroslm.readme_renderer import resolve_log_macros
        _write_log(tmp_path, "logs/run.log", _LOG_THREE_OODS)
        # Combined scores:
        #   step  500: 100 + 4*2.00 = 108
        #   step 1000:  30 + 4*1.50 =  36   ← BEST
        #   step 1500:  25 + 4*3.00 =  37
        out = resolve_log_macros("${LOG_TAIL:latest:best:1}", {}, repo_root=tmp_path)
        assert "[mid-ood] step 1000" in out, \
            "best-selector should pick min-combined line (step 1000)"
        # Not the LAST one (step 1500), not the FIRST one (step 500)
        assert "[mid-ood] step 1500" not in out
        assert "[mid-ood] step 500" not in out

    def test_best_selector_returns_n_lines_ending_at_best(self, tmp_path):
        from neuroslm.readme_renderer import resolve_log_macros
        _write_log(tmp_path, "logs/run.log", _LOG_THREE_OODS)
        # 3 lines ending at [mid-ood] step 1000 (line 7):
        # line 5 step 1000
        # line 6 step 1100
        # line 7 [mid-ood] step 1000
        out = resolve_log_macros("${LOG_TAIL:latest:best:3}", {}, repo_root=tmp_path)
        assert "step  1000" in out
        assert "step  1100" in out
        assert "[mid-ood] step 1000" in out
        # Excluded: step 1500 and beyond
        assert "step  1500" not in out


# ── LT-5: best:ood:N uses .brian/best_run.ln source ────────────────────

class TestSelectorWithBestSource:

    def test_best_source_with_ood_selector(self, tmp_path):
        from neuroslm.readme_renderer import resolve_log_macros
        log = _write_log(tmp_path, "logs/run.log", _LOG_THREE_OODS)
        _write_best_ln(tmp_path, log)
        out = resolve_log_macros("${LOG_TAIL:best:ood:1}", {}, repo_root=tmp_path)
        assert "[mid-ood] step 1500" in out


# ── LT-6/LT-7: edge cases on N ─────────────────────────────────────────

class TestNEdgeCases:

    def test_n_larger_than_lines_returns_whole_prefix(self, tmp_path):
        """N=999 means "give me everything up to and including the match"."""
        from neuroslm.readme_renderer import resolve_log_macros
        _write_log(tmp_path, "logs/run.log", _LOG_THREE_OODS)
        # ood selector with huge N — gets everything from start through last OOD
        out = resolve_log_macros("${LOG_TAIL:latest:ood:999}", {}, repo_root=tmp_path)
        assert "[train_dsl] boot" in out         # first line included
        assert "[mid-ood] step 1500" in out      # match included
        assert "step  2000" not in out           # nothing AFTER the match


# ── LT-8: graceful fallback when selector finds no match ──────────────

class TestSelectorFallback:

    def test_ood_selector_no_match_falls_back_to_last_n(self, tmp_path):
        """No [mid-ood] lines → fall back to last-N-lines semantics."""
        from neuroslm.readme_renderer import resolve_log_macros
        _write_log(tmp_path, "logs/run.log", _LOG_NO_OOD)
        out = resolve_log_macros("${LOG_TAIL:latest:ood:2}", {}, repo_root=tmp_path)
        # last 2 lines of _LOG_NO_OOD are step 500 and step 1000
        assert "step  500" in out
        assert "step  1000" in out

    def test_best_selector_no_match_falls_back_to_last_n(self, tmp_path):
        from neuroslm.readme_renderer import resolve_log_macros
        _write_log(tmp_path, "logs/run.log", _LOG_NO_OOD)
        out = resolve_log_macros("${LOG_TAIL:latest:best:1}", {}, repo_root=tmp_path)
        assert "step  1000" in out


# ── LT-9: missing source ───────────────────────────────────────────────

class TestSelectorMissingSource:

    def test_no_log_renders_not_available(self, tmp_path):
        from neuroslm.readme_renderer import resolve_log_macros
        # no log file exists
        out = resolve_log_macros("${LOG_TAIL:latest:ood:3}", {}, repo_root=tmp_path)
        assert "not available" in out


# ── LT-10: multiple macros in one template ─────────────────────────────

class TestMultipleMacros:

    def test_legacy_and_extended_macros_coexist(self, tmp_path):
        from neuroslm.readme_renderer import resolve_log_macros
        _write_log(tmp_path, "logs/run.log", _LOG_THREE_OODS)
        template = (
            "First:\n${LOG_TAIL:latest:2}\n\n"
            "OOD:\n${LOG_TAIL:latest:ood:1}\n\n"
            "Best:\n${LOG_TAIL:latest:best:1}\n"
        )
        out = resolve_log_macros(template, {}, repo_root=tmp_path)
        # legacy: last 2 → step 2000, step 2100
        assert "step  2000" in out
        # ood selector: [mid-ood] step 1500
        assert "[mid-ood] step 1500" in out
        # best selector: [mid-ood] step 1000
        assert "[mid-ood] step 1000" in out


# ── LT-11: GitHub link preserved on extended form ─────────────────────

class TestLinkPreservation:

    def test_extended_form_includes_link(self, tmp_path):
        from neuroslm.readme_renderer import resolve_log_macros
        _write_log(tmp_path, "logs/run.log", _LOG_THREE_OODS)
        out = resolve_log_macros("${LOG_TAIL:latest:ood:1}", {}, repo_root=tmp_path)
        # Markdown link should still be present
        assert "[`logs/run.log`]" in out or "[`logs\\run.log`]" in out
        # Fenced code block
        assert "```" in out
