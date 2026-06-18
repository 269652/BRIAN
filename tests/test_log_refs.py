# -*- coding: utf-8 -*-
"""TDD: best-run detection, .ln reference files, and log quoting.

System under test: neuroslm.log_refs

The system lets users keep a permanent pointer to the current best training
run, "quote" (lock) arbitrary log files so they survive cleanup, and detect
the best run from collected logs by scanning metrics.

Design contracts pinned here:

  1. LogRef IO  — .ln files are text files with one path + optional # comments
  2. RefScanner — scan_refs / locked_logs collects all referenced paths
  3. RunScore   — score_log extracts gap_ratio / train_ppl from log text
  4. FindBest   — find_best_log picks the log with lowest gap_ratio (or ppl)
  5. Pointer    — update_best_run_pointer writes .brian/best_run.ln

Conventions:
  * All paths inside .ln files are stored relative to the repo root
    (the same root passed to scan_refs / locked_logs).
  * locked_logs returns frozenset[Path] of ABSOLUTE paths that must not
    be deleted, regardless of whether the file currently exists.
"""
from __future__ import annotations

import textwrap
from pathlib import Path

import pytest

# ---------------------------------------------------------------------------
# Synthetic log fragments used by score_log tests
# ---------------------------------------------------------------------------

_LOG_WITH_OOD = textwrap.dedent("""\
    [train_dsl] boot @ 2026-06-17T10:00:00Z
    [train_dsl] git_commit deadbeef (master)
    step   500 | loss 4.512 | lm 4.512 | ppl 91.2 | gnorm 1.10 | lr 3.00e-04 | 600 tok/s
    [mid-ood] step 500: wikitext ppl=310.0 gap_ratio=3.40 (train_ppl=91.2) (50 seq, 6430 tok)
    step  1000 | loss 4.110 | lm 4.110 | ppl 61.0 | gnorm 0.92 | lr 2.80e-04 | 610 tok/s
    [mid-ood] step 1000: wikitext ppl=255.0 gap_ratio=4.18 (train_ppl=61.0) (50 seq, 6430 tok)
    step  1500 | loss 3.920 | lm 3.920 | ppl 50.3 | gnorm 0.85 | lr 2.50e-04 | 615 tok/s
    [mid-ood] step 1500: wikitext ppl=200.0 gap_ratio=3.98 (train_ppl=50.3) (50 seq, 6430 tok)
""")

_LOG_NO_OOD = textwrap.dedent("""\
    [train_dsl] boot @ 2026-06-17T11:00:00Z
    step   500 | loss 5.100 | lm 5.100 | ppl 164.0 | gnorm 1.50 | lr 3.00e-04 | 550 tok/s
    step  1000 | loss 4.800 | lm 4.800 | ppl 121.5 | gnorm 1.30 | lr 2.80e-04 | 555 tok/s
""")

_LOG_EMPTY = ""

_LOG_WITH_LOWER_GAP = textwrap.dedent("""\
    [train_dsl] boot @ 2026-06-17T12:00:00Z
    step  1000 | loss 3.500 | lm 3.500 | ppl 33.1 | gnorm 0.70 | lr 2.50e-04 | 700 tok/s
    [mid-ood] step 1000: wikitext ppl=88.0 gap_ratio=2.66 (train_ppl=33.1) (50 seq, 6430 tok)
""")


# ===========================================================================
# 1. LogRef IO — .ln file read / write
# ===========================================================================

class TestLogRefIO:

    def test_write_ref_creates_ln_file(self, tmp_path):
        from neuroslm.log_refs import write_ref
        ln = tmp_path / "best_run.ln"
        target = Path("logs/20260617/SmolLM/170512_20_2000.log")
        ref = write_ref(ln, target)
        assert ln.is_file(), ".ln file must be created by write_ref"
        assert ref.path == ln

    def test_read_ref_returns_target(self, tmp_path):
        from neuroslm.log_refs import write_ref, read_ref
        ln = tmp_path / "run.ln"
        target = Path("logs/20260617/SmolLM/170512_20_2000.log")
        write_ref(ln, target)
        ref = read_ref(ln)
        assert ref.target == target

    def test_write_read_roundtrip_with_comment(self, tmp_path):
        from neuroslm.log_refs import write_ref, read_ref
        ln = tmp_path / "run.ln"
        target = Path("logs/20260616/SmolLM/174900_20_2140.log")
        write_ref(ln, target, comment="manually quoted — best gap_ratio so far")
        ref = read_ref(ln)
        assert ref.target == target
        assert "best gap_ratio" in ref.comment

    def test_read_ref_ignores_comment_lines(self, tmp_path):
        from neuroslm.log_refs import read_ref
        ln = tmp_path / "run.ln"
        ln.write_text(
            "# this is a comment\n"
            "# another comment\n"
            "logs/20260617/SmolLM/170512_20_2000.log\n",
            encoding="utf-8",
        )
        ref = read_ref(ln)
        assert ref.target == Path("logs/20260617/SmolLM/170512_20_2000.log")

    def test_write_ref_overwrites_existing(self, tmp_path):
        from neuroslm.log_refs import write_ref, read_ref
        ln = tmp_path / "run.ln"
        write_ref(ln, Path("logs/old.log"))
        write_ref(ln, Path("logs/new.log"))
        assert read_ref(ln).target == Path("logs/new.log")

    def test_read_ref_raises_on_missing_file(self, tmp_path):
        from neuroslm.log_refs import read_ref
        with pytest.raises(FileNotFoundError):
            read_ref(tmp_path / "nonexistent.ln")

    def test_read_ref_raises_on_empty_file(self, tmp_path):
        from neuroslm.log_refs import read_ref
        ln = tmp_path / "empty.ln"
        ln.write_text("# comment only\n", encoding="utf-8")
        with pytest.raises(ValueError, match="no target path"):
            read_ref(ln)


# ===========================================================================
# 2. RefScanner — scan_refs / locked_logs
# ===========================================================================

class TestRefScanner:

    def _make_ln(self, directory: Path, name: str, target: Path,
                 comment: str = "") -> Path:
        from neuroslm.log_refs import write_ref
        ln = directory / name
        write_ref(ln, target, comment=comment)
        return ln

    def test_scan_refs_finds_ln_files(self, tmp_path):
        from neuroslm.log_refs import scan_refs
        refs_dir = tmp_path / ".neuro" / "refs"
        refs_dir.mkdir(parents=True)
        self._make_ln(refs_dir, "a.ln", Path("logs/a.log"))
        self._make_ln(refs_dir, "b.ln", Path("logs/b.log"))
        found = scan_refs(tmp_path)
        paths = {r.path for r in found}
        assert refs_dir / "a.ln" in paths
        assert refs_dir / "b.ln" in paths

    def test_scan_refs_finds_nested_ln_files(self, tmp_path):
        from neuroslm.log_refs import scan_refs
        deep = tmp_path / ".neuro"
        deep.mkdir(parents=True)
        self._make_ln(deep, "best_run.ln", Path("logs/best.log"))
        found = scan_refs(tmp_path)
        assert any(r.path == deep / "best_run.ln" for r in found)

    def test_scan_refs_empty_root(self, tmp_path):
        from neuroslm.log_refs import scan_refs
        assert scan_refs(tmp_path) == []

    def test_locked_logs_returns_absolute_target_paths(self, tmp_path):
        from neuroslm.log_refs import locked_logs, write_ref
        neuro = tmp_path / ".neuro"
        neuro.mkdir()
        write_ref(neuro / "best_run.ln", Path("logs/best.log"))
        locked = locked_logs(tmp_path)
        assert tmp_path / "logs" / "best.log" in locked

    def test_locked_logs_includes_refs_subdir(self, tmp_path):
        from neuroslm.log_refs import locked_logs, write_ref
        refs = tmp_path / ".neuro" / "refs"
        refs.mkdir(parents=True)
        write_ref(refs / "pinned.ln", Path("logs/pinned.log"))
        locked = locked_logs(tmp_path)
        assert tmp_path / "logs" / "pinned.log" in locked

    def test_locked_logs_works_when_target_does_not_exist(self, tmp_path):
        """A referenced but missing log is still locked — the ref protects the
        path even before the file arrives (e.g. in-progress run)."""
        from neuroslm.log_refs import locked_logs, write_ref
        neuro = tmp_path / ".neuro"
        neuro.mkdir()
        write_ref(neuro / "future.ln", Path("logs/not_yet.log"))
        locked = locked_logs(tmp_path)
        assert tmp_path / "logs" / "not_yet.log" in locked

    def test_locked_logs_returns_frozenset(self, tmp_path):
        from neuroslm.log_refs import locked_logs
        result = locked_logs(tmp_path)
        assert isinstance(result, frozenset)


# ===========================================================================
# 3. RunScore — score_log parses log text for ranking metrics
# ===========================================================================

class TestScoreLog:

    def test_score_log_extracts_best_gap_ratio(self):
        from neuroslm.log_refs import score_log
        score = score_log_from_text(_LOG_WITH_OOD)
        # Best (minimum) gap_ratio across 3 OOD evals: 3.40, 4.18, 3.98 → 3.40
        assert score is not None
        assert score.gap_ratio == pytest.approx(3.40, rel=1e-3)

    def test_score_log_extracts_train_ppl_from_last_step(self):
        from neuroslm.log_refs import score_log
        score = score_log_from_text(_LOG_NO_OOD)
        assert score is not None
        assert score.gap_ratio is None
        assert score.train_ppl == pytest.approx(121.5, rel=1e-3)

    def test_score_log_returns_none_for_empty_log(self):
        from neuroslm.log_refs import score_log
        score = score_log_from_text(_LOG_EMPTY)
        assert score is None

    def test_score_log_extracts_final_ood_ppl(self):
        from neuroslm.log_refs import score_log
        score = score_log_from_text(_LOG_WITH_OOD)
        # Last OOD eval was step 1500 → ood_ppl 200.0
        assert score is not None
        assert score.ood_ppl == pytest.approx(200.0, rel=1e-3)

    def test_score_log_extracts_step_count(self):
        from neuroslm.log_refs import score_log
        score = score_log_from_text(_LOG_WITH_OOD)
        assert score is not None
        assert score.step == 1500

    def test_score_log_train_ppl_present_alongside_gap_ratio(self):
        from neuroslm.log_refs import score_log
        score = score_log_from_text(_LOG_WITH_OOD)
        assert score is not None
        assert score.train_ppl == pytest.approx(50.3, rel=1e-3)


# Helper: score_log accepts a Path, but our tests have text.
# We write to a tmp file via a module-level fixture-like helper.
import tempfile, os

def score_log_from_text(text: str):
    from neuroslm.log_refs import score_log
    with tempfile.NamedTemporaryFile(mode="w", suffix=".log",
                                     delete=False, encoding="utf-8") as f:
        f.write(text)
        name = f.name
    try:
        return score_log(Path(name))
    finally:
        os.unlink(name)


# ===========================================================================
# 4. FindBest — find_best_log picks the log with the best metric
# ===========================================================================

class TestFindBestLog:

    def _write_log(self, directory: Path, name: str, content: str) -> Path:
        p = directory / name
        p.write_text(content, encoding="utf-8")
        return p

    def test_find_best_log_by_gap_ratio(self, tmp_path):
        from neuroslm.log_refs import find_best_log
        self._write_log(tmp_path, "a.log", _LOG_WITH_OOD)         # best gap 3.40
        self._write_log(tmp_path, "b.log", _LOG_WITH_LOWER_GAP)   # best gap 2.66
        best = find_best_log(tmp_path)
        assert best == tmp_path / "b.log"

    def test_find_best_log_fallback_to_ppl(self, tmp_path):
        from neuroslm.log_refs import find_best_log
        self._write_log(tmp_path, "a.log", _LOG_NO_OOD)   # ppl 121.5
        best = find_best_log(tmp_path)
        assert best == tmp_path / "a.log"

    def test_find_best_log_prefers_gap_ratio_over_ppl_only(self, tmp_path):
        """A run with gap_ratio always beats a run with only train_ppl."""
        from neuroslm.log_refs import find_best_log
        self._write_log(tmp_path, "ppl_only.log", _LOG_NO_OOD)      # no OOD
        self._write_log(tmp_path, "with_ood.log", _LOG_WITH_OOD)    # has gap_ratio
        best = find_best_log(tmp_path)
        assert best == tmp_path / "with_ood.log"

    def test_find_best_log_returns_none_for_empty_dir(self, tmp_path):
        from neuroslm.log_refs import find_best_log
        assert find_best_log(tmp_path) is None

    def test_find_best_log_scans_subdirectories(self, tmp_path):
        from neuroslm.log_refs import find_best_log
        sub = tmp_path / "20260617" / "SmolLM"
        sub.mkdir(parents=True)
        self._write_log(sub, "170512_20_2000.log", _LOG_WITH_LOWER_GAP)
        best = find_best_log(tmp_path)
        assert best == sub / "170512_20_2000.log"

    def test_find_best_log_ignores_non_log_files(self, tmp_path):
        from neuroslm.log_refs import find_best_log
        (tmp_path / "notes.txt").write_text("not a log", encoding="utf-8")
        (tmp_path / "run.ln").write_text("logs/foo.log\n", encoding="utf-8")
        best = find_best_log(tmp_path)
        assert best is None

    def test_find_best_log_accepts_metric_ppl(self, tmp_path):
        from neuroslm.log_refs import find_best_log
        self._write_log(tmp_path, "a.log", _LOG_NO_OOD)    # ppl 121.5
        self._write_log(tmp_path, "b.log", _LOG_WITH_OOD)  # final train ppl 50.3
        best = find_best_log(tmp_path, metric="ppl")
        assert best == tmp_path / "b.log"


# ===========================================================================
# 5. BestRunPointer — update_best_run_pointer manages .brian/best_run.ln
# ===========================================================================

class TestBestRunPointer:

    def test_update_writes_best_run_ln(self, tmp_path):
        from neuroslm.log_refs import update_best_run_pointer, BEST_RUN_LN
        logs_dir = tmp_path / "logs"
        logs_dir.mkdir()
        (logs_dir / "run.log").write_text(_LOG_WITH_LOWER_GAP, encoding="utf-8")
        result = update_best_run_pointer(root=tmp_path, log_dir=logs_dir)
        ln_path = tmp_path / BEST_RUN_LN
        assert ln_path.is_file(), ".brian/best_run.ln must be created"
        assert result == logs_dir / "run.log"

    def test_update_returns_none_when_no_logs(self, tmp_path):
        from neuroslm.log_refs import update_best_run_pointer
        logs_dir = tmp_path / "logs"
        logs_dir.mkdir()
        result = update_best_run_pointer(root=tmp_path, log_dir=logs_dir)
        assert result is None

    def test_best_run_ln_appears_in_locked_logs(self, tmp_path):
        """After update, the best run log path is in locked_logs."""
        from neuroslm.log_refs import update_best_run_pointer, locked_logs
        logs_dir = tmp_path / "logs"
        logs_dir.mkdir()
        log_file = logs_dir / "run.log"
        log_file.write_text(_LOG_WITH_LOWER_GAP, encoding="utf-8")
        update_best_run_pointer(root=tmp_path, log_dir=logs_dir)
        locked = locked_logs(tmp_path)
        assert log_file.resolve() in locked

    def test_update_overwrites_previous_best(self, tmp_path):
        from neuroslm.log_refs import update_best_run_pointer, read_ref, BEST_RUN_LN
        logs_dir = tmp_path / "logs"
        logs_dir.mkdir()
        (logs_dir / "old_best.log").write_text(_LOG_WITH_OOD, encoding="utf-8")
        update_best_run_pointer(root=tmp_path, log_dir=logs_dir)
        # Now add a better log
        (logs_dir / "new_best.log").write_text(_LOG_WITH_LOWER_GAP, encoding="utf-8")
        update_best_run_pointer(root=tmp_path, log_dir=logs_dir)
        ref = read_ref(tmp_path / BEST_RUN_LN)
        assert ref.target == Path("logs/new_best.log")

    def test_quote_adds_ref_to_refs_dir(self, tmp_path):
        """quote_log creates .neuro/refs/<name>.ln pointing to the log."""
        from neuroslm.log_refs import quote_log, REFS_DIR
        logs_dir = tmp_path / "logs"
        logs_dir.mkdir()
        log_file = logs_dir / "special.log"
        log_file.write_text(_LOG_WITH_OOD, encoding="utf-8")
        ref_path = quote_log(root=tmp_path, log_path=log_file, name="special")
        assert ref_path == tmp_path / REFS_DIR / "special.ln"
        assert ref_path.is_file()

    def test_quote_log_appears_in_locked_logs(self, tmp_path):
        from neuroslm.log_refs import quote_log, locked_logs
        logs_dir = tmp_path / "logs"
        logs_dir.mkdir()
        log_file = logs_dir / "special.log"
        log_file.write_text(_LOG_WITH_OOD, encoding="utf-8")
        quote_log(root=tmp_path, log_path=log_file, name="special")
        locked = locked_logs(tmp_path)
        assert log_file.resolve() in locked

    def test_unquote_removes_ref(self, tmp_path):
        from neuroslm.log_refs import quote_log, unquote_log, locked_logs
        logs_dir = tmp_path / "logs"
        logs_dir.mkdir()
        log_file = logs_dir / "special.log"
        log_file.write_text(_LOG_WITH_OOD, encoding="utf-8")
        quote_log(root=tmp_path, log_path=log_file, name="special")
        unquote_log(root=tmp_path, name="special")
        locked = locked_logs(tmp_path)
        assert log_file.resolve() not in locked

    def test_unquote_nonexistent_raises(self, tmp_path):
        from neuroslm.log_refs import unquote_log
        with pytest.raises(FileNotFoundError):
            unquote_log(root=tmp_path, name="doesnotexist")


# ===========================================================================
# 6. CheckpointExtraction — extract_checkpoint_url + checkpoint.ln
# ===========================================================================

_LOG_WITH_CKPT = (
    "step 500 | loss 3.100 | lm 3.100 | ppl 22.2 |\n"
    "[ckpt_push] ✓ pushed step500.pt → hf://moritzroessler/BRIAN/checkpoints/20260617-170512_abc1234_SmolLM/step500.pt\n"
    "[mid-ood] step 500: wikitext ppl=310.0 gap_ratio=3.40 (train_ppl=91.2)\n"
    "step 1000 | loss 2.800 | lm 2.800 | ppl 16.4 |\n"
    "[ckpt_push] ✓ pushed step1000.pt (optimizer stripped) → hf://moritzroessler/BRIAN/checkpoints/20260617-170512_abc1234_SmolLM/step1000.pt\n"
)

_LOG_WITHOUT_CKPT = (
    "step 500 | loss 3.100 | lm 3.100 | ppl 22.2 |\n"
    "[mid-ood] step 500: wikitext ppl=310.0 gap_ratio=3.40 (train_ppl=91.2)\n"
)


class TestCheckpointExtraction:

    def test_extracts_last_hf_url(self):
        from neuroslm.log_refs import extract_checkpoint_url
        url = extract_checkpoint_url(_LOG_WITH_CKPT)
        assert url == "hf://moritzroessler/BRIAN/checkpoints/20260617-170512_abc1234_SmolLM/step1000.pt"

    def test_returns_none_when_no_push_lines(self):
        from neuroslm.log_refs import extract_checkpoint_url
        assert extract_checkpoint_url(_LOG_WITHOUT_CKPT) is None

    def test_returns_none_for_empty_text(self):
        from neuroslm.log_refs import extract_checkpoint_url
        assert extract_checkpoint_url("") is None

    def test_handles_optimizer_stripped_variant(self):
        """(optimizer stripped) note between filename and → must not break parsing."""
        from neuroslm.log_refs import extract_checkpoint_url
        line = "[ckpt_push] ✓ pushed step500.pt (optimizer stripped) → hf://owner/repo/checkpoints/dir/step500.pt"
        assert extract_checkpoint_url(line) == "hf://owner/repo/checkpoints/dir/step500.pt"


class TestCheckpointLn:

    def _write_log_file(self, path: Path, content: str) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")

    def test_update_writes_checkpoint_ln_when_url_found(self, tmp_path):
        from neuroslm.log_refs import update_best_run_pointer, CHECKPOINT_LN
        logs_dir = tmp_path / "logs"
        self._write_log_file(logs_dir / "run.log", _LOG_WITH_CKPT)
        update_best_run_pointer(root=tmp_path, log_dir=logs_dir)
        ckpt_ln = tmp_path / CHECKPOINT_LN
        assert ckpt_ln.is_file(), ".brian/checkpoint.ln must be created when HF URL is present"

    def test_checkpoint_ln_contains_hf_url(self, tmp_path):
        from neuroslm.log_refs import update_best_run_pointer, read_checkpoint_url
        logs_dir = tmp_path / "logs"
        self._write_log_file(logs_dir / "run.log", _LOG_WITH_CKPT)
        update_best_run_pointer(root=tmp_path, log_dir=logs_dir)
        url = read_checkpoint_url(tmp_path)
        assert url == "hf://moritzroessler/BRIAN/checkpoints/20260617-170512_abc1234_SmolLM/step1000.pt"

    def test_update_skips_checkpoint_ln_when_no_url(self, tmp_path):
        """If the log has no checkpoint push lines, no checkpoint.ln is written."""
        from neuroslm.log_refs import update_best_run_pointer, CHECKPOINT_LN
        logs_dir = tmp_path / "logs"
        self._write_log_file(logs_dir / "run.log", _LOG_WITHOUT_CKPT)
        update_best_run_pointer(root=tmp_path, log_dir=logs_dir)
        assert not (tmp_path / CHECKPOINT_LN).exists()

    def test_checkpoint_ln_overwritten_by_better_run(self, tmp_path):
        """Calling update twice overwrites checkpoint.ln with the new best URL."""
        from neuroslm.log_refs import update_best_run_pointer, read_checkpoint_url
        logs_dir = tmp_path / "logs"
        # First call: log with step1000 checkpoint
        # _LOG_WITH_CKPT: final train_ppl 16.4, best gap_ratio 3.40
        # → combined score = 16.4 + 4×3.40 = 30.0
        self._write_log_file(logs_dir / "run1.log", _LOG_WITH_CKPT)
        update_best_run_pointer(root=tmp_path, log_dir=logs_dir)
        # Second run with a STRICTLY BETTER combined score than run1
        # (lower train_ppl + lower gap_ratio → much lower combined).
        # combined = 5.0 + 4×2.10 = 13.4  <  30.0
        better_log = (
            "step 2000 | loss 1.609 | lm 1.609 | ppl 5.0 | gnorm 0.4 | lr 1.00e-04 | 700 tok/s\n"
            "[ckpt_push] ✓ pushed step2000.pt → hf://owner/repo/checkpoints/run2/step2000.pt\n"
            "[mid-ood] step 2000: wikitext ppl=10.5 gap_ratio=2.10 (train_ppl=5.0) (50 seq, 6430 tok)\n"
        )
        self._write_log_file(logs_dir / "run2.log", better_log)
        update_best_run_pointer(root=tmp_path, log_dir=logs_dir)
        url = read_checkpoint_url(tmp_path)
        assert url is not None and "run2" in url
