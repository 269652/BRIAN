"""Boot-stamp + log-naming contracts (deploy 40923107 forensic-friction
fix).

Background
----------
Deploy 40923107 trained for 1480 steps with a wild PPL trajectory.
Investigating the cause required:

* Hand-correlating the deploy timestamp with the git HEAD at launch
  time (no log line records it).
* Reading ``arch.neuro`` from master to guess at what the unfolded
  DSL looked like (no log line records the canonical hash either).
* Greping through 5 log files named ``logs/vast/{instance}__neuroslm-full.log``
  with no timestamp suffix — when two runs share the same instance
  id the second clobbers the first.

These contracts pin the artifacts that make forensic re-runs cheap:

1. ``_print_boot_stamp()`` in ``neuroslm.train_dsl`` prints a
   structured 3-line stamp covering ``git_commit``, ``arch_dsl_sha256``,
   and ``utc_iso8601`` BEFORE any other train_dsl output.
2. ``log_pusher.sh`` filename convention includes the UTC boot
   timestamp (``YYYYMMDDTHHMMSSZ``) so two runs on the same instance
   never alias.
3. The boot-stamp format must be parseable — these tests pin the
   exact prefix/regex so ``brian analyze-log`` and any future tooling
   can rely on it.
"""
from __future__ import annotations

import io
import re
import subprocess
import sys
from contextlib import redirect_stdout
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parent.parent.parent


# ─────────────────────────────────────────────────────────────────
# Contract A — `_print_boot_stamp` exists and emits 3 lines
# ─────────────────────────────────────────────────────────────────


def test_print_boot_stamp_function_is_importable():
    """The helper must be a public symbol so tests, the CLI, and any
    out-of-tree integration (notebooks, Colab harness, etc.) can call
    it. Pin the import path so a rename triggers a CI failure rather
    than silent loss of the stamp."""
    from neuroslm.train_dsl import _print_boot_stamp  # noqa: F401


def test_print_boot_stamp_prints_three_labeled_lines():
    """The stamp must be a fixed 3-line block:

      [train_dsl] boot @ <UTC ISO-8601 timestamp>
      [train_dsl] git_commit <40-hex sha> (<branch>)
      [train_dsl] arch_dsl_sha256 <64-hex sha> (<arch_root or '-'>)

    Each line begins with the ``[train_dsl] `` prefix so the log
    pipeline's grep filters keep matching. Order is fixed so
    forensic tools can rely on positional indexing.
    """
    from neuroslm.train_dsl import _print_boot_stamp

    buf = io.StringIO()
    with redirect_stdout(buf):
        _print_boot_stamp(arch_root=REPO_ROOT / "architectures" / "master")
    lines = [ln for ln in buf.getvalue().splitlines() if ln.strip()]
    assert len(lines) == 3, (
        f"expected exactly 3 stamp lines; got {len(lines)}:\n"
        + "\n".join(lines)
    )
    assert lines[0].startswith("[train_dsl] boot @ "), lines[0]
    assert lines[1].startswith("[train_dsl] git_commit "), lines[1]
    assert lines[2].startswith("[train_dsl] arch_dsl_sha256 "), lines[2]


def test_boot_timestamp_is_utc_iso8601_with_Z_suffix():
    """The first line's timestamp must be UTC ISO-8601 with a trailing
    ``Z`` (zulu). Avoids the timezone-confusion that bit deploy
    40921910 when the analyzer parsed local-time strings as UTC."""
    from neuroslm.train_dsl import _print_boot_stamp

    buf = io.StringIO()
    with redirect_stdout(buf):
        _print_boot_stamp(arch_root=REPO_ROOT / "architectures" / "master")
    first = buf.getvalue().splitlines()[0]
    # Match "2026-06-14T17:42:09Z" exactly
    m = re.match(
        r"^\[train_dsl\] boot @ "
        r"(\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}Z)$",
        first,
    )
    assert m, (
        f"timestamp doesn't match UTC ISO-8601 Z format: {first!r}"
    )


def test_git_commit_line_carries_40_hex_sha():
    """The git_commit line must include a 40-char hex sha so
    ``git checkout <sha>`` from the log Just Works."""
    from neuroslm.train_dsl import _print_boot_stamp

    buf = io.StringIO()
    with redirect_stdout(buf):
        _print_boot_stamp(arch_root=REPO_ROOT / "architectures" / "master")
    second = buf.getvalue().splitlines()[1]
    m = re.match(
        r"^\[train_dsl\] git_commit ([0-9a-f]{40})\s*(\([^)]+\))?$",
        second,
    )
    assert m, f"git_commit line malformed: {second!r}"
    # The sha must be the actual current HEAD (or '-' if not in a repo,
    # in which case the regex above wouldn't match anyway)
    head = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=REPO_ROOT, capture_output=True, text=True,
    ).stdout.strip()
    if head:
        assert m.group(1) == head, (
            f"stamp sha {m.group(1)} != git HEAD {head}"
        )


def test_arch_dsl_sha256_line_carries_64_hex_sha_for_real_arch():
    """For a real architecture folder, the arch_dsl_sha256 line must
    carry the SHA-256 of the unfolded DSL bytes — the canonical
    fingerprint that identifies which exact arch.neuro produced the
    log. Two runs with identical SHA must be byte-identical
    architectures."""
    from neuroslm.train_dsl import _print_boot_stamp

    arch_root = REPO_ROOT / "architectures" / "master"
    buf = io.StringIO()
    with redirect_stdout(buf):
        _print_boot_stamp(arch_root=arch_root)
    third = buf.getvalue().splitlines()[2]
    m = re.match(
        r"^\[train_dsl\] arch_dsl_sha256 ([0-9a-f\-]{1,64})\s*(\([^)]+\))?$",
        third,
    )
    assert m, f"arch_dsl_sha256 line malformed: {third!r}"
    # For the real rcc_bowtie folder the SHA must be 64 hex chars (not
    # the '-' fallback used when arch_root is None)
    assert len(m.group(1)) == 64, (
        f"expected 64-hex SHA for real arch, got {m.group(1)!r}"
    )


def test_arch_dsl_sha256_is_deterministic_for_same_arch():
    """Two calls on the same arch_root must produce the same SHA."""
    from neuroslm.train_dsl import _print_boot_stamp

    arch_root = REPO_ROOT / "architectures" / "master"

    def _capture_third_line():
        buf = io.StringIO()
        with redirect_stdout(buf):
            _print_boot_stamp(arch_root=arch_root)
        return buf.getvalue().splitlines()[2]

    a = _capture_third_line()
    b = _capture_third_line()
    assert a == b, (
        f"arch SHA is not deterministic:\n  a={a}\n  b={b}"
    )


def test_print_boot_stamp_tolerates_missing_arch_root():
    """When ``arch_root=None`` (DNA-only mode, or test stub), the third
    line must still print but with a sentinel value instead of crashing.
    Boot stamp printing must never fail — better to lose the field than
    the whole boot record."""
    from neuroslm.train_dsl import _print_boot_stamp

    buf = io.StringIO()
    with redirect_stdout(buf):
        _print_boot_stamp(arch_root=None)
    third = buf.getvalue().splitlines()[2]
    # The line must be present and carry the sentinel '-'
    assert third.startswith("[train_dsl] arch_dsl_sha256 "), third
    assert " - " in third or third.endswith(" -"), (
        f"arch line missing sentinel for None arch_root: {third!r}"
    )


# ─────────────────────────────────────────────────────────────────
# Contract B — `_arch_dsl_sha256` is the SHA-256 hash function
# ─────────────────────────────────────────────────────────────────


def test_arch_dsl_sha256_helper_returns_64_hex_chars():
    """The helper must return a string of exactly 64 hex chars (SHA-256)."""
    from neuroslm.train_dsl import _arch_dsl_sha256

    arch_root = REPO_ROOT / "architectures" / "master"
    sha = _arch_dsl_sha256(arch_root)
    assert isinstance(sha, str)
    assert len(sha) == 64, f"expected 64-hex SHA, got len {len(sha)}: {sha}"
    assert re.fullmatch(r"[0-9a-f]{64}", sha), f"not hex: {sha!r}"


def test_arch_dsl_sha256_changes_when_dsl_changes(tmp_path):
    """Bit-changes in arch.neuro must change the SHA. Catches the
    failure mode where the hash is computed on a sub-field instead of
    the canonical unfolded bytes."""
    from neuroslm.train_dsl import _arch_dsl_sha256

    arch_a = tmp_path / "a"
    arch_a.mkdir()
    (arch_a / "arch.neuro").write_text("architecture foo { d_model = 256 }\n")
    arch_b = tmp_path / "b"
    arch_b.mkdir()
    (arch_b / "arch.neuro").write_text("architecture foo { d_model = 512 }\n")

    sha_a = _arch_dsl_sha256(arch_a)
    sha_b = _arch_dsl_sha256(arch_b)
    assert sha_a != sha_b, (
        f"SHA didn't change between distinct DSL bodies: both={sha_a}"
    )
