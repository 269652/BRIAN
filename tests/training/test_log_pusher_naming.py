"""Log-filename naming contract: per-run folder layout that matches
the ``0001_logs_to_run_folders`` migration's output.

Background
----------
Deploy 40923107 reused instance ID ``38569395`` (vast.ai recycles IDs
across destroyed instances). The previous log at
``logs/vast/38569395_..._step0of2k.log`` was silently overwritten on
relaunch, destroying forensic evidence of the prior crash mode.

The fix is a UTC timestamp prefix on every log filename so reused
instance ids never alias.

Layout history:

    pre-2026-06-15  flat:    logs/vast/<stamp>_<id>_..._stepNofN.log
    2026-06-15-am   nested:  logs/vast/<YYYYMMDD>/<ARCH>/<stamp>_<id>_..._stepNofN.log
    2026-06-15-pm   per-run: logs/<YYYYMMDD>-<HHMMSS>_<arch>_<short-sha>/train.log

The per-run layout matches what ``brian migrate 0001_logs_to_run_folders``
writes when normalising old logs, so on-box writes are immediately
migration-clean and ``git pull`` on a workstation just gets the
freshly-named folder. Stable filename per run also means no
``_PREV_LOG`` cleanup race.
"""
from __future__ import annotations

import re
from pathlib import Path


SCRIPT_PATH = (
    Path(__file__).resolve().parent.parent.parent
    / "scripts" / "log_pusher.sh"
)


def _read_script() -> str:
    return SCRIPT_PATH.read_text(encoding="utf-8")


# ─────────────────────────────────────────────────────────────────
# Contract N1 — UTC boot timestamp is referenced in the script
# ─────────────────────────────────────────────────────────────────

def test_log_pusher_compose_filename_includes_utc_timestamp_prefix():
    """The composed filename template must use a ``BOOT_TIMESTAMP``
    env var (or compute one via ``date -u``) so two runs sharing an
    instance id never alias."""
    body = _read_script()
    has_env = "BOOT_TIMESTAMP" in body
    has_date = re.search(r"date\s+-u\s+\+%Y%m%dT%H%M%SZ", body) is not None
    assert has_env or has_date, (
        "log_pusher.sh must either accept a BOOT_TIMESTAMP env var "
        "or compute one via `date -u +%Y%m%dT%H%M%SZ`"
    )


# ─────────────────────────────────────────────────────────────────
# Contract N2 — per-run folder layout: logs/<date>_<arch>_<sha>/train.log
# ─────────────────────────────────────────────────────────────────

def test_log_pusher_uses_per_run_folder_layout():
    """_compose_logfile must emit a path of the form
    ``logs/<YYYYMMDD>-<HHMMSS>_<arch>_<sha>/train.log`` matching the
    ``0001_logs_to_run_folders`` migration's output."""
    body = _read_script()
    # The echo in _compose_logfile should end with /train.log
    assert re.search(
        r'echo\s+"\$\{?folder\}?/train\.log"',
        body,
    ), (
        "_compose_logfile must echo \"${folder}/train.log\" — that's "
        "the canonical per-run folder layout matching what the "
        "0001_logs_to_run_folders migration writes. Found neither "
        "${folder}/train.log nor 'train.log' as the leaf filename."
    )


def test_log_pusher_folder_uses_date_arch_sha_format():
    """The folder name MUST encode date_token + arch + short-sha so it
    matches the migration's ``_new_folder_name`` and so two runs of
    the same arch at different boot times don't alias."""
    body = _read_script()
    # Look for the folder assembly: must use BOOT_TIMESTAMP slices for
    # date+time, ARCH_NAME, and a git short sha (via rev-parse).
    assert "${BOOT_TIMESTAMP:0:8}" in body, (
        "_compose_logfile must slice the YYYYMMDD date from "
        "BOOT_TIMESTAMP via ${BOOT_TIMESTAMP:0:8} for the folder name."
    )
    assert "${BOOT_TIMESTAMP:9:6}" in body, (
        "_compose_logfile must slice the HHMMSS time from BOOT_TIMESTAMP "
        "via ${BOOT_TIMESTAMP:9:6} (positions 9-14, skipping the 'T') "
        "so the folder name carries seconds-resolution and matches "
        "the migration's date_token format."
    )
    assert re.search(r'git\s+rev-parse\s+--short=\d+\s+HEAD', body), (
        "_compose_logfile must include a git short sha (via "
        "`git rev-parse --short=N HEAD`) in the folder name so the "
        "folder identifies the exact deployed commit — matches the "
        "migration's <date>_<arch>_<sha> convention."
    )
    assert "${ARCH_NAME}" in body, (
        "_compose_logfile must include ${ARCH_NAME} in the folder "
        "name so runs of different architectures land in distinct "
        "folders."
    )


# ─────────────────────────────────────────────────────────────────
# Contract N3 — stable filename per run (no _PREV_LOG cleanup)
# ─────────────────────────────────────────────────────────────────

def test_log_pusher_no_prev_log_cleanup_with_stable_filenames():
    """Since the new layout has ONE filename per run that never mutates,
    there must be no ``_PREV_LOG`` tracking variable. Its presence
    indicates leftover state from the old mutating-filename layout."""
    body = _read_script()
    # _PREV_LOG should NOT be referenced anywhere except possibly in a
    # comment explaining its removal — i.e. it must never appear as a
    # variable assignment or read.
    assignment = re.search(r'^\s*_PREV_LOG\s*=', body, re.MULTILINE)
    dereference = re.search(r'\$\{?_PREV_LOG\}?', body)
    assert assignment is None and dereference is None, (
        "_PREV_LOG must not appear in log_pusher.sh — it was used by "
        "the pre-2026-06-15 mutating-filename layout to track the "
        "previous step's filename for `git rm` cleanup. With the new "
        "stable per-run filename, there's nothing to clean up."
    )
