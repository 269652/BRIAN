"""Log-filename naming contract: must include UTC boot timestamp AND
be placed under logs/vast/<YYYYMMDD>/<ARCH_NAME>/ subdirectories.

Background
----------
Deploy 40923107 reused instance ID ``38569395`` (vast.ai recycles IDs
across destroyed instances). The previous log at
``logs/vast/38569395_..._step0of2k.log`` was silently overwritten on
relaunch, destroying forensic evidence of the prior crash mode.

The fix is a UTC timestamp prefix on every log filename so reused
instance ids never alias.

Additionally, flat logs/vast/ became unwieldy as the number of runs grew.
Logs are now written into date + arch subdirectories so the repo tree
groups them naturally:

    logs/vast/<YYYYMMDD>/<ARCH_NAME>/<UTC_YYYYMMDDTHHMMSSZ>_<instance>_<arch>_<params>_<label>_step<cur>of<target>.log

The YYYYMMDD date is the first 8 characters of BOOT_TIMESTAMP.
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
# Contract N2 — subdirectory layout: logs/vast/<DATE>/<ARCH>/
# ─────────────────────────────────────────────────────────────────

def test_log_pusher_filename_uses_date_subfolder():
    """_compose_logfile must extract the YYYYMMDD date from BOOT_TIMESTAMP
    and use it as the first subdirectory under logs/vast/ so runs group
    by calendar day in the repo tree."""
    body = _read_script()
    # bash substring: ${BOOT_TIMESTAMP:0:8} → "20260615"
    assert "${BOOT_TIMESTAMP:0:8}" in body, (
        "_compose_logfile must slice the YYYYMMDD date from BOOT_TIMESTAMP "
        "via ${BOOT_TIMESTAMP:0:8} to use as the date subdirectory under "
        "logs/vast/.  Current script writes logs flat into logs/vast/ instead."
    )


def test_log_pusher_filename_uses_arch_subfolder():
    """The log path must have the form logs/vast/<date>/<arch>/<file>.log
    so logs from different architectures land in separate subdirectories."""
    body = _read_script()
    # The echo in _compose_logfile should be:
    #   echo "logs/vast/${date_var}/${ARCH_NAME}/${BOOT_TIMESTAMP}_..."
    m = re.search(
        r'echo\s+"logs/vast/\$\{?(\w+)\}?/\$\{?(ARCH[A-Za-z_]*)\}?/',
        body,
    )
    assert m, (
        "couldn't find 'logs/vast/${date_var}/${ARCH_NAME}/...' in "
        "_compose_logfile.  Log files must go into "
        "logs/vast/<date>/<arch>/ subdirectories — current script "
        "writes them flat."
    )
    date_var = m.group(1)
    assert "date" in date_var.lower() or "timestamp" in date_var.lower(), (
        f"first subdirectory variable is ${{{date_var}}}, expected a "
        "date variable (e.g. date_dir extracted from BOOT_TIMESTAMP)"
    )


def test_log_pusher_timestamp_still_in_filename():
    """BOOT_TIMESTAMP must still appear in the leaf FILENAME (after the
    subdirs) so files within an arch folder sort chronologically."""
    body = _read_script()
    # After the two subdirs the filename must start with BOOT_TIMESTAMP
    assert re.search(
        r'logs/vast/[^"]+/[^"]+/\$\{?BOOT_TIMESTAMP\}?_\$\{?INSTANCE',
        body,
    ), (
        "the leaf filename inside logs/vast/<date>/<arch>/ must still "
        "begin with ${BOOT_TIMESTAMP}_${INSTANCE_ID}_… so that "
        "ls within the arch subfolder sorts chronologically and "
        "reused instance ids never collide."
    )
