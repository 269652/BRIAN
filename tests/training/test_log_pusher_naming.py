"""Log-filename naming contract: must include UTC boot timestamp.

Background
----------
Deploy 40923107 reused instance ID ``38569395`` (vast.ai recycles IDs
across destroyed instances). The previous log at
``logs/vast/38569395_..._step0of2k.log`` was silently overwritten on
relaunch, destroying forensic evidence of the prior crash mode.

The fix is a UTC timestamp prefix on every log filename so reused
instance ids never alias. The format below is the contract that
``brian analyze-log`` and the log-pusher must both honour:

    logs/vast/<UTC_YYYYMMDDTHHMMSSZ>_<instance>_<arch>_<params>_<label>_step<cur>of<target>.log

The timestamp is the **boot** time (when the train script started),
not the push time, so all snapshots from the same run share the same
prefix and the directory groups them visually.
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
# Contract — UTC boot timestamp is in the log filename
# ─────────────────────────────────────────────────────────────────


def test_log_pusher_compose_filename_includes_utc_timestamp_prefix():
    """The composed filename template must prepend a ``UTC_<stamp>_``
    prefix so two runs sharing an instance id never alias.

    Specifically we check that ``_compose_logfile`` either references a
    ``BOOT_TIMESTAMP`` env var OR computes one with
    ``date -u +%Y%m%dT%H%M%SZ`` so the contract is self-enforcing
    inside the script.
    """
    body = _read_script()
    # Either form is acceptable
    has_env = "BOOT_TIMESTAMP" in body
    has_date = re.search(
        r"date\s+-u\s+\+%Y%m%dT%H%M%SZ", body
    ) is not None
    assert has_env or has_date, (
        "log_pusher.sh must either accept a BOOT_TIMESTAMP env var "
        "or compute one via `date -u +%Y%m%dT%H%M%SZ`; otherwise "
        "two runs on the same instance can produce filename "
        "collisions (regression: deploy 40923107 vs 40921910 on "
        "instance 38569395)."
    )


def test_log_pusher_filename_template_uses_timestamp_prefix():
    """The actual ``logs/vast/...`` echo path in ``_compose_logfile``
    must place the timestamp BEFORE the instance id so directory
    listings sort chronologically and the human eye groups runs by
    when they started."""
    body = _read_script()
    # Find the echo line inside _compose_logfile that produces the
    # filename. Pin the order: timestamp before instance.
    m = re.search(
        r"echo\s+\"?logs/vast/\$\{?([A-Za-z_]+)\}?_\$\{?([A-Za-z_]+)\}?",
        body,
    )
    assert m, (
        "couldn't find the `echo logs/vast/${X}_${Y}...` line in "
        "log_pusher.sh — has the file structure changed?"
    )
    first_var, second_var = m.group(1), m.group(2)
    # First field should be the timestamp variable (BOOT_TIMESTAMP or
    # an equivalent). Second should be the instance id.
    assert "TIMESTAMP" in first_var.upper() or "STAMP" in first_var.upper(), (
        f"first filename field is {first_var!r}, expected a "
        "timestamp variable. The timestamp must come first so "
        "`ls logs/vast/` sorts chronologically."
    )
    assert "INSTANCE" in second_var.upper(), (
        f"second filename field is {second_var!r}, expected the "
        "instance id."
    )
