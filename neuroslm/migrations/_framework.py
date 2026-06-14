"""brian migrate — versioned, ledger-tracked, idempotent repo migrations.

Why exist:
    Repo conventions evolve. Today we want `logs/vast/*.log` reorganized
    into `logs/<date>_<arch>_<sha>/train.log` folders, tomorrow we want
    something else. Hand-running a Python script is fine the first time;
    hand-running it twice (or for a new contributor) is not. This
    framework formalizes the pattern:

      * Migrations live in `neuroslm/migrations/NNNN_<slug>.py`.
      * Each exports `ID`, `DESCRIPTION`, `plan(ctx)`, `apply(ctx, ops)`.
      * A ledger at `.brian/migrations.json` records what was applied.
      * `plan()` is ALWAYS run (even on applied migrations) so we can
        detect drift — i.e. somebody added a file the migration would
        have moved, but the migration was already marked done.

CLI surface (provided by `neuroslm.cli`):

    brian migrate --list                 # status of every migration
    brian migrate <id>                   # dry-run a specific one
    brian migrate <id> --force           # actually apply it
    brian migrate --all                  # dry-run every pending
    brian migrate --all --force          # apply every pending
    brian migrate <id> --rerun --force   # re-apply even if in ledger

Exit codes:
    0  on success (incl. clean dry-run)
    1  on unknown migration id, apply failure, or unhandled drift

Idempotency contract:
    A well-written `plan()` is pure (no I/O writes) and returns the
    *current* operations required to bring the repo to the post-
    migration state. If plan() returns `[]`, the migration is a no-op
    for this repo state, regardless of ledger status.
"""
from __future__ import annotations

import datetime as _dt
import importlib
import importlib.util
import json
import os
import platform
import socket
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Protocol

from neuroslm.references import ReferenceIndex


# ── Data types ────────────────────────────────────────────────────────


@dataclass(frozen=True)
class Op:
    """A single planned operation.

    `kind` is a free-form string — migration authors choose the
    vocabulary (e.g. ``"copy"``, ``"move"``, ``"delete"``, ``"write"``).
    The framework only inspects the *count* of ops, not their semantics.
    """
    kind: str
    src: Optional[Path]
    dst: Optional[Path]
    note: str = ""


@dataclass
class Context:
    """Runtime context handed to every migration's plan/apply.

    * ``root`` — repo root (typically REPO_ROOT)
    * ``refs`` — a built ReferenceIndex; migrations that need to know
      "is this file referenced anywhere?" use ``refs.references(name)``
    * ``dry_run`` — True when the user did NOT pass --force; apply()
      should still be callable but the orchestrator will not invoke it
    * ``force``  — True when the user passed --force
    """
    root: Path
    refs: ReferenceIndex
    dry_run: bool = True
    force: bool = False


class MigrationProtocol(Protocol):
    """The duck-typed interface every migration module must satisfy."""
    ID: str
    DESCRIPTION: str

    def plan(self, ctx: Context) -> List[Op]: ...
    def apply(self, ctx: Context, ops: List[Op]) -> int: ...


# ── Discovery ──────────────────────────────────────────────────────────


def _is_migration_filename(name: str) -> bool:
    """Migration files are `NNNN_<slug>.py`; underscore-prefixed and
    non-.py files are skipped."""
    if not name.endswith(".py"):
        return False
    if name.startswith("_"):
        return False
    if name == "__init__.py":
        return False
    return True


def _load_module_from_path(path: Path) -> Any:
    """Load a Python file as a fresh module by file path.

    We use the stem as the module name. Each call yields a *new* module
    object (we don't cache in `sys.modules` long-term) so tests can
    rebuild migrations between cases without import shadowing.
    """
    spec = importlib.util.spec_from_file_location(path.stem, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"could not load migration spec from {path}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def discover_migrations(pkg_dir: Path) -> List[Any]:
    """Return every migration module under ``pkg_dir``, sorted by ID.

    A "migration module" is any ``*.py`` file (not starting with ``_``)
    that defines ``ID``, ``DESCRIPTION``, ``plan``, and ``apply``.
    """
    out: List[Any] = []
    if not pkg_dir.exists():
        return out
    for entry in sorted(pkg_dir.iterdir()):
        if not entry.is_file() or not _is_migration_filename(entry.name):
            continue
        mod = _load_module_from_path(entry)
        # Sanity check the protocol surface — fail loud on malformed
        # migrations rather than silently dropping them.
        for attr in ("ID", "DESCRIPTION", "plan", "apply"):
            if not hasattr(mod, attr):
                raise AttributeError(
                    f"migration {entry.name} missing required attr `{attr}`"
                )
        out.append(mod)
    out.sort(key=lambda m: m.ID)
    return out


# ── Ledger ─────────────────────────────────────────────────────────────


_LEDGER_DIR = ".brian"
_LEDGER_FILE = "migrations.json"
_LEDGER_VERSION = 1


class Ledger:
    """JSON-backed record of applied migrations.

    Schema::

        {
          "version": 1,
          "applied": {
            "<id>": {
              "applied_at_utc": "2026-01-27T14:30:00Z",
              "commit": "<git sha or unknown>",
              "ops_applied": <int>,
              "host": "<hostname>"
            },
            ...
          }
        }
    """

    def __init__(self, root: Path) -> None:
        self.root = root
        self.path = root / _LEDGER_DIR / _LEDGER_FILE

    def applied(self) -> Dict[str, Dict[str, Any]]:
        if not self.path.exists():
            return {}
        try:
            data = json.loads(self.path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return {}
        return dict(data.get("applied", {}))

    def record(self, *, mig_id: str, commit: str, ops_applied: int) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        current = self.applied()
        current[mig_id] = {
            "applied_at_utc": _dt.datetime.now(_dt.timezone.utc)
                .replace(microsecond=0).isoformat().replace("+00:00", "Z"),
            "commit": commit,
            "ops_applied": int(ops_applied),
            "host": socket.gethostname() or platform.node() or "unknown",
        }
        payload = {"version": _LEDGER_VERSION, "applied": current}
        self.path.write_text(
            json.dumps(payload, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )


# ── Status ─────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class Status:
    """Status of a single migration relative to the current repo state."""
    mig_id: str
    description: str
    kind: str            # APPLIED | PENDING | DRIFT | NOOP_PENDING
    planned_ops: int
    ledger_entry: Optional[Dict[str, Any]] = None


_STATUS_KINDS = ("APPLIED", "PENDING", "DRIFT", "NOOP_PENDING")


def _classify(in_ledger: bool, planned_ops: int) -> str:
    if in_ledger and planned_ops == 0:
        return "APPLIED"
    if in_ledger and planned_ops > 0:
        return "DRIFT"
    if not in_ledger and planned_ops == 0:
        return "NOOP_PENDING"
    return "PENDING"  # not in_ledger and planned_ops > 0


def status_all(pkg_dir: Path, ctx: Context) -> Dict[str, Status]:
    """Status of EVERY discovered migration. Always invokes plan()."""
    led = Ledger(ctx.root).applied()
    out: Dict[str, Status] = {}
    for mod in discover_migrations(pkg_dir):
        ops = mod.plan(ctx)
        out[mod.ID] = Status(
            mig_id=mod.ID,
            description=str(mod.DESCRIPTION),
            kind=_classify(mod.ID in led, len(ops)),
            planned_ops=len(ops),
            ledger_entry=led.get(mod.ID),
        )
    return out


# ── Git helper ─────────────────────────────────────────────────────────


def _git_head(root: Path) -> str:
    try:
        r = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=root, capture_output=True, text=True,
            check=False, timeout=10,
        )
        if r.returncode == 0:
            return r.stdout.strip() or "unknown"
    except (OSError, FileNotFoundError, subprocess.TimeoutExpired):
        pass
    return "unknown"


# ── Orchestration ──────────────────────────────────────────────────────


def run_one(
    pkg_dir: Path,
    mig_id: str,
    ctx: Context,
    *,
    rerun: bool = False,
) -> int:
    """Run ONE migration. Returns exit code.

    Semantics:
        * Unknown ``mig_id``       -> exit 1
        * Already in ledger, NOT --rerun:
            - Dry-run: print noop and return 0
            - --force: print noop, do not re-apply, return 0
        * Dry-run with ops planned: print ops, return 0
        * --force with ops planned: apply(), record ledger, return 0
        * --rerun --force         : apply() regardless of ledger state,
                                    overwrite ledger entry
    """
    mods = {m.ID: m for m in discover_migrations(pkg_dir)}
    mod = mods.get(mig_id)
    if mod is None:
        print(f"[migrate] unknown migration id: {mig_id}", flush=True)
        return 1

    led = Ledger(ctx.root)
    already_applied = mig_id in led.applied()

    ops = mod.plan(ctx)
    print(
        f"[migrate] {mig_id}: {mod.DESCRIPTION}  "
        f"({len(ops)} op{'s' if len(ops) != 1 else ''} planned)",
        flush=True,
    )

    # Already applied, not rerunning -> no-op
    if already_applied and not rerun:
        kind = _classify(True, len(ops))
        print(f"[migrate] {mig_id}: status={kind} (already in ledger)", flush=True)
        return 0

    # Dry-run: print the plan, no I/O writes, no ledger
    if ctx.dry_run or not ctx.force:
        for op in ops[:50]:
            print(
                f"  - {op.kind}: "
                f"{_safe(op.src)} -> {_safe(op.dst)}  {op.note}".rstrip(),
                flush=True,
            )
        if len(ops) > 50:
            print(f"  ... and {len(ops) - 50} more", flush=True)
        print(f"[migrate] {mig_id}: DRY-RUN (use --force to apply)", flush=True)
        return 0

    # Apply
    try:
        applied = int(mod.apply(ctx, ops))
    except Exception as exc:  # noqa: BLE001
        print(f"[migrate] {mig_id}: apply() FAILED: {exc!r}", flush=True)
        return 1

    led.record(mig_id=mig_id, commit=_git_head(ctx.root), ops_applied=applied)
    print(
        f"[migrate] {mig_id}: APPLIED ({applied} op{'s' if applied != 1 else ''})",
        flush=True,
    )
    return 0


def run_all(pkg_dir: Path, ctx: Context) -> int:
    """Run every PENDING migration in order. Returns 0 if all clean."""
    statuses = status_all(pkg_dir, ctx)
    rc = 0
    for mig_id in sorted(statuses):
        s = statuses[mig_id]
        if s.kind in ("PENDING", "NOOP_PENDING"):
            r = run_one(pkg_dir, mig_id, ctx)
            if r != 0:
                rc = r
    return rc


# ── CLI helpers (called by neuroslm.cli) ───────────────────────────────


_STATUS_GLYPH = {
    "APPLIED":      "[OK]    ",
    "PENDING":      "[PEND]  ",
    "DRIFT":        "[DRIFT] ",
    "NOOP_PENDING": "[NOOP]  ",
}


def cli_list(pkg_dir: Path, root: Path) -> int:
    """Print status of every migration. Always exit 0 unless errors."""
    ctx = Context(root=root, refs=ReferenceIndex(),
                  dry_run=True, force=False)
    statuses = status_all(pkg_dir, ctx)
    if not statuses:
        print("[migrate] no migrations discovered", flush=True)
        return 0

    print(f"[migrate] {len(statuses)} migration(s) discovered:", flush=True)
    for mig_id in sorted(statuses):
        s = statuses[mig_id]
        glyph = _STATUS_GLYPH.get(s.kind, "[?]     ")
        ops_str = f"{s.planned_ops:4d} op(s)" if s.planned_ops else "  no ops"
        # Both the glyph and the kind name are printed so machine-
        # readable consumers (and the test suite) can grep for the
        # status word, while humans get a compact aligned column.
        print(
            f"  {glyph}{s.kind:12s}  {s.mig_id:32s}  "
            f"{ops_str}  {s.description}",
            flush=True,
        )
    return 0


# ── Internals ──────────────────────────────────────────────────────────


def _safe(p: Optional[Path]) -> str:
    if p is None:
        return "-"
    try:
        return str(p)
    except Exception:  # noqa: BLE001
        return "<?>"


__all__ = [
    "Op",
    "Context",
    "Ledger",
    "Status",
    "MigrationProtocol",
    "discover_migrations",
    "status_all",
    "run_one",
    "run_all",
    "cli_list",
]
