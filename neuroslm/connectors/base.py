# -*- coding: utf-8 -*-
"""Platform-agnostic deploy configuration, job tracking, and connector base.

This module defines the contract every cloud connector implements:

* :class:`DeployConfig` — the launch request (what to train, where to push).
* :class:`JobInfo` — the persisted record of a launched run.
* :class:`JobStatus` — coarse lifecycle states normalised across platforms.
* :class:`BaseConnector` — abstract base that defines ``launch``,
  ``list_jobs``, ``tail_logs``, ``status`` (the polling surface used by
  ``brian ps`` to monitor remote runs).

Job registry layout
───────────────────
Every successful ``launch()`` writes ``.brian/jobs/<job_id>.json`` describing
the run. The file is the source of truth for ``brian ps`` to reconnect to
remote handles (e.g. Lightning Studios) without needing the original
``DeployConfig`` in scope. Removed when the user runs ``brian destroy
<job_id>`` or after a manual cleanup pass.
"""
from __future__ import annotations

import json
import os
import time
from abc import ABC, abstractmethod
from dataclasses import asdict, dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional


# ── Job registry on disk ─────────────────────────────────────────────
#
# Path is computed at module import — but we resolve it lazily inside the
# helpers so tests can monkeypatch ``BRIAN_JOBS_DIR`` to redirect writes.
_REPO_ROOT = Path(__file__).resolve().parents[2]


def _jobs_dir() -> Path:
    """Return the on-disk jobs directory, creating it if missing.

    Honours ``BRIAN_JOBS_DIR`` for tests / alternate workspaces.
    """
    override = os.environ.get("BRIAN_JOBS_DIR")
    base = Path(override) if override else _REPO_ROOT / ".brian" / "jobs"
    base.mkdir(parents=True, exist_ok=True)
    return base


class JobStatus(str, Enum):
    """Normalised lifecycle state, shared across platforms.

    Connectors map their native status enum (e.g. Lightning's
    ``NotCreated | Pending | Running | Stopping | Stopped | Completed
    | Failed``) onto these seven values so ``brian ps`` can render a
    consistent column regardless of which cloud hosts the run.
    """
    PENDING = "pending"        # provisioned but not yet running
    STARTING = "starting"      # boot / cold-start in progress
    RUNNING = "running"        # actively training
    STOPPING = "stopping"      # shutdown in progress
    STOPPED = "stopped"        # halted; can be resumed
    COMPLETED = "completed"    # training reached target steps
    FAILED = "failed"          # crashed / errored out
    UNKNOWN = "unknown"        # connector can't determine current state


@dataclass
class JobInfo:
    """One launched training run, as seen by ``brian ps``.

    Persisted to ``.brian/jobs/<job_id>.json``. Connectors read it back
    on every ``list_jobs()`` so the SDK handles can be reconstructed
    (``Studio(name=..., teamspace=...)``) without holding state across
    process boundaries.

    Fields are deliberately stringly-typed for forwards-compat with
    JSON serialisation. Numeric / datetime fields are stored as ints
    (epoch seconds) or strings.
    """
    # Identity (always present)
    job_id: str                          # short, human-readable id
    platform: str                        # connector platform_name()
    label: str                           # user-supplied label or "(none)"
    status: str = JobStatus.UNKNOWN.value

    # Provisioning details
    machine: str = ""                    # GPU tier (T4, A10G, …)
    branch: str = ""
    arch: str = ""                       # path used at launch
    steps: int = 0

    # Platform-specific handles (used to re-attach for polling)
    studio_name: str = ""                # Lightning Studio name
    teamspace: str = ""                  # Lightning teamspace
    host: str = ""                       # Vast SSH host / Lightning user

    # Telemetry
    started_at: int = 0                  # epoch seconds
    log_path: str = ""                   # remote path to the live train log
    source_dna: str = ""                 # original DNA path (if DNA mode)
    extra: Dict[str, Any] = field(default_factory=dict)

    # ── JSON round-trip helpers ───────────────────────────────────

    def to_json(self) -> Dict[str, Any]:
        """Return a JSON-serialisable dict (no enums, no Paths)."""
        return asdict(self)

    @classmethod
    def from_json(cls, data: Dict[str, Any]) -> "JobInfo":
        """Inverse of :meth:`to_json` — tolerant of missing fields."""
        # Filter to known fields so future schema additions in the
        # JSON don't crash older code paths.
        known = {f for f in cls.__dataclass_fields__}
        return cls(**{k: v for k, v in data.items() if k in known})


def register_job(info: JobInfo) -> Path:
    """Persist *info* to ``.brian/jobs/<job_id>.json`` and return the path.

    Called by connectors at the end of a successful ``launch()`` so
    ``brian ps`` can later re-attach to the remote run.
    """
    if not info.started_at:
        info.started_at = int(time.time())
    path = _jobs_dir() / f"{info.job_id}.json"
    path.write_text(json.dumps(info.to_json(), indent=2, sort_keys=True),
                    encoding="utf-8")
    return path


def load_jobs(platform: Optional[str] = None) -> List[JobInfo]:
    """Return all persisted :class:`JobInfo` records, optionally filtered.

    Args:
        platform: When given, only jobs whose ``platform`` matches are
            returned. Useful for connector-specific list_jobs() impls.

    Returns:
        Jobs sorted by ``started_at`` descending (newest first).
        Malformed files are silently skipped.
    """
    out: List[JobInfo] = []
    for p in _jobs_dir().glob("*.json"):
        try:
            data = json.loads(p.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        try:
            info = JobInfo.from_json(data)
        except TypeError:
            continue
        if platform and info.platform != platform:
            continue
        out.append(info)
    out.sort(key=lambda j: j.started_at, reverse=True)
    return out


def load_job(job_id: str) -> Optional[JobInfo]:
    """Return the single :class:`JobInfo` for *job_id*, or ``None``."""
    path = _jobs_dir() / f"{job_id}.json"
    if not path.is_file():
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        return JobInfo.from_json(data)
    except (OSError, json.JSONDecodeError, TypeError):
        return None


def remove_job(job_id: str) -> bool:
    """Delete the on-disk record for *job_id*. Returns True if removed."""
    path = _jobs_dir() / f"{job_id}.json"
    if path.is_file():
        path.unlink()
        return True
    return False


@dataclass
class DeployConfig:
    """Everything a connector needs to launch one training run.

    ``arch`` is always a resolved local path by the time the connector
    sees it. If the original request was DNA-mode, ``cmd_deploy`` compiles
    the workspace and sets ``arch = str(workspace.arch_root)`` BEFORE
    calling ``connector.launch()``. ``source_dna`` carries the original
    DNA path for telemetry labels only.

    Cadence fields (``log_every``, ``save_every``, ``push_every``,
    ``ood_every``) default to 0, meaning "use the trainer's own default".
    Connectors skip zero-valued cadence fields to avoid overriding
    defaults that the on-box training loop already manages.
    """
    steps: int
    branch: Optional[str] = None
    arch: Optional[str] = None
    scale: Optional[str] = None
    label: Optional[str] = None
    resume_from: Optional[str] = None
    source_dna: Optional[str] = None
    ood_every: int = 0
    log_every: int = 0
    save_every: int = 0
    push_every: int = 0
    push_backend: str = "hf"
    hf_repo_id: str = "moritzroessler/BRIAN"
    push_optimizer: bool = False
    seq_len: int = 0       # 0 = use trainer default (128); set e.g. 256 for P2
    batch_size: int = 0    # 0 = use trainer default; forwarded as --batch N
    extra_env: dict = field(default_factory=dict)


class BaseConnector(ABC):
    """Abstract base for training platform connectors.

    A connector translates a :class:`DeployConfig` into a real launch
    on a specific cloud platform (vast.ai, lightning.ai, …). Each
    implementation is registered in :mod:`neuroslm.connectors` and
    selected by the ``[deploy] platform`` key in ``brian.toml`` or the
    ``--platform`` CLI flag.

    Connectors that support post-launch polling (``brian ps`` per-platform
    integration) override :meth:`list_jobs`, :meth:`status`, and
    :meth:`tail_logs`. The default no-op implementations make these
    optional so legacy connectors keep working transparently.
    """

    @classmethod
    @abstractmethod
    def platform_name(cls) -> str:
        """Stable identifier used in ``brian.toml`` and ``--platform``."""

    @abstractmethod
    def launch(self, config: DeployConfig) -> int:
        """Launch a training run. Returns an exit code (0 = success).

        Connectors that support detached launches (e.g. SSH-style remote
        exec) should call :func:`register_job` with a populated
        :class:`JobInfo` BEFORE returning so ``brian ps`` can re-attach.
        """

    # ── Polling surface (used by `brian ps`) ────────────────────────
    #
    # Default impls return empty / unknown — connectors with a real
    # remote handle (Lightning Studios, Vast SSH, …) override these.

    def list_jobs(self) -> List[JobInfo]:
        """Return live :class:`JobInfo` records for this platform.

        Default reads the on-disk registry and filters by
        :meth:`platform_name`. Override to enrich each record with
        live status / log tail / cost.
        """
        return load_jobs(platform=self.platform_name())

    def status(self, job_id: str) -> JobStatus:
        """Return the current :class:`JobStatus` for *job_id*.

        Default reads the persisted status from disk; override to
        query the platform for the live state.
        """
        info = load_job(job_id)
        if info is None:
            return JobStatus.UNKNOWN
        try:
            return JobStatus(info.status)
        except ValueError:
            return JobStatus.UNKNOWN

    def tail_logs(self, job_id: str, n: int = 200) -> str:
        """Return the last *n* lines of the remote training log.

        Default raises :class:`NotImplementedError`. Connectors that
        can reach the remote host (e.g. Lightning's ``Studio.run``,
        Vast's SSH tunnel) override with a real ``tail -n`` call.
        """
        raise NotImplementedError(
            f"{self.platform_name()} connector does not support log tailing"
        )

    def stop(self, job_id: str) -> int:
        """Stop the running job *job_id*. Returns an exit code.

        Default raises :class:`NotImplementedError`. Connectors override
        to cleanly halt the remote run and update the registry.
        """
        raise NotImplementedError(
            f"{self.platform_name()} connector does not support stop()"
        )
