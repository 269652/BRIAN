# -*- coding: utf-8 -*-
"""Lightning AI training connector — SSH-style remote provisioning.

Launches a training run on a Lightning AI Studio over the SDK's remote-exec
channel (which is internally SSH/RPC — the public surface is ``Studio.run``
and ``Studio.run_and_detach``). The flow is::

    1.   Bootstrap secrets from .env → os.environ.
    2.   Resolve the authed user + teamspace (auto if exactly one).
    3.   Start (or reuse) a Studio on the requested machine tier.
    4.   ``Studio.set_env`` to push HF_TOKEN / GITHUB_PAT / branch / arch.
    5.   ``Studio.run_with_exit_code`` — synchronously CLONE the repo,
         checkout the branch, ``pip install -e .[ml]`` (plus
         ``requirements.txt`` as belt-and-braces) so training imports
         resolve. The setup VERIFIES the critical ML imports (torch,
         transformers, tiktoken, einops) and aborts with exit 2 if
         any are missing — better to fail loud here than crash inside
         the detached training run.
    6.   ``Studio.run_and_detach`` — fire-and-forget the training command,
         redirected to ``~/brian/logs/run-<job_id>.log`` so it survives
         the SDK session disconnect.
    7.   Persist ``.brian/jobs/<job_id>.json`` so ``brian ps`` can re-attach.
    8.   Return immediately (the Studio keeps running until ``brian stop``).

Prerequisites
─────────────
  pip install lightning-sdk        # Lightning AI SDK
  # then either:
  lightning login                  # CLI flow (writes ~/.lightning/credentials)
  # OR set in .env / shell:
  LIGHTNING_USER_ID=<api-key-uuid>
  LIGHTNING_API_KEY=<your-api-key>

Authentication
──────────────
On every ``launch()`` we call :func:`neuroslm.utils.secrets.bootstrap_secrets`
which walks the chain ``os.environ → Colab → Kaggle → .env (CWD upward)``
to populate ``LIGHTNING_USER_ID`` and ``LIGHTNING_API_KEY`` before the
SDK reads them.  Friendly aliases are accepted so older ``.env`` files
keep working::

  LIGHTNING_API_KEY  ← LIGHTNING_AI, LIGHTNING_TOKEN, LIGHTNING_AUTH_TOKEN
  LIGHTNING_USER_ID  ← LIGHTNING_USERNAME, LIGHTNING_USER

Repo cloning on the Studio
──────────────────────────
The Studio needs read access to the repo.  We use HTTPS with a token to
avoid needing to forward SSH keys:

  https://x-access-token:${GITHUB_PAT}@github.com/<owner>/<repo>.git

``GITHUB_PAT`` is sourced from the same secrets chain as the Lightning
credentials.  Set it as a fine-grained PAT scoped to the repo
(``contents:read`` is enough for training; ``contents:write`` only if
the in-Studio checkpoint pusher should also push back to the repo).

Machine selection (highest precedence wins)
───────────────────────────────────────────
  1. ``config.extra_env["LIGHTNING_MACHINE"]``  (CLI ``--machine``,
     brian.toml ``[deploy].machine``, or any caller-supplied override)
  2. ``config.scale``                            (CLI ``--scale``,
     re-purposed as a GPU hint when no ``--machine`` is given)
  3. ``Machine.T4``                              (sensible default)

Substring match — case-insensitive.

Teamspace resolution
────────────────────
  1. ``config.extra_env["LIGHTNING_TEAMSPACE"]`` / ``LIGHTNING_TEAMSPACE`` env
  2. The user's only teamspace (auto-resolved via the SDK)
  3. Hard error with a chooser hint when the user has multiple

Studio naming
─────────────
  brian-{config.label}   when label is set
  brian-train            fallback

Job tracking
────────────
Every successful launch writes ``.brian/jobs/<job_id>.json`` so
``brian ps --platform lightning`` can re-attach and stream logs without
the user having to remember Studio names.
"""
from __future__ import annotations

import os
import re
import shlex
import sys
import time
from pathlib import Path
from typing import TYPE_CHECKING, List, Optional, Tuple

from neuroslm.connectors.base import (
    BaseConnector,
    DeployConfig,
    JobInfo,
    JobStatus,
    load_job,
    load_jobs,
    register_job,
    remove_job,
)

if TYPE_CHECKING:
    try:  # ``lightning_sdk`` is the modern wheel name (no dot).
        from lightning_sdk import Machine  # type: ignore[import]
    except ImportError:  # Fall back to the legacy namespaced import.
        from lightning.sdk import Machine  # type: ignore[import]


# ── Remote layout on the Studio ──────────────────────────────────────
# Everything BRIAN-related lives under ``~/brian`` so the Studio's
# default ``/teamspace/studios`` layout stays uncluttered. The repo
# itself sits at ``~/brian/repo`` and live logs at ``~/brian/logs``.
REMOTE_BASE = "~/brian"
REMOTE_REPO = REMOTE_BASE + "/repo"
REMOTE_LOGS = REMOTE_BASE + "/logs"


# ── Status mapping: Lightning SDK Status → JobStatus ─────────────────
_STATUS_MAP = {
    "NotCreated": JobStatus.STOPPED,
    "Pending": JobStatus.STARTING,
    "Running": JobStatus.RUNNING,
    "Stopping": JobStatus.STOPPING,
    "Stopped": JobStatus.STOPPED,
    "Completed": JobStatus.COMPLETED,
    "Failed": JobStatus.FAILED,
}


def _import_lightning_sdk():
    """Import (Studio, Machine, sdk_name) from whichever SDK is installed.

    The wheel was renamed from ``lightning.sdk`` to ``lightning_sdk``
    (no dot) in mid-2024. We try the modern name first, then the legacy
    one, so the same connector code works against both releases.
    """
    try:
        from lightning_sdk import Studio, Machine  # type: ignore[import]
        return Studio, Machine, "lightning_sdk"
    except ImportError:
        pass
    try:
        from lightning.sdk import Studio, Machine  # type: ignore[import]
        return Studio, Machine, "lightning.sdk"
    except ImportError:
        return None, None, None


def _bootstrap_lightning_secrets() -> None:
    """Walk .env/Colab/Kaggle/env into os.environ for the SDK's sake.

    Lightning SDK only inspects ``os.environ`` directly, so a token
    sitting in ``.env`` is invisible unless we bootstrap it into the
    process environment first.

    Friendly aliases are accepted for the BRIAN-side ergonomics —
    older ``.env`` files used ``LIGHTNING_AI`` for the API key.
    """
    try:
        from neuroslm.utils.secrets import bootstrap_secrets
        bootstrap_secrets(
            ["LIGHTNING_USER_ID", "LIGHTNING_API_KEY", "GITHUB_PAT",
             "HF_TOKEN", "LIGHTNING_SSH_TARGET", "LIGHTNING_TEAMSPACE"],
            aliases={
                "LIGHTNING_API_KEY": (
                    "LIGHTNING_AI",
                    "LIGHTNING_TOKEN",
                    "LIGHTNING_AUTH_TOKEN",
                ),
                "LIGHTNING_USER_ID": (
                    "LIGHTNING_USERNAME",
                    "LIGHTNING_USER",
                ),
                "GITHUB_PAT": (
                    "GH_TOKEN",
                    "GITHUB_TOKEN",
                ),
            },
            verbose=False,
        )
    except Exception as exc:
        # Secrets resolution must never crash the deploy chain — if
        # the user has the env vars set the SDK will see them and
        # everything works regardless.
        print(f"[lightning] (note) secrets bootstrap skipped: "
              f"{type(exc).__name__}: {exc}", file=sys.stderr)


def _git_repo_url() -> Optional[str]:
    """Detect the origin remote URL of the local repo, or ``None``.

    Returns the HTTPS form when the local checkout uses git@ SSH,
    so the in-Studio clone (which has no SSH keys) can be tokenised
    with ``x-access-token:${GITHUB_PAT}``.
    """
    import subprocess
    try:
        r = subprocess.run(
            ["git", "remote", "get-url", "origin"],
            capture_output=True, text=True, check=True,
        )
    except (FileNotFoundError, subprocess.CalledProcessError):
        return None
    url = r.stdout.strip()
    if not url:
        return None
    # Normalise git@github.com:owner/repo.git → https://github.com/owner/repo.git
    m = re.match(r"git@([^:]+):(.+?)(?:\.git)?$", url)
    if m:
        return f"https://{m.group(1)}/{m.group(2)}.git"
    return url


def _git_current_branch() -> str:
    """Return the current local branch name, or ``"master"`` as fallback."""
    import subprocess
    try:
        r = subprocess.run(
            ["git", "rev-parse", "--abbrev-ref", "HEAD"],
            capture_output=True, text=True, check=True,
        )
        return r.stdout.strip() or "master"
    except (FileNotFoundError, subprocess.CalledProcessError):
        return "master"


def _short_id() -> str:
    """Generate a short, human-readable job id (lightning-YYYYMMDD-HHMMSS-XX).

    Includes a 2-char random suffix so two launches in the same
    second don't collide.
    """
    import secrets
    ts = time.strftime("%Y%m%d-%H%M%S")
    suffix = secrets.token_hex(2)
    return f"ln-{ts}-{suffix}"


def _resolve_teamspace_handle(user_handle, requested_name: str):
    """Return a Teamspace object for *requested_name*, or the user's
    only one when *requested_name* is empty.

    Raises :class:`RuntimeError` with a helpful chooser message when
    multiple teamspaces exist and none was named.
    """
    teamspaces = list(user_handle.teamspaces)
    if requested_name:
        for ts in teamspaces:
            if ts.name == requested_name:
                return ts
        names = ", ".join(t.name for t in teamspaces)
        raise RuntimeError(
            f"Teamspace {requested_name!r} not found for user "
            f"{user_handle.name!r}. Available: [{names}]"
        )
    if len(teamspaces) == 1:
        return teamspaces[0]
    if not teamspaces:
        raise RuntimeError(
            f"User {user_handle.name!r} has no teamspaces. "
            "Create one at https://lightning.ai first."
        )
    names = ", ".join(t.name for t in teamspaces)
    raise RuntimeError(
        f"User {user_handle.name!r} has multiple teamspaces ({names}). "
        "Pick one via `brian deploy --teamspace <name>` or "
        "[deploy].teamspace in brian.toml."
    )


def _get_authed_user_safe():
    """Return the authenticated User via the SDK's internal resolver.

    Wraps the private :func:`_get_authed_user` so we can handle the
    case where the API-key UUID is in ``LIGHTNING_USER_ID`` (which
    the public ``User(name=…)`` constructor rejects).
    """
    try:
        from lightning_sdk.utils.resolve import _get_authed_user
        return _get_authed_user()
    except Exception as exc:
        raise RuntimeError(
            "Could not resolve the authenticated Lightning user. "
            "Make sure LIGHTNING_USER_ID and LIGHTNING_API_KEY are set "
            "in your .env (or run `lightning login`). "
            f"Underlying error: {type(exc).__name__}: {exc}"
        ) from exc


class LightningConnector(BaseConnector):
    """Launch + monitor training on Lightning AI Studios."""

    @classmethod
    def platform_name(cls) -> str:
        return "lightning"

    # ── launch ──────────────────────────────────────────────────────

    def launch(self, config: DeployConfig) -> int:
        """SSH-style deploy: clone repo on Studio, set up env, run training.

        Auth options (highest-precedence first):
          1. ``LIGHTNING_SSH_TARGET=s_<id>@ssh.lightning.ai`` in .env /
             shell — pure-SSH mode: Studio discovery via SDK is skipped
             entirely.  Requires the SSH key at ``~/.lightning/lightning_rsa``
             (downloaded once by ``lightning login`` or ``brian deploy``
             with credentials, then reused forever).
          2. ``lightning login`` (writes ``~/.lightning/credentials.json``)
          3. ``LIGHTNING_USER_ID`` + ``LIGHTNING_API_KEY`` in .env / shell.

        The training process is *detached* — the SSH call returns as soon
        as the background ``nohup`` is in flight, so ``brian deploy`` exits
        while training continues on the Studio.
        Re-attach via ``brian ps --platform lightning``.
        """
        # ── 1. Resolve auth from .env / env ──
        _bootstrap_lightning_secrets()

        branch = (config.branch
                  or config.extra_env.get("BRANCH")
                  or _git_current_branch())
        repo_url = (config.extra_env.get("REPO_URL")
                    or _git_repo_url()
                    or "https://github.com/269652/BRIAN.git")

        # ── 2. Check for pure-SSH mode (LIGHTNING_SSH_TARGET shortcut) ──
        ssh_target_override = (
            os.environ.get("LIGHTNING_SSH_TARGET")
            or config.extra_env.get("LIGHTNING_SSH_TARGET")
            or ""
        )
        if ssh_target_override:
            key_path = self._try_existing_ssh_key()
            if key_path is None:
                # Attempt download with API key if available.
                try:
                    key_path = self._get_ssh_key()
                except RuntimeError:
                    pass
            if key_path is None:
                print(
                    "[lightning] LIGHTNING_SSH_TARGET is set but no SSH key "
                    "found at ~/.lightning/lightning_rsa.\n"
                    "  Download it once by setting LIGHTNING_API_KEY and "
                    "running `brian deploy` (it will be cached for future runs).",
                    file=sys.stderr,
                )
                return 1
            job_id = _short_id()
            log_path = f"{REMOTE_LOGS}/{job_id}.log"
            studio_name = (f"brian-{config.label}" if config.label
                           else "brian-train")
            machine_s = (
                config.extra_env.get("LIGHTNING_MACHINE")
                or config.scale
                or "A100"
            )
            print(f"[lightning] mode        : pure-SSH (no SDK auth)")
            print(f"[lightning] ssh_target  : {ssh_target_override}")
            print(f"[lightning] SSH key     : {key_path}")
            print(f"[lightning] studio      : {studio_name}")
            print(f"[lightning] machine     : {machine_s}")
            print(f"[lightning] repo        : {repo_url}")
            print(f"[lightning] branch      : {branch}")
            print(f"[lightning] arch        : {config.arch or '(default)'}")
            print(f"[lightning] steps       : {config.steps}")
            if config.resume_from:
                print(f"[lightning] resume_from : {config.resume_from}")
            print(f"[lightning] job_id      : {job_id}")
            print(f"[lightning] remote log  : {log_path}")
            return self._run_setup_and_train(
                config=config,
                key_path=key_path,
                ssh_target=ssh_target_override,
                repo_url=repo_url,
                branch=branch,
                job_id=job_id,
                log_path=log_path,
                studio_name=studio_name,
                teamspace="(ssh-target)",
                host="(ssh-target)",
                machine_s=machine_s,
                sdk_name="none",
            )

        # ── 3. Import SDK ──
        Studio, Machine, sdk_name = _import_lightning_sdk()
        if Studio is None:
            print(
                "[lightning] lightning-sdk is not installed.\n"
                "  pip install lightning-sdk\n"
                "  Docs: https://lightning.ai/docs/overview/studios",
                file=sys.stderr,
            )
            return 1

        # ── 4. Resolve user + teamspace via SDK ──
        try:
            user = _get_authed_user_safe()
        except RuntimeError as exc:
            print(
                f"[lightning] SDK authentication failed: {exc}\n"
                "  Options:\n"
                "  1. Run `lightning login` (writes ~/.lightning/credentials)\n"
                "  2. Set LIGHTNING_USER_ID + LIGHTNING_API_KEY in .env\n"
                "     (get them from https://lightning.ai → Profile → API Keys)\n"
                "  3. Set LIGHTNING_SSH_TARGET=s_<id>@ssh.lightning.ai in .env\n"
                "     (copy from Lightning web UI → Studio → SSH; requires SSH key\n"
                "      at ~/.lightning/lightning_rsa from a prior `lightning login`)",
                file=sys.stderr,
            )
            return 1

        requested_ts = (
            config.extra_env.get("LIGHTNING_TEAMSPACE")
            or os.environ.get("LIGHTNING_TEAMSPACE")
            or ""
        )
        try:
            teamspace = _resolve_teamspace_handle(user, requested_ts)
        except RuntimeError as exc:
            print(f"[lightning] {exc}", file=sys.stderr)
            return 1

        # ── 5. Assemble launch parameters ──
        job_id = _short_id()
        studio_name = (f"brian-{config.label}" if config.label
                       else "brian-train")
        machine = self._resolve_machine(config, Machine)
        log_path = f"{REMOTE_LOGS}/{job_id}.log"

        # Pre-flight summary
        print(f"[lightning] SDK         : {sdk_name}")
        print(f"[lightning] user        : {user.name}")
        print(f"[lightning] teamspace   : {teamspace.name}")
        print(f"[lightning] studio      : {studio_name}")
        print(f"[lightning] machine     : {machine}")
        print(f"[lightning] repo        : {repo_url}")
        print(f"[lightning] branch      : {branch}")
        print(f"[lightning] arch        : {config.arch or '(default)'}")
        print(f"[lightning] steps       : {config.steps}")
        if config.resume_from:
            print(f"[lightning] resume_from : {config.resume_from}")
        print(f"[lightning] job_id      : {job_id}")
        print(f"[lightning] remote log  : {log_path}")

        # ── 6. Provision the Studio ──
        try:
            studio = Studio(
                name=studio_name,
                teamspace=teamspace,
                user=user,
                create_ok=True,
            )
        except Exception as exc:
            print(f"[lightning] Studio() failed: {type(exc).__name__}: "
                  f"{exc}", file=sys.stderr)
            return 1

        print(f"[lightning] starting Studio on {machine} ...")
        try:
            studio.start(machine=machine)
        except Exception as exc:
            msg = str(exc).lower()
            if "already" in msg or "running" in msg:
                print(f"[lightning] (note) Studio already running: {exc}")
            else:
                print(f"[lightning] start() failed: "
                      f"{type(exc).__name__}: {exc}", file=sys.stderr)
                return 1

        # ── 7. Push env vars onto the Studio ──
        remote_env = self._build_remote_env(config, repo_url, branch, job_id)
        try:
            studio.set_env(remote_env, partial=True)
        except Exception as exc:
            print(f"[lightning] (note) set_env: {type(exc).__name__}: "
                  f"{exc}", file=sys.stderr)

        # ── 8. Get SSH key + target ──
        try:
            key_path = self._get_ssh_key()
            print(f"[lightning] SSH key : {key_path}")
        except RuntimeError as exc:
            print(f"[lightning] {exc}", file=sys.stderr)
            return 1
        ssh_target = f"s_{studio._studio.id}@ssh.lightning.ai"
        print(f"[lightning] SSH target: {ssh_target}")

        return self._run_setup_and_train(
            config=config,
            key_path=key_path,
            ssh_target=ssh_target,
            repo_url=repo_url,
            branch=branch,
            job_id=job_id,
            log_path=log_path,
            studio_name=studio_name,
            teamspace=teamspace.name,
            host=user.name,
            machine_s=str(machine),
            sdk_name=sdk_name,
            remote_env=remote_env,
        )

    def _run_setup_and_train(
        self, config: DeployConfig, key_path: str, ssh_target: str,
        repo_url: str, branch: str, job_id: str, log_path: str,
        studio_name: str, teamspace: str, host: str, machine_s: str,
        sdk_name: str, remote_env: Optional[dict] = None,
    ) -> int:
        """SSH into the Studio, run setup + detached training, register job."""
        if remote_env is None:
            remote_env = self._build_remote_env(config, repo_url, branch,
                                                job_id)

        env_prefix = "".join(
            f"export {k}={shlex.quote(str(v))}\n"
            for k, v in remote_env.items()
            if v
        )

        # Tokenise the clone URL here in Python (not via sed-in-shell).
        # sed quoting is fragile: an empty PAT produces 'x-access-token:@...'
        # which libcurl rejects as CURLE_URL_MALFORMAT ("Malformed input to a
        # URL function") and git silently hides the token in its error output,
        # making the failure look like a plain-URL rejection.
        github_pat = (remote_env.get("GITHUB_PAT") or "").strip()
        if github_pat and repo_url.startswith("https://"):
            from urllib.parse import quote as _urlquote
            _enc = _urlquote(github_pat, safe="")
            clone_url = repo_url.replace("https://", f"https://x-access-token:{_enc}@", 1)
        else:
            clone_url = repo_url

        # ── Setup (clone + install) ──
        setup_cmd = self._build_setup_command(clone_url, branch, log_path)
        print(f"[lightning] running setup via SSH (clone + install) ...")
        try:
            out, exit_code = self._ssh_run(
                key_path, ssh_target, env_prefix + setup_cmd, timeout=600
            )
        except Exception as exc:
            print(f"[lightning] setup SSH failed: "
                  f"{type(exc).__name__}: {exc}", file=sys.stderr)
            return 1
        print(out[-4000:] if len(out) > 4000 else out)
        if exit_code != 0:
            print(f"[lightning] setup exited {exit_code}", file=sys.stderr)
            return exit_code or 1
        print(f"[lightning] setup OK")

        # ── Launch detached training ──
        train_cmd = self._build_train_command(config, log_path)
        print(f"[lightning] launching detached training via SSH:")
        print(f"            {train_cmd}")
        try:
            out, rc = self._ssh_run(
                key_path, ssh_target, env_prefix + train_cmd, timeout=60
            )
        except Exception as exc:
            print(f"[lightning] launch SSH failed: "
                  f"{type(exc).__name__}: {exc}", file=sys.stderr)
            return 1
        print(out)
        if rc != 0:
            print(f"[lightning] launch exited {rc}", file=sys.stderr)
            return rc or 1

        # ── Persist job record ──
        info = JobInfo(
            job_id=job_id,
            platform=self.platform_name(),
            label=config.label or "(none)",
            status=JobStatus.RUNNING.value,
            machine=machine_s,
            branch=branch,
            arch=config.arch or "",
            steps=config.steps,
            studio_name=studio_name,
            teamspace=teamspace,
            host=host,
            log_path=log_path,
            source_dna=config.source_dna or "",
            extra={
                "sdk": sdk_name,
                "repo_url": repo_url,
                "resume_from": config.resume_from or "",
                "ssh_target": ssh_target,
                "ssh_key": key_path,
            },
        )
        path = register_job(info)
        print(f"[lightning] job registered: {path}")
        print(f"")
        print(f"  monitor : brian ps --platform lightning")
        print(f"  logs    : brian ps --logs {job_id} [--tail 200] [--it]")
        print(f"  stop    : brian stop {job_id}")
        print(f"")
        return 0

    # ── list_jobs / status / tail_logs / stop ───────────────────────

    def list_jobs(self) -> List[JobInfo]:
        """Refresh every persisted Lightning job's live status.

        Each on-disk record is re-attached via the SDK (cheap — no
        machine restart) so the status column reflects the current
        Studio state, not the moment-of-launch value.
        """
        records = load_jobs(platform=self.platform_name())
        if not records:
            return []
        # Bootstrap secrets once for the whole batch so SDK auth works.
        _bootstrap_lightning_secrets()
        Studio, _, _ = _import_lightning_sdk()
        if Studio is None:
            return records  # SDK missing — return stale records as-is

        try:
            user = _get_authed_user_safe()
        except RuntimeError:
            return records

        for r in records:
            try:
                teamspace = _resolve_teamspace_handle(user, r.teamspace)
                # SDK quirk: ``create_ok=False`` fails with "does not
                # exist" on Stopped/Completed Studios — only currently
                # Running ones are found. ``create_ok=True`` re-attaches
                # to ANY existing Studio (and only creates a new one if
                # the name truly isn't on the account). Safe here
                # because we already know the studio existed at launch.
                s = Studio(
                    name=r.studio_name,
                    teamspace=teamspace,
                    user=user,
                    create_ok=True,
                )
                native = getattr(s.status, "value", str(s.status))
                native_name = (native.name if hasattr(native, "name")
                               else str(native))
                r.status = _STATUS_MAP.get(
                    native_name, JobStatus.UNKNOWN
                ).value
            except Exception:
                # Studio deleted / unreachable — fall through with
                # whatever was on disk (likely "running" → stale).
                pass
        return records

    def status(self, job_id: str) -> JobStatus:
        info = load_job(job_id)
        if info is None or info.platform != self.platform_name():
            return JobStatus.UNKNOWN
        _bootstrap_lightning_secrets()
        Studio, _, _ = _import_lightning_sdk()
        if Studio is None:
            return JobStatus.UNKNOWN
        try:
            user = _get_authed_user_safe()
            teamspace = _resolve_teamspace_handle(user, info.teamspace)
            # create_ok=True so we re-attach to Stopped / Completed
            # Studios too (the SDK's create_ok=False only finds Running).
            s = Studio(name=info.studio_name, teamspace=teamspace,
                       user=user, create_ok=True)
            native = s.status
            native_name = (native.name if hasattr(native, "name")
                           else str(native))
            return _STATUS_MAP.get(native_name, JobStatus.UNKNOWN)
        except Exception:
            return JobStatus.UNKNOWN

    def tail_logs(self, job_id: str, n: int = 200) -> str:
        """Stream the last *n* lines of the remote training log."""
        info = load_job(job_id)
        if info is None or info.platform != self.platform_name():
            raise ValueError(
                f"No Lightning job with id {job_id!r} in registry"
            )
        log_path = info.log_path or f"{REMOTE_LOGS}/{job_id}.log"
        if log_path.startswith("~/"):
            quoted = '"$HOME/' + log_path[2:] + '"'
        else:
            quoted = shlex.quote(log_path)
        script = (f"tail -n {int(n)} {quoted} 2>/dev/null || "
                  f"echo '(log not yet created at {log_path})'")

        # ── Fast path: use stored SSH target + key from job record ──
        stored_target = info.extra.get("ssh_target", "") if info.extra else ""
        stored_key = info.extra.get("ssh_key", "") if info.extra else ""
        key_path = (stored_key if stored_key and Path(stored_key).exists()
                    else self._try_existing_ssh_key())
        if stored_target and key_path:
            out, _ = self._ssh_run(key_path, stored_target, script, timeout=30)
            return out or ""

        # ── SDK path: look up Studio to get SSH target ──
        _bootstrap_lightning_secrets()
        Studio, _, _ = _import_lightning_sdk()
        if Studio is None:
            raise RuntimeError("lightning-sdk is not installed")
        try:
            user = _get_authed_user_safe()
            teamspace = _resolve_teamspace_handle(user, info.teamspace)
            s = Studio(name=info.studio_name, teamspace=teamspace,
                       user=user, create_ok=True)
        except Exception as exc:
            raise RuntimeError(
                f"Could not attach to Studio {info.studio_name!r}: "
                f"{type(exc).__name__}: {exc}"
            ) from exc
        native = getattr(s, "status", None)
        native_name = (native.name if hasattr(native, "name") else str(native))
        if native_name in ("Stopped", "NotCreated", "Failed"):
            return (f"(studio {info.studio_name!r} is {native_name}; "
                    f"start it via the Lightning UI to inspect "
                    f"{info.log_path or 'the log'})")
        try:
            key_path = self._get_ssh_key()
            ssh_target = f"s_{s._studio.id}@ssh.lightning.ai"
            out, _ = self._ssh_run(key_path, ssh_target, script, timeout=30)
        except Exception:
            try:
                out = s.run(script)
            except Exception as exc2:
                raise RuntimeError(
                    f"tail_logs failed for {job_id}: "
                    f"{type(exc2).__name__}: {exc2}"
                ) from exc2
        return out or ""

    def stop(self, job_id: str) -> int:
        """Stop the Studio backing *job_id* and clean up the registry."""
        info = load_job(job_id)
        if info is None or info.platform != self.platform_name():
            print(f"[lightning] no job {job_id!r} in registry",
                  file=sys.stderr)
            return 1
        _bootstrap_lightning_secrets()
        Studio, _, _ = _import_lightning_sdk()
        if Studio is None:
            return 1
        try:
            user = _get_authed_user_safe()
            teamspace = _resolve_teamspace_handle(user, info.teamspace)
            # create_ok=True for re-attach across all studio states.
            s = Studio(name=info.studio_name, teamspace=teamspace,
                       user=user, create_ok=True)
            # If the Studio is already stopped, treat as a no-op
            # success — the user's intent ("don't run any more") is
            # already satisfied. Lightning auto-stops crashed runs.
            native = getattr(s, "status", None)
            native_name = (native.name if hasattr(native, "name")
                           else str(native))
            if native_name in ("Stopped", "NotCreated", "Failed",
                               "Completed"):
                print(f"[lightning] studio {info.studio_name!r} already "
                      f"{native_name} (no-op)")
            else:
                s.stop()
                print(f"[lightning] stopped studio {info.studio_name!r}")
        except Exception as exc:
            print(f"[lightning] stop() failed: "
                  f"{type(exc).__name__}: {exc}", file=sys.stderr)
            return 1
        # Mark as stopped + remove from active registry so brian ps
        # forgets about it. The actual SDK stop already happened above.
        info.status = JobStatus.STOPPED.value
        remove_job(job_id)
        print(f"[lightning] job {job_id} unregistered from .brian/jobs/")
        return 0

    # ── internal helpers ────────────────────────────────────────────

    @staticmethod
    def _try_existing_ssh_key() -> Optional[str]:
        """Return the SSH key path if it already exists locally, else None.

        Does NOT contact Lightning servers — safe to call with no credentials.
        """
        try:
            from lightning_sdk.utils.config import _DEFAULT_CONFIG_FILE_PATH
            import pathlib
            key = pathlib.Path(_DEFAULT_CONFIG_FILE_PATH).parent / "lightning_rsa"
            if key.exists():
                return str(key)
        except Exception:
            pass
        # Fallback: ~/.ssh/lightning_rsa
        import pathlib
        alt = pathlib.Path.home() / ".ssh" / "lightning_rsa"
        if alt.exists():
            return str(alt)
        return None

    @staticmethod
    def _get_ssh_key() -> str:
        """Return the Lightning SSH private key path, downloading if needed.

        Checks for an existing key first (``~/.lightning/lightning_rsa``)
        so repeated calls after the first credential setup are free.
        Falls back to :func:`configure_ssh_internal` which needs either
        ``lightning login`` local credentials or ``LIGHTNING_API_KEY``.
        """
        # Fast path: key already on disk → no API call needed.
        existing = LightningConnector._try_existing_ssh_key()
        if existing:
            return existing
        try:
            from lightning_sdk.cli.utils.ssh_connection import (
                configure_ssh_internal,
            )
            return configure_ssh_internal()
        except Exception as exc:
            raise RuntimeError(
                f"Could not obtain Lightning SSH key: "
                f"{type(exc).__name__}: {exc}\n"
                "Run `lightning login` or set LIGHTNING_API_KEY in .env,\n"
                "OR copy ~/.lightning/lightning_rsa from a machine that\n"
                "has already authenticated."
            ) from exc

    @staticmethod
    def _ssh_run(key_path: str, ssh_target: str, script: str,
                 timeout: int = 300) -> "Tuple[str, int]":
        """Execute *script* on a Lightning Studio via real SSH.

        Uses ``bash --login`` so the remote ``~/.bashrc`` / PATH are
        loaded — same environment the user gets when running
        ``lightning studio ssh``.  The *script* is piped to stdin so
        complex multi-line commands with arbitrary quoting work without
        shell-escaping gymnastics on the caller side.

        Returns ``(stdout+stderr, exit_code)``.
        """
        import subprocess as _sp
        proc = _sp.run(
            [
                "ssh",
                "-i", key_path,
                "-o", "StrictHostKeyChecking=no",
                "-o", "UserKnownHostsFile=/dev/null",
                "-o", "BatchMode=yes",
                "-o", "ServerAliveInterval=15",
                "-o", "ServerAliveCountMax=4",
                "-o", "ConnectTimeout=30",
                ssh_target,
                "bash", "--login",
            ],
            input=script,
            capture_output=True,
            text=True,
            timeout=timeout,
        )
        return proc.stdout + proc.stderr, proc.returncode

    @staticmethod
    def _resolve_machine(config: DeployConfig, Machine) -> "Machine":
        """Map ``config`` → :class:`Machine` enum value.

        Precedence: ``extra_env["LIGHTNING_MACHINE"]`` > ``config.scale``
        > ``Machine.T4`` (default — matches ``[hardware.T4]`` in
        ``brian.toml``).
        """
        override = config.extra_env.get("LIGHTNING_MACHINE", "").lower()
        scale = (override or config.scale or "").lower()

        if "a100" in scale:
            return Machine.A100
        if "a10g" in scale:
            return Machine.A10G
        if "l4" in scale:
            return Machine.L4
        if "t4" in scale:
            return Machine.T4
        # Fall back to T4 — the default hardware tier in brian.toml
        # ``[hardware.T4]`` and the cheapest GPU on Lightning AI.
        return Machine.T4

    @staticmethod
    def _build_remote_env(config: DeployConfig, repo_url: str,
                          branch: str, job_id: str) -> dict:
        """Env vars to push onto the Studio before training starts."""
        env: dict = {
            "BRIAN_JOB_ID": job_id,
            "BRIAN_REPO_URL": repo_url,
            "BRIAN_BRANCH": branch,
            "PYTHONIOENCODING": "utf-8",
        }
        # Forward secrets that the training loop needs for pushes.
        for k in ("HF_TOKEN", "GITHUB_PAT", "HF_REPO_ID"):
            v = os.environ.get(k)
            if v:
                env[k] = v
        # Forward caller-supplied extras (minus Lightning-internal keys
        # that don't belong on the remote side).
        skip = {"LIGHTNING_MACHINE", "LIGHTNING_TEAMSPACE",
                "LIGHTNING_API_KEY", "LIGHTNING_USER_ID"}
        for k, v in config.extra_env.items():
            if k in skip or not v:
                continue
            env[str(k)] = str(v)
        return env

    @staticmethod
    def _build_setup_command(repo_url: str, branch: str,
                             log_path: str) -> str:
        """Bash one-liner that clones/updates the repo and installs deps.

        Idempotent — re-running on a warm Studio just does a ``git
        fetch`` + ``checkout`` instead of a full clone.

        ``repo_url`` must already be tokenised by the caller (i.e.
        ``https://x-access-token:PAT@github.com/...``) when auth is
        needed. The caller is responsible for URL-encoding the PAT.
        """
        # CLONE_URL is passed pre-baked — no sed substitution in shell.
        clone_url_expr = f"CLONE_URL={shlex.quote(repo_url)}"
        return (
            f"set -e; "
            f"mkdir -p {REMOTE_BASE} {REMOTE_LOGS}; "
            f"cd {REMOTE_BASE}; "
            f"{clone_url_expr}; "
            # First-time clone OR fast-forward update.
            f"if [ -d {REMOTE_REPO}/.git ]; then "
            f"  echo '[setup] repo exists - fetching latest'; "
            f"  cd {REMOTE_REPO}; "
            f"  git remote set-url origin \"$CLONE_URL\"; "
            f"  git fetch --all --prune; "
            f"  git checkout {shlex.quote(branch)}; "
            f"  git reset --hard origin/{shlex.quote(branch)}; "
            f"else "
            f"  echo '[setup] cloning repo'; "
            f"  git clone --depth 50 --branch {shlex.quote(branch)} "
            f"      \"$CLONE_URL\" {REMOTE_REPO}; "
            f"  cd {REMOTE_REPO}; "
            f"fi; "
            f"echo '[setup] installing python deps'; "
            # Two-step install:
            #  1. ``pip install -e .[ml]`` gets the CLI + heavy training
            #     stack (torch, transformers, datasets, tiktoken, einops).
            #     Critical — base ``.`` deps are CLI-only by design so a
            #     fresh CLI install doesn't pull a 3 GB torch wheel.
            #  2. ``pip install -r requirements.txt`` is the belt-and-
            #     braces — picks up anything pyproject.toml might miss
            #     (e.g. transitive pins, optional accelerators).
            # Run both so a partial failure of one still produces a
            # working training environment.
            f"python -m pip install --upgrade pip --quiet; "
            f"python -m pip install -e '.[ml]' --quiet || "
            f"  echo '[setup] WARN: pip install -e .[ml] failed; falling back to requirements.txt'; "
            f"python -m pip install -r requirements.txt --quiet || "
            f"  echo '[setup] WARN: pip install -r requirements.txt failed'; "
            # Verify the critical training imports resolve before we
            # hand off to the (detached) train command — a missing
            # ``transformers`` here means training will crash inside
            # ``nohup`` and leave no usable trace except the log file.
            f"python -c 'import torch, transformers, tiktoken, einops; "
            f"  print(f\"[setup] verified torch={{torch.__version__}} "
            f"transformers={{transformers.__version__}}\")' || "
            f"  {{ echo '[setup] FATAL: required training deps missing'; exit 2; }}; "
            f"echo '[setup] done at '$(date)"
        )

    @staticmethod
    def _build_train_command(config: DeployConfig, log_path: str) -> str:
        """Bash one-liner that launches training in the background.

        ``nohup`` + ``&`` + ``disown`` ensures the process survives the
        SDK session disconnect. Output is redirected to *log_path* so
        ``brian ps --logs`` can ``tail -n`` it back.
        """
        arch = config.arch or "architectures/current"
        parts: list[str] = [
            "python -m neuroslm.train_dsl",
            f"--arch {shlex.quote(arch)}",
            f"--steps {int(config.steps)}",
        ]
        if config.log_every > 0:
            parts.append(f"--log_every {int(config.log_every)}")
        if config.save_every > 0:
            parts.append(f"--save_every {int(config.save_every)}")
        if config.push_every > 0:
            parts.append(f"--push_every {int(config.push_every)}")
        if config.push_backend:
            parts.append(f"--push_backend {shlex.quote(config.push_backend)}")
        if config.resume_from:
            parts.append(f"--resume_from {shlex.quote(config.resume_from)}")
        if config.ood_every > 0:
            parts.append(f"--ood_every {int(config.ood_every)}")

        train_inner = " ".join(parts)
        # Wrap in cd + nohup + disown so the process survives the SDK
        # detach. Append a "[train] done" marker so the log tail can
        # detect completion.
        return (
            f"cd {REMOTE_REPO} && "
            f"mkdir -p {REMOTE_LOGS} && "
            f"nohup bash -c '{train_inner}; "
            f"  echo \"[train] done at \"$(date)' "
            f"  > {log_path} 2>&1 &"
            f" disown; "
            f"echo \"[launch] pid=$! log={log_path}\""
        )
