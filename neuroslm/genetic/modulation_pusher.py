# -*- coding: utf-8 -*-
"""Auto-push discovered modulations during Colab/vast runs.

When the explorer finds a modulation and saves it to ``modulations/*.neuro``,
this commits and pushes *only that directory* so a long GPU run streams its
discoveries back to git as they happen — the same pattern as the training
log-pusher, scoped to the modulation store so it never sweeps up unrelated
working-tree changes. Best-effort: failures return a reason, never raise, so a
discovery run is never interrupted by a git hiccup.
"""
from __future__ import annotations

import subprocess
from pathlib import Path
from typing import Optional


def _git(args, cwd) -> subprocess.CompletedProcess:
    return subprocess.run(["git", *args], cwd=str(cwd), capture_output=True, text=True)


def _identity_args(root) -> list:
    """Fallback ``-c user.*`` args when the runtime has no git identity.

    A fresh Colab/vast runtime has no ``user.name``/``user.email`` configured, so
    ``git commit`` refuses ("Please tell me who you are"). Supply a bot identity
    just for the artifact commit so a discovery run streams its results anyway.
    Respects an existing identity when one is set.
    """
    email = _git(["config", "user.email"], root).stdout.strip()
    name = _git(["config", "user.name"], root).stdout.strip()
    if email and name:
        return []
    return ["-c", "user.name=brian-discovery",
            "-c", "user.email=brian-discovery@users.noreply.github.com"]


def push_artifacts(repo_root, paths, message: Optional[str] = None,
                   remote: str = "origin", branch: Optional[str] = None) -> dict:
    """git add → commit → push, scoped to ``paths`` (files or dirs). Status dict.

    Used to stream *all* discovery artifacts — the modulation store, the search
    ledger, run JSONs — back to git during a long Colab/vast run, scoped so it
    never sweeps up unrelated working-tree changes. Best-effort; never raises.
    """
    root = Path(repo_root)
    if not (root / ".git").exists() and _git(["rev-parse", "--git-dir"], root).returncode != 0:
        return {"pushed": False, "reason": "not a git repo"}

    paths = [str(p) for p in paths]
    # keep only paths that exist (relative to root)
    present = [p for p in paths if (root / p).exists()]
    if not present:
        return {"pushed": False, "reason": "no artifacts present"}

    _git(["add", "-f", *present], root)
    staged = _git(["diff", "--cached", "--quiet", "--", *present], root)
    if staged.returncode == 0:
        return {"pushed": False, "reason": "no changes"}

    msg = message or "artifacts: auto-push discovery artifacts"
    commit = _git([*_identity_args(root), "commit", "-m", msg, "--", *present], root)
    if commit.returncode != 0:
        return {"pushed": False, "reason": "commit failed",
                "detail": (commit.stderr or commit.stdout)[-300:]}

    if branch is None:
        branch = _git(["rev-parse", "--abbrev-ref", "HEAD"], root).stdout.strip() or "master"

    # push with fetch+rebase retry — a concurrent writer (another run's log/artifact
    # pusher, or a code push) may advance the remote branch between our commit and
    # push; rebase over it and retry instead of failing with "fetch first".
    for attempt in range(4):
        push = _git(["push", remote, branch], root)
        if push.returncode == 0:
            return {"pushed": True, "branch": branch, "paths": present,
                    "rebased": attempt > 0}
        err = (push.stderr or push.stdout or "")
        if "fetch first" not in err and "rejected" not in err and "non-fast-forward" not in err:
            return {"pushed": False, "reason": "push failed",
                    "detail": err[-300:], "branch": branch}
        # integrate remote changes then retry
        _git(["fetch", remote, branch], root)
        rebase = _git(["rebase", f"{remote}/{branch}"], root)
        if rebase.returncode != 0:
            _git(["rebase", "--abort"], root)
            return {"pushed": False, "reason": "rebase conflict",
                    "detail": (rebase.stderr or rebase.stdout)[-300:], "branch": branch}
    return {"pushed": False, "reason": "push failed after retries", "branch": branch}


def push_modulations(repo_root, message: Optional[str] = None,
                     remote: str = "origin", branch: Optional[str] = None,
                     subdir: str = "modulations") -> dict:
    """Thin wrapper: push just the modulation store (``modulations/``)."""
    return push_artifacts(repo_root, [subdir], message=message, remote=remote,
                          branch=branch)
