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
    commit = _git(["commit", "-m", msg, "--", *present], root)
    if commit.returncode != 0:
        return {"pushed": False, "reason": "commit failed",
                "detail": (commit.stderr or commit.stdout)[-300:]}

    if branch is None:
        branch = _git(["rev-parse", "--abbrev-ref", "HEAD"], root).stdout.strip() or "master"
    push = _git(["push", remote, branch], root)
    if push.returncode != 0:
        return {"pushed": False, "reason": "push failed",
                "detail": (push.stderr or push.stdout)[-300:], "branch": branch}
    return {"pushed": True, "branch": branch, "paths": present}


def push_modulations(repo_root, message: Optional[str] = None,
                     remote: str = "origin", branch: Optional[str] = None,
                     subdir: str = "modulations") -> dict:
    """Thin wrapper: push just the modulation store (``modulations/``)."""
    return push_artifacts(repo_root, [subdir], message=message, remote=remote,
                          branch=branch)
