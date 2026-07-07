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


def push_modulations(repo_root, message: Optional[str] = None,
                     remote: str = "origin", branch: Optional[str] = None,
                     subdir: str = "modulations") -> dict:
    """git add → commit → push, scoped to ``subdir``. Returns a status dict."""
    root = Path(repo_root)
    if not (root / ".git").exists() and not _git(["rev-parse", "--git-dir"], root).returncode == 0:
        return {"pushed": False, "reason": "not a git repo"}

    _git(["add", "-f", subdir], root)
    staged = _git(["diff", "--cached", "--quiet", "--", subdir], root)
    if staged.returncode == 0:
        return {"pushed": False, "reason": "no changes"}

    msg = message or "modulations: auto-push discovered modulations"
    commit = _git(["commit", "-m", msg, "--", subdir], root)
    if commit.returncode != 0:
        return {"pushed": False, "reason": "commit failed",
                "detail": (commit.stderr or commit.stdout)[-300:]}

    if branch is None:
        branch = _git(["rev-parse", "--abbrev-ref", "HEAD"], root).stdout.strip() or "master"
    push = _git(["push", remote, branch], root)
    if push.returncode != 0:
        return {"pushed": False, "reason": "push failed",
                "detail": (push.stderr or push.stdout)[-300:], "branch": branch}
    return {"pushed": True, "branch": branch}
