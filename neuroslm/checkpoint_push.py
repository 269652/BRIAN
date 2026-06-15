# -*- coding: utf-8 -*-
"""Standalone checkpoint → Git LFS push helper.

Extracted from ``neuroslm.train.push_checkpoint_to_lfs`` so the DSL
trainer (``neuroslm.train_dsl``) can call it without dragging the
legacy Brain trainer's import surface (and its torch / circuit
dependencies) into the import graph. Both trainers re-export from
here.

Contract pinned by ``tests/test_checkpoint_push_cadence.py``:

  1. Module is import-safe with **no** torch / model dependency
     (pure subprocess + shutil).
  2. ``push_checkpoint_to_lfs(path, repo_root=None)`` issues
     ``git add`` → ``git commit`` → ``git push`` in that order, with
     up-to-5 ``git pull --rebase`` retries on concurrent-push reject.
  3. Failures NEVER raise — print and return, so a crashed push
     during training never aborts the run.

Why "extract" instead of "import-from-train": ``neuroslm.train``
imports the full Brain at module top, which pulls torch + every
circuit module. The DSL training loop is supposed to work with a
DSL-defined harness that may not be a ``Brain``. Putting the push
helper in its own module breaks the import-time coupling.
"""
from __future__ import annotations

import os
import re
import shutil
import subprocess
from typing import Optional


def push_checkpoint_to_lfs(
        ckpt_path: str,
        repo_root: Optional[str] = None,
) -> None:
    """Copy checkpoint + optional ``.mem`` sidecar to
    ``lfs_checkpoints/`` (no-op if already there) and push via Git LFS.

    Auth: relies on ``~/.git-credentials`` (Colab cell 2 writes one),
    falls back to injecting the ``GITHUB`` env-var token into the
    remote URL.

    Parameters
    ----------
    ckpt_path : str
        Path to the ``.pt`` file produced by ``harness.save_checkpoint``.
        Already-inside-``lfs_checkpoints`` paths are fine (the copy
        is short-circuited when src == dst).
    repo_root : str, optional
        Repository root to run ``git`` commands in. Defaults to the
        parent of this file's parent (so ``neuroslm/checkpoint_push.py``
        → repo root). Used by tests to point at a tmp_path.

    Returns
    -------
    None
        All errors are swallowed and printed. Per the legacy
        ``train.py`` contract, "a crashed `git push` during training
        should never abort the run".
    """
    try:
        if repo_root is None:
            # Walk up two levels: neuroslm/checkpoint_push.py → repo root.
            repo_root = os.path.dirname(
                os.path.dirname(os.path.abspath(__file__)))
        lfs_dir = os.path.join(repo_root, "lfs_checkpoints")
        os.makedirs(lfs_dir, exist_ok=True)

        basename = os.path.basename(ckpt_path)

        # ── Copy .pt to lfs_checkpoints/ (or per-run subdir) ──
        # H24+ layout: the ckpt is ALREADY inside
        # ``lfs_checkpoints/<RUN_DIR>/step<N>.pt`` and the copy is
        # a no-op. Legacy flat layout: ``lfs_checkpoints/<file>.pt``,
        # also a no-op. The copy only kicks in when callers pass a
        # path outside ``lfs_checkpoints/`` (e.g. a /tmp save).
        dest = os.path.join(lfs_dir, basename)
        src_abs = os.path.abspath(ckpt_path)
        dst_abs = os.path.abspath(dest)
        if src_abs != dst_abs and not src_abs.startswith(
                os.path.abspath(lfs_dir) + os.sep):
            shutil.copy2(ckpt_path, dest)
            track_path = dest
        else:
            # Already inside lfs_checkpoints — track the original path
            # so per-run subdir layout (H24+) is preserved.
            track_path = ckpt_path

        # ── Copy .mem sidecar if present ──
        mem_src = ckpt_path.replace('.pt', '.mem')
        if os.path.exists(mem_src):
            mem_dst = os.path.join(
                os.path.dirname(track_path),
                os.path.basename(mem_src),
            )
            mem_src_abs = os.path.abspath(mem_src)
            mem_dst_abs = os.path.abspath(mem_dst)
            if mem_src_abs != mem_dst_abs:
                shutil.copy2(mem_src, mem_dst)

        # ── Ensure git identity ──
        subprocess.run(
            ["git", "config", "user.email", "train@neuroslm"],
            cwd=repo_root, capture_output=True,
        )
        subprocess.run(
            ["git", "config", "user.name", "NeuroSLM Train"],
            cwd=repo_root, capture_output=True,
        )

        # ── Inject GITHUB token into remote if no credentials file ──
        creds_file = os.path.expanduser("~/.git-credentials")
        token = (
            os.environ.get('GITHUB') or os.environ.get('GITHUB_TOKEN', '')
        ).strip()
        if token and not os.path.exists(creds_file):
            result = subprocess.run(
                ["git", "remote", "get-url", "origin"],
                cwd=repo_root, capture_output=True, text=True,
            )
            url = re.sub(
                r'https://[^@]+@', 'https://', result.stdout.strip(),
            )
            subprocess.run(
                ["git", "remote", "set-url", "origin",
                 url.replace('https://', f'https://{token}@', 1)],
                cwd=repo_root, capture_output=True,
            )

        # ── git add (10-min timeout for large LFS objects) ──
        r_add = subprocess.run(
            ["git", "add", "-f", "lfs_checkpoints/"],
            cwd=repo_root, capture_output=True, timeout=600, text=True,
        )
        if r_add.returncode != 0:
            print(f"[ckpt_push] ⚠ git add failed: {r_add.stderr[:200]}",
                  flush=True)
            return

        # ── git commit (fail loudly if add staged nothing) ──
        r_commit = subprocess.run(
            ["git", "commit", "-m", f"chkpt: {basename}"],
            cwd=repo_root, capture_output=True, text=True, timeout=60,
        )
        if r_commit.returncode != 0:
            stdout_low = (r_commit.stdout or "").lower()
            if ("nothing to commit" in stdout_low
                    or "no changes added" in stdout_low):
                print(
                    f"[ckpt_push] ⚠ nothing to commit for {basename} — "
                    f"file may already be tracked, or git add timed out "
                    f"silently",
                    flush=True,
                )
            else:
                print(
                    f"[ckpt_push] ⚠ git commit failed: "
                    f"{(r_commit.stderr or r_commit.stdout)[:200]}",
                    flush=True,
                )
            return

        # ── git push (with up-to-5 rebase retries on concurrent push) ──
        _branch = subprocess.run(
            ["git", "rev-parse", "--abbrev-ref", "HEAD"],
            cwd=repo_root, capture_output=True, text=True, timeout=30,
        ).stdout.strip() or "HEAD"

        pushed = False
        for attempt in range(5):
            r_push = subprocess.run(
                ["git", "push", "origin", "HEAD"],
                cwd=repo_root, capture_output=True, text=True, timeout=600,
            )
            if r_push.returncode == 0:
                print(f"[ckpt_push] ✓ pushed {basename} to Git LFS",
                      flush=True)
                pushed = True
                break
            print(
                f"[ckpt_push] push attempt {attempt+1} rejected; "
                f"rebasing on origin/{_branch} and retrying ...",
                flush=True,
            )
            subprocess.run(
                ["git", "pull", "--rebase", "origin", _branch],
                cwd=repo_root, capture_output=True, text=True, timeout=300,
            )
        if not pushed:
            print(
                f"[ckpt_push] ⚠ git push failed after retries: "
                f"{r_push.stderr.strip()[:300]}",
                flush=True,
            )
    except subprocess.TimeoutExpired as e:
        print(
            f"[ckpt_push] ⚠ LFS push timed out at: "
            f"{e.cmd[:3] if isinstance(e.cmd, list) else e.cmd}... "
            f"(timeout={e.timeout}s)",
            flush=True,
        )
    except Exception as e:
        print(f"[ckpt_push] ⚠ LFS push failed: {e}", flush=True)


__all__ = ["push_checkpoint_to_lfs"]
