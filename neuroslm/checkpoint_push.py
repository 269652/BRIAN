# -*- coding: utf-8 -*-
"""Standalone checkpoint push helpers — Git LFS + HuggingFace Hub.

Extracted from ``neuroslm.train.push_checkpoint_to_lfs`` so the DSL
trainer (``neuroslm.train_dsl``) can call it without dragging the
legacy Brain trainer's import surface (and its torch / circuit
dependencies) into the import graph. Both trainers re-export from
here.

Two backends + a dispatcher
───────────────────────────

* ``push_checkpoint_to_hf(path, repo_id=..., token=None)`` —
  the new (2026-06-15) default. Uses ``huggingface_hub.upload_file``
  which is a single sync HTTPS PUT, ~10–50 MB/s on consumer links,
  no git rebase race, no shared-repo bandwidth quota. Push the
  ``.pt`` and its ``.mem`` sidecar individually. Auth chain:
  explicit ``token`` arg → ``HF_TOKEN`` env → cached token from
  ``~/.huggingface/token`` (via ``HfFolder.get_token``).

* ``push_checkpoint_to_lfs(path, repo_root=None)`` — the legacy
  Git LFS path. Issues ``git add`` → ``git commit`` → ``git push``
  with up-to-5 ``git pull --rebase`` retries on concurrent-push
  reject. Sound, but synchronously waits on a multi-hundred-MB
  upload AND races every other process pushing to ``origin/master``
  (chiefly the background ``log_pusher.sh``). Run 41063959 hung at
  exactly step 500 because of that race.

* ``push_checkpoint(path, backend="hf"|"lfs"|"none", **kw)`` — the
  dispatcher. Default is ``"hf"``; ``CHECKPOINT_PUSH_BACKEND`` env
  flips it. ``train_dsl`` calls THIS, not either backend directly,
  so the choice is per-deploy without re-editing the trainer.

Contracts pinned by:
  * ``tests/test_checkpoint_push_cadence.py`` — the legacy LFS path.
  * ``tests/test_checkpoint_push_hf.py`` — the HF path + dispatcher.

All push paths share the "never raise" contract: a crashed push
during training prints + returns; it must never abort the run.
"""
from __future__ import annotations

import os
import re
import shutil
import subprocess
from typing import Optional

# Default HF repo when neither ``repo_id`` arg nor ``HF_REPO_ID`` env
# is set. Matches the workspace's canonical checkpoint repo as
# specified by the user.
_DEFAULT_HF_REPO_ID = "moritzroessler/BRIAN"

# Prefix under which checkpoints are stored on the HF repo. Mirrors
# the on-disk ``lfs_checkpoints/<RUN_DIR>/step<N>.pt`` layout, just
# with a friendlier root name on the Hub. Picked so the repo can
# also hold non-checkpoint artefacts (datasets, tokeniser dumps,
# etc.) without collision.
_HF_CKPT_PREFIX = "checkpoints"


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


# ─────────────────────────────────────────────────────────────────────
# HuggingFace Hub push (2026-06-15 — new default)
# ─────────────────────────────────────────────────────────────────────


def _hf_path_in_repo(ckpt_path: str, repo_root: Optional[str]) -> str:
    """Map an on-disk ``lfs_checkpoints/<RUN_DIR>/step<N>.pt`` (or
    ``.mem``) to its HF Hub ``path_in_repo`` ``checkpoints/<RUN_DIR>/step<N>.pt``.

    The per-run subdir is preserved so concurrent runs never collide
    on the HF repo. If the checkpoint sits OUTSIDE ``lfs_checkpoints/``
    (e.g. an ad-hoc ``/tmp/foo/step.pt``) we fall back to a flat
    ``checkpoints/<basename>`` layout — the caller will still get a
    unique-ish path because vast.ai run dirs are commit-prefixed.
    """
    ckpt_abs = os.path.abspath(ckpt_path)
    if repo_root is None:
        repo_root = os.path.dirname(
            os.path.dirname(os.path.abspath(__file__)))
    lfs_root = os.path.abspath(os.path.join(repo_root, "lfs_checkpoints"))
    if ckpt_abs.startswith(lfs_root + os.sep):
        # Strip the lfs_checkpoints/ prefix → keep the per-run subdir
        rel = os.path.relpath(ckpt_abs, lfs_root)
        # Always emit POSIX separators (HF Hub uses '/')
        rel_posix = rel.replace(os.sep, "/")
        return f"{_HF_CKPT_PREFIX}/{rel_posix}"
    # Outside lfs_checkpoints/: flat layout
    return f"{_HF_CKPT_PREFIX}/{os.path.basename(ckpt_path)}"


def _resolve_hf_token(explicit: Optional[str]) -> Optional[str]:
    """Pick an HF token using the canonical chain.

    Returns ``None`` only when *all* sources are empty. The caller
    treats ``None`` as "skip push — there is no auth to use".
    """
    if explicit:
        return explicit
    env_tok = os.environ.get("HF_TOKEN", "").strip()
    if env_tok:
        return env_tok
    # Fall back to the cached token from ``~/.huggingface/token``.
    # We ask huggingface_hub itself (it implements the cache lookup
    # consistently with how upload_file resolves auth when ``token``
    # is None) but we import it lazily so this module stays
    # import-safe in environments without huggingface_hub installed.
    try:
        from huggingface_hub import HfFolder  # type: ignore[import]
        cached = HfFolder.get_token()
        return cached or None
    except Exception:
        return None


def push_checkpoint_to_hf(
        ckpt_path: str,
        repo_id: Optional[str] = None,
        *,
        token: Optional[str] = None,
        repo_root: Optional[str] = None,
        repo_type: str = "model",
) -> None:
    """Upload a checkpoint (+ ``.mem`` sidecar if present) to a HF Hub
    model repo.

    This is the post-LFS-hang default backend. Run 41063959 (2026-06-15)
    confirmed that synchronous ``git push`` of a 569 MB LFS object
    inside the training loop races the background log-pusher and
    blocks step 501+ for at least 600 s (often forever, once GitHub
    LFS bandwidth caps kick in). HF Hub avoids the race entirely.

    Parameters
    ----------
    ckpt_path : str
        Local path to the ``.pt`` file produced by
        ``harness.save_checkpoint``. Looks for a ``.mem`` sidecar
        next to it and uploads that too when present.
    repo_id : str, optional
        Target HF repo, e.g. ``"moritzroessler/BRIAN"``. Defaults to
        ``HF_REPO_ID`` env, then to
        :data:`_DEFAULT_HF_REPO_ID`.
    token : str, optional
        Explicit HF write token. Wins over ``HF_TOKEN`` env which
        wins over the cached ``~/.huggingface/token``.
    repo_root : str, optional
        Workspace root — used to compute the per-run subdir in the
        target ``path_in_repo``. Defaults to the parent of this
        module's parent.
    repo_type : str
        Forwarded to ``upload_file``; default ``"model"`` matches a
        HF Hub model repo.

    Returns
    -------
    None
        Errors are printed and swallowed — same "never raise"
        contract as :func:`push_checkpoint_to_lfs`.
    """
    try:
        if repo_id is None:
            repo_id = os.environ.get("HF_REPO_ID", "").strip() \
                or _DEFAULT_HF_REPO_ID

        tok = _resolve_hf_token(token)
        if tok is None:
            print(
                f"[ckpt_push] no HF token (HF_TOKEN env empty and "
                f"~/.huggingface/token absent) — skipping HF push "
                f"of {os.path.basename(ckpt_path)}",
                flush=True,
            )
            return

        # Lazy-import: keeps this module import-safe when
        # huggingface_hub is not installed (e.g. CI envs that only
        # exercise the LFS backend).
        try:
            from huggingface_hub import upload_file  # type: ignore[import]
        except ImportError as e:
            print(
                f"[ckpt_push] ⚠ huggingface_hub not installed "
                f"({e}); skipping HF push. Add `huggingface_hub` to "
                f"requirements.txt to enable.",
                flush=True,
            )
            return

        # ── Upload .pt ──
        pt_path_in_repo = _hf_path_in_repo(ckpt_path, repo_root)
        upload_file(
            path_or_fileobj=ckpt_path,
            path_in_repo=pt_path_in_repo,
            repo_id=repo_id,
            repo_type=repo_type,
            token=tok,
            commit_message=f"chkpt: {os.path.basename(ckpt_path)}",
        )
        print(
            f"[ckpt_push] ✓ pushed {os.path.basename(ckpt_path)} → "
            f"hf://{repo_id}/{pt_path_in_repo}",
            flush=True,
        )

        # ── Upload .mem sidecar if present ──
        mem_path = ckpt_path[:-3] + ".mem" if ckpt_path.endswith(".pt") \
            else ckpt_path + ".mem"
        if os.path.exists(mem_path):
            mem_path_in_repo = _hf_path_in_repo(mem_path, repo_root)
            upload_file(
                path_or_fileobj=mem_path,
                path_in_repo=mem_path_in_repo,
                repo_id=repo_id,
                repo_type=repo_type,
                token=tok,
                commit_message=f"chkpt: {os.path.basename(mem_path)}",
            )
            print(
                f"[ckpt_push] ✓ pushed sidecar "
                f"{os.path.basename(mem_path)} → "
                f"hf://{repo_id}/{mem_path_in_repo}",
                flush=True,
            )
    except Exception as e:
        print(
            f"[ckpt_push] ⚠ HF push failed: {type(e).__name__}: {e}",
            flush=True,
        )


# ─────────────────────────────────────────────────────────────────────
# Dispatcher
# ─────────────────────────────────────────────────────────────────────


def push_checkpoint(
        ckpt_path: str,
        *,
        backend: Optional[str] = None,
        repo_root: Optional[str] = None,
        # HF-specific kwargs (ignored by LFS path)
        repo_id: Optional[str] = None,
        token: Optional[str] = None,
) -> None:
    """Push a checkpoint via the configured backend.

    Resolution order for ``backend``:
      1. Explicit ``backend=...`` kwarg          ← wins
      2. ``CHECKPOINT_PUSH_BACKEND`` env var
      3. ``"hf"`` (post-LFS-hang default)

    Recognised values:

    * ``"hf"``   — :func:`push_checkpoint_to_hf` (recommended)
    * ``"lfs"``  — :func:`push_checkpoint_to_lfs` (legacy)
    * ``"none"`` — no-op (useful for local-dev runs that don't want
      any remote push at all)

    Unknown values fall through to the HF path with a warning.

    All errors inside the chosen backend are swallowed — see the
    per-backend "never raise" contracts.
    """
    if backend is None:
        backend = os.environ.get("CHECKPOINT_PUSH_BACKEND", "").strip() \
            or "hf"
    backend = backend.lower()

    if backend == "none":
        return  # explicitly disabled
    if backend == "lfs":
        push_checkpoint_to_lfs(ckpt_path, repo_root=repo_root)
        return
    if backend == "hf":
        push_checkpoint_to_hf(
            ckpt_path, repo_id=repo_id,
            token=token, repo_root=repo_root,
        )
        return

    print(
        f"[ckpt_push] ⚠ unknown push backend {backend!r}; "
        f"falling back to 'hf'",
        flush=True,
    )
    push_checkpoint_to_hf(
        ckpt_path, repo_id=repo_id, token=token, repo_root=repo_root,
    )


__all__ = [
    "push_checkpoint",
    "push_checkpoint_to_hf",
    "push_checkpoint_to_lfs",
]
