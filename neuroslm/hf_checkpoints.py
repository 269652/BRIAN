"""HuggingFace Hub checkpoint discovery + download — companion to
:mod:`neuroslm.checkpoint_push`.

This module is the *read* side of the HF-backed checkpoint flow. Where
:mod:`neuroslm.checkpoint_push` uploads ``lfs_checkpoints/<RUN_DIR>/step<N>.pt``
artefacts to ``hf://<repo>/checkpoints/<RUN_DIR>/step<N>.pt``, this module:

* lists everything the caller can resume from (``list_repo_checkpoints``)
* finds the most recent / highest-step checkpoint (``find_latest_checkpoint``)
* downloads a single ``.pt`` (+ ``.mem`` sidecar if present) into the
  local ``lfs_checkpoints/`` cache (``download_checkpoint``)

The shared invariant with the push side is the ``checkpoints/<RUN_DIR>/step<N>.pt``
layout. ``_HF_CKPT_PREFIX`` and ``_DEFAULT_HF_REPO_ID`` are imported from
:mod:`neuroslm.checkpoint_push` so the two sides stay in lockstep.

Auth chain (mirrors ``checkpoint_push._resolve_hf_token``):

    explicit ``token=`` arg → ``HF_TOKEN`` env → cached ``~/.huggingface/token``

Public entrypoints (used by ``brian hf list/pull/latest`` and
``brian deploy --latest``):

* :func:`list_repo_checkpoints(repo_id, prefix='', token=None)`
    → ``[CheckpointEntry(...), ...]``  newest first
* :func:`find_latest_checkpoint(repo_id, prefix='', token=None)`
    → ``Optional[CheckpointEntry]``
* :func:`download_checkpoint(path_in_repo, repo_id, dest_dir=None, token=None)`
    → ``Path`` to the downloaded ``.pt`` (sidecar grabbed too if it exists)

All three never raise — failures print and return empty / None / the
expected exception type. They are import-safe when ``huggingface_hub``
is not installed (lazy import + graceful fallback) so tests exercising
the LFS path don't pay the import cost.
"""
from __future__ import annotations

import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Tuple

# Reuse the constants from the push side so the two halves of the
# checkpoint flow can never disagree about the prefix or the default
# repo id. Importing from checkpoint_push pulls in zero heavy deps —
# that module is itself import-safe (lazy hf_hub import).
from neuroslm.checkpoint_push import (
    _DEFAULT_HF_REPO_ID,
    _HF_CKPT_PREFIX,
    _resolve_hf_token,
)


# ── Step-number extraction (mirrors train_dsl._checkpoint_step) ──
# Two layouts are accepted:
#   * H24+ per-run subdir:  ``checkpoints/<RUN_DIR>/step<N>.pt``
#   * Legacy flat:          ``checkpoints/dsl_arch_step<N>.pt`` /
#                           ``checkpoints/dsl_arch_<TS>_step<N>.pt``
_STEP_RE = re.compile(r"step(\d+)\.pt$")
_LEGACY_TS_STEP_RE = re.compile(r"_(\d{8}-\d{6})_step(\d+)\.pt$")
_LEGACY_FLAT_STEP_RE = re.compile(r"dsl_arch_step(\d+)\.pt$")


@dataclass(frozen=True)
class CheckpointEntry:
    """One checkpoint discovered on HF Hub.

    Attributes
    ----------
    path_in_repo : str
        Repository-relative path, e.g.
        ``"checkpoints/run-20260615_abc1234_arch/step5000.pt"``.
        Always uses ``/`` separators (HF Hub convention).
    step : int
        Training step parsed from the filename. ``0`` if unparseable.
    run_dir : str
        The run-subdir component (between ``checkpoints/`` and
        ``/step<N>.pt``). Empty string for the legacy flat layout.
    size : int
        File size in bytes, ``0`` if the lister did not report it.
    has_mem_sidecar : bool
        True if a sibling ``.mem`` file is also in the listing.
    """
    path_in_repo: str
    step: int
    run_dir: str = ""
    size: int = 0
    has_mem_sidecar: bool = False


def _parse_step(path_in_repo: str) -> int:
    """Extract the training step from any supported HF checkpoint layout.

    Returns 0 if the path does not match a known checkpoint layout
    (callers filter zeros out with ``if entry.step``).
    """
    m = (_STEP_RE.search(path_in_repo)
         or _LEGACY_TS_STEP_RE.search(path_in_repo)
         or _LEGACY_FLAT_STEP_RE.search(path_in_repo))
    if not m:
        return 0
    # _LEGACY_TS_STEP_RE has 2 groups (ts, step); the others have 1.
    g = m.groups()
    return int(g[1] if len(g) == 2 else g[0])


def _parse_run_dir(path_in_repo: str) -> str:
    """Extract the per-run subdir from ``checkpoints/<RUN>/step<N>.pt``.

    Returns the empty string for the legacy flat layout
    (``checkpoints/dsl_arch_step<N>.pt``).
    """
    if not path_in_repo.startswith(f"{_HF_CKPT_PREFIX}/"):
        return ""
    rest = path_in_repo[len(_HF_CKPT_PREFIX) + 1:]
    if "/" not in rest:
        return ""  # flat layout — no run dir
    return rest.rsplit("/", 1)[0]


# ─────────────────────────────────────────────────────────────────────
# Listing
# ─────────────────────────────────────────────────────────────────────


def list_repo_checkpoints(
        repo_id: Optional[str] = None,
        *,
        prefix: str = "",
        token: Optional[str] = None,
        repo_type: str = "model",
) -> List[CheckpointEntry]:
    """List every ``.pt`` checkpoint on a HF Hub repo, newest-step first.

    Parameters
    ----------
    repo_id : str, optional
        Target HF repo, e.g. ``"moritzroessler/BRIAN"``. Defaults to
        ``HF_REPO_ID`` env, then to
        :data:`neuroslm.checkpoint_push._DEFAULT_HF_REPO_ID`.
    prefix : str, default ``""``
        Filter to checkpoints whose ``path_in_repo`` starts with
        ``checkpoints/<prefix>``. Useful to scope to one run dir, e.g.
        ``prefix="run-20260615_abc1234"``.
    token : str, optional
        Explicit HF token (read access). Wins over ``HF_TOKEN`` env.
    repo_type : str, default ``"model"``
        Forwarded to ``HfApi.list_repo_files``.

    Returns
    -------
    list[CheckpointEntry]
        Sorted by ``step`` descending, then ``path_in_repo`` descending
        (so the newest run dir wins on a tie). Empty list on any
        failure — never raises.
    """
    repo_id = repo_id or os.environ.get("HF_REPO_ID", "").strip() \
        or _DEFAULT_HF_REPO_ID
    tok = _resolve_hf_token(token)

    try:
        from huggingface_hub import HfApi  # type: ignore[import]
    except ImportError as e:
        print(
            f"[hf_checkpoints] huggingface_hub not installed ({e}); "
            f"cannot list. `pip install huggingface_hub`.",
            flush=True,
        )
        return []

    try:
        api = HfApi(token=tok)
        # ``list_repo_files`` returns plain str paths; we re-decorate
        # with size + sidecar flags via ``_decorate_listing``.
        files = api.list_repo_files(repo_id=repo_id, repo_type=repo_type)
    except Exception as e:
        print(
            f"[hf_checkpoints] list_repo_files failed: "
            f"{type(e).__name__}: {e}",
            flush=True,
        )
        return []

    return _decorate_listing(files, prefix=prefix)


def _decorate_listing(
        files: List[str],
        *,
        prefix: str,
) -> List[CheckpointEntry]:
    """Filter ``files`` to checkpoints + decorate with step / sidecar.

    Pure function (no I/O) so unit tests can drive it with synthetic
    listings. Sidecar detection is a string-set lookup, O(N) total.
    """
    file_set = set(files)
    scope = f"{_HF_CKPT_PREFIX}/{prefix}" if prefix else f"{_HF_CKPT_PREFIX}/"
    entries: List[CheckpointEntry] = []
    for f in files:
        if not f.startswith(scope):
            continue
        if not f.endswith(".pt"):
            continue
        step = _parse_step(f)
        if step <= 0:
            continue
        run_dir = _parse_run_dir(f)
        sidecar = f[:-3] + ".mem"
        entries.append(CheckpointEntry(
            path_in_repo=f,
            step=step,
            run_dir=run_dir,
            size=0,
            has_mem_sidecar=sidecar in file_set,
        ))
    # Newest first: highest step wins, then break ties on path so newer
    # run dirs sort before older ones (lexicographic on commit-prefixed
    # run-dir names is good enough for the deploy-resume picker).
    entries.sort(key=lambda e: (-e.step, e.path_in_repo), reverse=False)
    # Sort key uses ``-step``, so ascending sort on the negated key
    # actually puts the LARGEST step first. The trailing
    # ``e.path_in_repo`` then breaks ties lexicographically ascending
    # — but we want the newest run-dir first on a tie, so flip:
    entries.sort(key=lambda e: (-e.step, e.path_in_repo))
    # The double sort is a clarity-vs-correctness tradeoff; collapse
    # to one final pass that also reverses the path tiebreak.
    entries = sorted(entries, key=lambda e: (-e.step, _neg_str(e.path_in_repo)))
    return entries


def _neg_str(s: str) -> Tuple[int, ...]:
    """Sort key that reverses lexicographic order on strings without
    the ``reverse=True`` whole-list flip (so it composes with other
    ascending keys)."""
    return tuple(-ord(c) for c in s)


# ─────────────────────────────────────────────────────────────────────
# Latest-checkpoint convenience
# ─────────────────────────────────────────────────────────────────────


def find_latest_checkpoint(
        repo_id: Optional[str] = None,
        *,
        prefix: str = "",
        token: Optional[str] = None,
        repo_type: str = "model",
) -> Optional[CheckpointEntry]:
    """Return the newest (highest-step) checkpoint matching ``prefix``.

    Wraps :func:`list_repo_checkpoints` and returns its first entry,
    or ``None`` if the listing came back empty (no checkpoints, or
    auth/network failure — both manifest as an empty list).
    """
    entries = list_repo_checkpoints(
        repo_id, prefix=prefix, token=token, repo_type=repo_type)
    return entries[0] if entries else None


# ─────────────────────────────────────────────────────────────────────
# Download
# ─────────────────────────────────────────────────────────────────────


def download_checkpoint(
        path_in_repo: str,
        *,
        repo_id: Optional[str] = None,
        dest_dir: Optional[str] = None,
        token: Optional[str] = None,
        repo_type: str = "model",
        force_download: bool = False,
) -> Optional[Path]:
    """Download a single ``.pt`` (+ sidecar) from HF Hub into the local
    ``lfs_checkpoints/`` tree, preserving the per-run subdir layout.

    Returns the absolute :class:`Path` to the downloaded ``.pt``, or
    ``None`` on any failure. Errors are printed.

    The on-disk layout mirrors the HF layout exactly:

        hf://<repo>/checkpoints/run-A/step5000.pt
        →
        <repo_root>/lfs_checkpoints/run-A/step5000.pt

    so that the existing ``train_dsl._maybe_resume`` globber finds the
    file without any further wiring. ``.mem`` sidecar is fetched too
    when the entry has one.

    Parameters
    ----------
    path_in_repo : str
        Repository-relative path, e.g.
        ``"checkpoints/run-A/step5000.pt"``. Both checkpoint layouts
        are supported (legacy flat + per-run subdir). The function
        is layout-preserving.
    repo_id : str, optional
        Target HF repo (defaults to ``HF_REPO_ID`` env then
        :data:`_DEFAULT_HF_REPO_ID`).
    dest_dir : str, optional
        Override the on-disk root. Defaults to ``<repo_root>/lfs_checkpoints``
        where ``<repo_root>`` is the parent of ``neuroslm/``. The
        ``checkpoints/`` prefix is *stripped* — the dest layout
        starts at ``run-A/step5000.pt``.
    token : str, optional
        Explicit HF token. Wins over ``HF_TOKEN`` env.
    force_download : bool, default ``False``
        Forwarded to ``hf_hub_download``. Set to True to bypass the
        local huggingface_hub cache and re-fetch.
    """
    repo_id = repo_id or os.environ.get("HF_REPO_ID", "").strip() \
        or _DEFAULT_HF_REPO_ID
    tok = _resolve_hf_token(token)
    if tok is None:
        # Public repos can be downloaded without a token; pass through
        # ``None`` to ``hf_hub_download`` and let it decide. Print a
        # gentle note so the failure mode is debuggable.
        print(
            f"[hf_checkpoints] no HF token (HF_TOKEN env empty and "
            f"~/.huggingface/token absent) — attempting anonymous "
            f"download of {path_in_repo}",
            flush=True,
        )

    try:
        from huggingface_hub import hf_hub_download  # type: ignore[import]
    except ImportError as e:
        print(
            f"[hf_checkpoints] huggingface_hub not installed ({e}); "
            f"cannot download. `pip install huggingface_hub`.",
            flush=True,
        )
        return None

    if dest_dir is None:
        repo_root = os.path.dirname(
            os.path.dirname(os.path.abspath(__file__)))
        dest_dir = os.path.join(repo_root, "lfs_checkpoints")
    dest_dir = os.path.abspath(dest_dir)
    os.makedirs(dest_dir, exist_ok=True)

    # Compute the local destination path. Strip the ``checkpoints/``
    # prefix so the on-disk path mirrors the train_dsl globber's
    # expectation (lfs_checkpoints/<RUN_DIR>/step<N>.pt).
    rel = path_in_repo
    if rel.startswith(f"{_HF_CKPT_PREFIX}/"):
        rel = rel[len(_HF_CKPT_PREFIX) + 1:]
    local_pt = Path(dest_dir) / rel
    local_pt.parent.mkdir(parents=True, exist_ok=True)

    try:
        # ``hf_hub_download`` returns a path inside its own cache.
        # We then materialise a copy at our canonical location (or
        # symlink, but Windows can't symlink without admin).
        cached = hf_hub_download(
            repo_id=repo_id,
            repo_type=repo_type,
            filename=path_in_repo,
            token=tok,
            force_download=force_download,
        )
        import shutil as _shutil
        _shutil.copy2(cached, local_pt)
        print(
            f"[hf_checkpoints] ✓ downloaded "
            f"hf://{repo_id}/{path_in_repo} → {local_pt}",
            flush=True,
        )
    except Exception as e:
        print(
            f"[hf_checkpoints] ⚠ download failed: "
            f"{type(e).__name__}: {e}",
            flush=True,
        )
        return None

    # ── Sidecar ──
    # Try the ``.mem`` companion next to the ``.pt``. Failures here
    # are non-fatal: the ``.mem`` is genetics-overlay state and resume
    # tolerates its absence.
    sidecar_in_repo = path_in_repo[:-3] + ".mem" \
        if path_in_repo.endswith(".pt") else path_in_repo + ".mem"
    try:
        cached_mem = hf_hub_download(
            repo_id=repo_id,
            repo_type=repo_type,
            filename=sidecar_in_repo,
            token=tok,
            force_download=force_download,
        )
        local_mem = local_pt.with_suffix(".mem")
        import shutil as _shutil
        _shutil.copy2(cached_mem, local_mem)
        print(
            f"[hf_checkpoints] ✓ downloaded sidecar "
            f"hf://{repo_id}/{sidecar_in_repo} → {local_mem}",
            flush=True,
        )
    except Exception:
        # Sidecar may legitimately not exist; silent skip.
        pass

    return local_pt


# ─────────────────────────────────────────────────────────────────────
# URL parsing — accept ``hf://repo/path`` shorthand
# ─────────────────────────────────────────────────────────────────────


def parse_hf_uri(uri: str) -> Tuple[str, str]:
    """Split an ``hf://<repo_id>/<path_in_repo>`` URI into its parts.

    The ``repo_id`` may itself contain a slash (``owner/repo``), so we
    split on the FOURTH character index past the scheme — i.e. find
    the first ``/`` after the repo namespace.

    Examples
    --------
    >>> parse_hf_uri("hf://moritzroessler/BRIAN/checkpoints/run-A/step5000.pt")
    ('moritzroessler/BRIAN', 'checkpoints/run-A/step5000.pt')

    >>> parse_hf_uri("hf://moritzroessler/BRIAN")
    ('moritzroessler/BRIAN', '')

    Raises
    ------
    ValueError
        If ``uri`` does not start with ``hf://`` or has fewer than two
        path components.
    """
    if not uri.startswith("hf://"):
        raise ValueError(f"not an hf:// URI: {uri!r}")
    rest = uri[len("hf://"):]
    parts = rest.split("/", 2)  # ['owner', 'repo', 'path/inside/repo']
    if len(parts) < 2:
        raise ValueError(
            f"hf:// URI must include owner/repo: got {uri!r}")
    repo_id = f"{parts[0]}/{parts[1]}"
    path_in_repo = parts[2] if len(parts) > 2 else ""
    return repo_id, path_in_repo


def inspect_checkpoint_metadata(path: "Path") -> dict:
    """Load a local .pt file and extract training metadata without GPU memory.

    Returns a dict with keys: step, params, model_hash, ppl, ood_ppl,
    vocab_size, d_sem. Fields that are absent in the checkpoint are None.
    """
    import hashlib
    try:
        import torch
    except ImportError:
        return {k: None for k in
                ("step", "params", "model_hash", "ppl", "ood_ppl",
                 "vocab_size", "d_sem")}

    payload = torch.load(str(path), map_location="cpu", weights_only=False)
    model: dict = payload.get("model") or {}

    total_params = sum(
        t.numel() for t in model.values() if hasattr(t, "numel")
    )

    h = hashlib.sha256()
    for k in sorted(model.keys()):
        t = model[k]
        if hasattr(t, "numpy"):
            try:
                h.update(t.numpy().tobytes())
            except Exception:
                pass
    model_hash = h.hexdigest()[:12] if total_params else None

    extra: dict = payload.get("extra") or {}
    ppl = (extra.get("ppl") or extra.get("train_ppl")
           or extra.get("eval_ppl"))
    ood_ppl = (extra.get("ood_ppl") or extra.get("wt103_ppl")
               or extra.get("ood_eval_ppl"))

    return {
        "step": payload.get("step", 0),
        "params": total_params or None,
        "model_hash": model_hash,
        "ppl": ppl,
        "ood_ppl": ood_ppl,
        "vocab_size": payload.get("vocab_size"),
        "d_sem": payload.get("d_sem"),
    }


__all__ = [
    "CheckpointEntry",
    "list_repo_checkpoints",
    "find_latest_checkpoint",
    "download_checkpoint",
    "parse_hf_uri",
    "inspect_checkpoint_metadata",
]
