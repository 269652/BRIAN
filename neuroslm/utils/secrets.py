# -*- coding: utf-8 -*-
"""Cross-platform secrets resolver for notebooks and local scripts.

The first cell of every notebook (Colab, Kaggle, local Jupyter) keeps
re-implementing the same try/except chain to fetch ``GITHUB`` and
``HF_TOKEN``. When one of those branches silently fails — as it did on
the 2026-06-17 Colab run where ``capture_output=True`` swallowed every
``git push`` error — you don't find out until an hour of compute is
gone.

This module centralises that chain into a single ``get_secret(name)``
call that tries (in this exact order):

  1. ``os.environ`` — already-exported values short-circuit.
  2. **Colab**     — :mod:`google.colab.userdata` (notebook secrets).
  3. **Kaggle**    — :class:`kaggle_secrets.UserSecretsClient`.
  4. **.env file** — searched from CWD up to filesystem root.
  5. **Custom**    — providers registered via
                     :func:`register_secret_provider`. Use this to plug
                     in HashiCorp Vault, AWS Secrets Manager, 1Password,
                     etc. without touching this file.

Every backend is **lazily** imported and guarded — importing this
module never fails just because ``kaggle_secrets`` isn't installed.
Backends that raise (rate-limit, network blip, denied scope) are
demoted to "missing" without taking down the chain; pass
``verbose=True`` to see the diagnostics.

Typical use at the top of a notebook::

    from neuroslm.utils.secrets import bootstrap_secrets
    bootstrap_secrets(["GITHUB", "HF_TOKEN"])
    # → both values are now in os.environ; subprocesses inherit them.

Or, for a single value with custom aliases::

    from neuroslm.utils.secrets import get_secret
    tok = get_secret("GITHUB", aliases=("GITHUB_TOKEN", "GH_TOKEN"))

To register a new backend (e.g. an internal vault)::

    from neuroslm.utils.secrets import register_secret_provider
    register_secret_provider("vault", _my_vault_lookup, priority=25)
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Callable, Iterable, Optional

__all__ = [
    "get_secret",
    "bootstrap_secrets",
    "register_secret_provider",
    "unregister_secret_provider",
    "list_secret_providers",
    "detect_environment",
]

# ── Provider registry ──────────────────────────────────────────────
# Each entry: (priority, name, fn)
# Priority gates the order. Built-ins occupy 10/20/30/40; custom
# defaults to 50. Lower number = tried first.
_PROVIDERS: list[tuple[int, str, Callable[[str], Optional[str]]]] = []


def _provider_env(name: str) -> Optional[str]:
    """OS env vars — already-exported values short-circuit the chain."""
    val = os.environ.get(name)
    if val is None:
        return None
    val = val.strip()
    return val or None


def _provider_colab(name: str) -> Optional[str]:
    """Google Colab notebook secrets (``Tools → Secrets``)."""
    try:
        from google.colab import userdata  # type: ignore
    except Exception:
        return None
    try:
        val = userdata.get(name)
    except Exception:
        # SecretNotFoundError, NotebookAccessError, etc. — all → missing
        return None
    if not val:
        return None
    return str(val).strip() or None


def _provider_kaggle(name: str) -> Optional[str]:
    """Kaggle UserSecretsClient (``Add-ons → Secrets``)."""
    try:
        from kaggle_secrets import UserSecretsClient  # type: ignore
    except Exception:
        return None
    try:
        client = UserSecretsClient()
        val = client.get_secret(name)
    except Exception:
        return None
    if not val:
        return None
    return str(val).strip() or None


def _provider_dotenv(name: str) -> Optional[str]:
    """Walk CWD upward, parsing the first ``.env`` we find.

    Hand-rolled in ~15 lines so we don't pull in ``python-dotenv``.
    Supports ``KEY=value``, ``KEY="value"``, ``KEY='value'``, and
    ``export KEY=value``. Lines beginning with ``#`` are comments.
    """
    cwd = Path.cwd().resolve()
    # Walk up, including cwd itself
    for parent in (cwd, *cwd.parents):
        env_path = parent / ".env"
        if not env_path.is_file():
            continue
        try:
            with env_path.open(encoding="utf-8", errors="replace") as f:
                for raw in f:
                    line = raw.strip()
                    if not line or line.startswith("#"):
                        continue
                    if line.startswith("export "):
                        line = line[7:].lstrip()
                    if "=" not in line:
                        continue
                    key, _, val = line.partition("=")
                    if key.strip() != name:
                        continue
                    val = val.strip()
                    # Strip matched surrounding quotes
                    if len(val) >= 2 and val[0] == val[-1] and val[0] in "\"'":
                        val = val[1:-1]
                    val = val.strip()
                    return val or None
        except OSError:
            continue
        # Found the .env but the key wasn't in it → don't keep walking;
        # the *first* .env wins (POSIX dotenv convention).
        return None
    return None


# Built-in providers, lowest-priority-number-first
_PROVIDERS.extend([
    (10, "env",     _provider_env),
    (20, "colab",   _provider_colab),
    (30, "kaggle",  _provider_kaggle),
    (40, "dotenv",  _provider_dotenv),
])


# ── Public API ─────────────────────────────────────────────────────


def register_secret_provider(
    name: str,
    fn: Callable[[str], Optional[str]],
    *,
    priority: int = 50,
) -> None:
    """Register a custom secrets backend.

    Args:
        name: Short label used in :func:`list_secret_providers` and the
            ``verbose=True`` diagnostics.
        fn:   Callable ``fn(secret_name: str) -> Optional[str]``. Return
            ``None`` if the secret isn't available. Exceptions are
            caught and logged at ``verbose=True``; they never crash the
            chain.
        priority: Lower numbers are tried first. Built-ins occupy
            10 (env), 20 (Colab), 30 (Kaggle), 40 (.env). The default
            of 50 puts your provider after all of them — set to e.g.
            ``25`` if you want it before Kaggle.

    Re-registering the same ``name`` replaces the previous entry, which
    is useful in long-lived notebooks where a cell may be re-executed.
    """
    if not callable(fn):
        raise TypeError(f"fn must be callable, got {type(fn).__name__}")
    # Drop any existing provider with the same name (re-registration)
    _PROVIDERS[:] = [(p, n, f) for p, n, f in _PROVIDERS if n != name]
    _PROVIDERS.append((int(priority), name, fn))
    _PROVIDERS.sort(key=lambda t: t[0])


def unregister_secret_provider(name: str) -> bool:
    """Remove a provider by name. Returns ``True`` if anything was removed."""
    before = len(_PROVIDERS)
    _PROVIDERS[:] = [(p, n, f) for p, n, f in _PROVIDERS if n != name]
    return len(_PROVIDERS) != before


def list_secret_providers() -> list[tuple[int, str]]:
    """Return ``[(priority, name), …]`` sorted by priority.

    Handy in a notebook cell to confirm which backends are active
    before you launch a long training job.
    """
    return [(p, n) for p, n, _ in _PROVIDERS]


def get_secret(
    name: str,
    *,
    aliases: Iterable[str] = (),
    default: Optional[str] = None,
    cache_env: bool = True,
    verbose: bool = False,
) -> Optional[str]:
    """Resolve a secret from the first backend that has it.

    Args:
        name: Primary key to look up in every backend.
        aliases: Alternative names tried *within each backend* before
            moving to the next backend. Lets you handle the common
            ``GITHUB`` / ``GITHUB_TOKEN`` / ``GH_TOKEN`` triad without
            spamming three calls.
        default: Returned if no backend has the secret. ``None`` by
            default so callers can branch with ``if tok is None``.
        cache_env: If ``True`` (default), the resolved value is also
            written back to ``os.environ[name]`` so subprocesses spawned
            after this call inherit it. Set ``False`` for one-shot
            lookups you don't want to leak into the process environment.
        verbose: Print one line per backend miss to stderr. Useful when
            a notebook claims "no token" but you swear you set one.

    Returns:
        The secret string (stripped) or ``default``.
    """
    keys = (name, *(a for a in aliases if a and a != name))
    for priority, prov_name, fn in _PROVIDERS:
        for key in keys:
            try:
                val = fn(key)
            except Exception as exc:
                if verbose:
                    _vprint(f"provider={prov_name!r} key={key!r} "
                            f"raised {type(exc).__name__}: {exc}")
                continue
            if val:
                if cache_env:
                    os.environ[name] = val
                    # Also mirror under the alias that hit, so downstream
                    # code looking up GITHUB_TOKEN finds it after we
                    # resolved via GITHUB (or vice-versa).
                    if key != name:
                        os.environ.setdefault(key, val)
                if verbose:
                    _vprint(f"resolved {name!r} from provider={prov_name!r} "
                            f"(key={key!r})")
                return val
            if verbose:
                _vprint(f"provider={prov_name!r} key={key!r} → miss")
    if verbose:
        _vprint(f"{name!r} not found in any provider; returning default")
    return default


def bootstrap_secrets(
    names: Iterable[str],
    *,
    aliases: Optional[dict[str, Iterable[str]]] = None,
    verbose: bool = True,
    required: Iterable[str] = (),
) -> dict[str, Optional[str]]:
    """Resolve and cache multiple secrets in one call.

    This is the one-liner you put at the top of a notebook::

        bootstrap_secrets(["GITHUB", "HF_TOKEN"])

    Args:
        names: Iterable of primary secret names to resolve.
        aliases: Optional ``{name: [alias, …]}`` map. E.g.
            ``{"GITHUB": ["GITHUB_TOKEN", "GH_TOKEN"]}``.
        verbose: Print a one-line summary per secret (``set (12 chars)``
            or ``missing``). Set ``False`` for quiet operation.
        required: Names that MUST resolve; raises :class:`RuntimeError`
            listing the missing ones if any are absent.

    Returns:
        ``{name: value_or_None}`` for every name in ``names``.
    """
    aliases = aliases or {}
    results: dict[str, Optional[str]] = {}
    missing_required: list[str] = []
    for name in names:
        val = get_secret(
            name,
            aliases=tuple(aliases.get(name, ())),
            cache_env=True,
            verbose=False,
        )
        results[name] = val
        if verbose:
            if val:
                print(f"[secrets] {name}: set ({len(val)} chars)")
            else:
                print(f"[secrets] {name}: missing")
        if val is None and name in required:
            missing_required.append(name)
    if missing_required:
        env = detect_environment()
        raise RuntimeError(
            f"Required secret(s) not found: {missing_required}. "
            f"Detected environment: {env}. Configure them via "
            f"{'Tools → Secrets' if env == 'colab' else 'Add-ons → Secrets' if env == 'kaggle' else 'a .env file or os.environ'}."
        )
    return results


def detect_environment() -> str:
    """Best-effort guess at the host environment.

    Returns one of ``"colab"``, ``"kaggle"``, ``"jupyter"``, ``"ipython"``,
    ``"script"``. Used by :func:`bootstrap_secrets` to render a helpful
    error message — not for behavioural branching (the provider chain
    already handles that).
    """
    # Colab sets COLAB_GPU even on CPU runtimes
    if "COLAB_GPU" in os.environ or "COLAB_RELEASE_TAG" in os.environ:
        return "colab"
    if "KAGGLE_KERNEL_RUN_TYPE" in os.environ or "KAGGLE_URL_BASE" in os.environ:
        return "kaggle"
    try:
        from IPython import get_ipython  # type: ignore
        ipy = get_ipython()
        if ipy is not None:
            cls = type(ipy).__name__
            if cls == "ZMQInteractiveShell":
                return "jupyter"
            return "ipython"
    except Exception:
        pass
    return "script"


# ── internal ───────────────────────────────────────────────────────


def _vprint(msg: str) -> None:
    """Verbose log — stderr so it doesn't pollute notebook return values."""
    import sys
    print(f"[secrets] {msg}", file=sys.stderr, flush=True)
