"""Cross-platform hook runner — declarative ``hooks/<event>.yaml`` files
that map an event name (``pre-deploy``, ``post-deploy``, …) to a pair
of OS-specific scripts (``.sh`` for unix, ``.ps1`` for windows).

Why this exists
===============

The CLI has a few well-defined ceremony points where the user might
want to inject project-specific logic — recompile the master arch,
sync a heatmap, push a notification, etc. Hard-coding any of that
into ``cmd_deploy`` couples the binary release to one specific
workflow. A declarative hooks folder under the repo root, on the
other hand:

  * versions cleanly with the rest of the project
  * works on Windows AND Linux out of the box (every YAML names BOTH
    a ``.ps1`` and a ``.sh`` impl; the runner picks the right one)
  * fails fast and loud (``fail_on_error: true`` is the default)
  * is trivial to disable (``enabled: false``) without deleting it

YAML schema
===========

.. code-block:: yaml

    name: pre-deploy           # informational (filename is authoritative)
    description: One-liner shown in --help and log banners.
    enabled: true              # default true; false → silent skip
    fail_on_error: true        # default true; false → log + return 0
    timeout_seconds: 300       # default 300; 0 = no timeout
    scripts:
      windows: hooks/scripts/pre-deploy.ps1
      unix:    hooks/scripts/pre-deploy.sh

Repo layout
===========

::

    hooks/
      pre-deploy.yaml
      post-deploy.yaml
      scripts/
        pre-deploy.ps1
        pre-deploy.sh
        post-deploy.ps1
        post-deploy.sh

CLI integration
===============

:func:`neuroslm.cli.cmd_deploy` calls
:func:`run_hook("pre-deploy", REPO_ROOT)` before any vast.ai network
call. A non-zero return code from a ``fail_on_error: true`` hook
aborts the deploy with the hook's exit code propagating up — the user
never pays for vast.ai provisioning when the local pre-flight fails.

See ``tests/test_hooks.py`` for the locked behavioural contract.
"""

from __future__ import annotations

import os
import platform
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Optional


# Force UTF-8 stdout/stderr so YAML descriptions and hook output that
# contain non-cp1252 chars (e.g. arrows like → that we use liberally
# in banners) don't crash the Windows default console codepage. The
# `reconfigure` API exists on TextIOWrapper since 3.7; we guard the
# call because patched stdouts (pytest's capsys) may not expose it.
for _stream_name in ("stdout", "stderr"):
    _stream = getattr(sys, _stream_name, None)
    if _stream is not None and hasattr(_stream, "reconfigure"):
        try:
            _stream.reconfigure(encoding="utf-8", errors="replace")
        except (ValueError, OSError):
            pass


@dataclass(frozen=True)
class Hook:
    """In-memory representation of a parsed ``hooks/<name>.yaml`` file.

    All paths are stored as strings exactly as the YAML had them (they
    are resolved against the repo root at run time so the YAML can
    travel with the repo without absolute paths leaking in).
    """

    name: str
    description: str = ""
    enabled: bool = True
    fail_on_error: bool = True
    timeout_seconds: int = 300
    script_windows: str = ""
    script_unix: str = ""


# ── YAML loading ────────────────────────────────────────────────────


def _load_yaml(path: Path) -> dict:
    """Parse a YAML file into a dict, returning {} on empty file.

    Uses PyYAML if available, otherwise falls back to a tiny
    indentation-based parser sufficient for the small hook schema
    (string scalars, booleans, integers, one level of nesting under
    ``scripts:``). Keeping the parser self-contained means a stale
    venv can still run hooks during initial setup.
    """
    try:
        import yaml   # type: ignore
        with path.open("r", encoding="utf-8") as fh:
            data = yaml.safe_load(fh)
        return data or {}
    except ImportError:
        pass

    # Tiny fallback parser. The hook schema is intentionally trivial:
    #   key: scalar
    #   scripts:
    #     windows: path
    #     unix: path
    # — anything fancier should pull PyYAML into the venv.
    out: dict = {}
    nested_key: Optional[str] = None
    nested: dict = {}
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.split("#", 1)[0].rstrip()
        if not line.strip():
            continue
        if not line.startswith((" ", "\t")):
            if nested_key is not None:
                out[nested_key] = nested
                nested_key, nested = None, {}
            if ":" not in line:
                continue
            key, _, value = line.partition(":")
            key, value = key.strip(), value.strip()
            if not value:
                nested_key = key
                continue
            out[key] = _coerce_scalar(value)
        else:
            if nested_key is None:
                continue
            stripped = line.strip()
            if ":" not in stripped:
                continue
            key, _, value = stripped.partition(":")
            nested[key.strip()] = _coerce_scalar(value.strip())
    if nested_key is not None:
        out[nested_key] = nested
    return out


def _coerce_scalar(s: str):
    """Loose YAML-ish scalar coercion: true/false/int/strip-quotes."""
    if s.lower() in ("true", "yes", "on"):
        return True
    if s.lower() in ("false", "no", "off"):
        return False
    try:
        return int(s)
    except ValueError:
        pass
    if (s.startswith('"') and s.endswith('"')) or (
            s.startswith("'") and s.endswith("'")):
        return s[1:-1]
    return s


def load_hook(repo_root: Path, name: str) -> Optional[Hook]:
    """Return the parsed ``hooks/<name>.yaml`` or ``None`` if it is
    missing.

    ``None`` is the explicit "no-op" signal — the runner treats it as
    a green skip so users can selectively delete hook YAMLs without
    breaking the CLI.
    """
    repo_root = Path(repo_root)
    yaml_path = repo_root / "hooks" / f"{name}.yaml"
    if not yaml_path.is_file():
        return None
    data = _load_yaml(yaml_path)
    scripts = data.get("scripts") or {}
    return Hook(
        name=str(data.get("name") or name),
        description=str(data.get("description") or ""),
        enabled=bool(data.get("enabled", True)),
        fail_on_error=bool(data.get("fail_on_error", True)),
        timeout_seconds=int(data.get("timeout_seconds", 300)),
        script_windows=str(scripts.get("windows") or ""),
        script_unix=str(scripts.get("unix") or ""),
    )


# ── Runner ──────────────────────────────────────────────────────────


def _is_windows() -> bool:
    """Indirection so tests can flip the OS without touching ``platform``."""
    return platform.system().lower() == "windows"


def _run_subprocess(argv, *, cwd, env, timeout) -> int:
    """Thin wrapper around ``subprocess.run`` — kept as a module-level
    function so tests can patch it without monkey-patching subprocess
    globally.

    Output streams to the caller's stdout/stderr live (no PIPE) so
    long-running compile hooks stay observable in the terminal.
    """
    try:
        completed = subprocess.run(
            argv,
            cwd=str(cwd) if cwd else None,
            env=env,
            timeout=timeout if timeout > 0 else None,
            check=False,
        )
        return completed.returncode
    except subprocess.TimeoutExpired:
        print(f"[hook] TIMEOUT after {timeout}s: {argv!r}", file=sys.stderr)
        return 124   # standard timeout exit code


def _build_argv(hook: Hook, repo_root: Path) -> list[str]:
    """Pick the right script for the current OS and wrap it in the
    right interpreter."""
    if _is_windows():
        rel = hook.script_windows
        if not rel:
            raise RuntimeError(
                f"hook {hook.name!r}: scripts.windows not set "
                f"and we're on Windows")
        script = repo_root / rel
        # -ExecutionPolicy Bypass: don't fail on un-signed local scripts
        # -NoProfile:             skip user profile (faster, deterministic)
        # -File:                  invoke the .ps1 as a script
        return ["powershell.exe", "-NoProfile", "-ExecutionPolicy", "Bypass",
                "-File", str(script)]
    else:
        rel = hook.script_unix
        if not rel:
            raise RuntimeError(
                f"hook {hook.name!r}: scripts.unix not set "
                f"and we're on a unix")
        script = repo_root / rel
        return ["bash", str(script)]


def run_hook(name: str, repo_root: Optional[Path] = None,
             env: Optional[Mapping[str, str]] = None) -> int:
    """Run the named hook. Return 0 when the hook is missing, disabled,
    or successful; otherwise return the script's exit code (subject to
    ``fail_on_error``).

    Args:
      name:       Hook name. Looked up as ``hooks/<name>.yaml`` under
                  ``repo_root``.
      repo_root:  Repo root directory. Defaults to the current working
                  directory.
      env:        Extra environment variables for the script. Merged
                  on top of ``os.environ``.

    Return:
      An exit code suitable for ``sys.exit``: 0 on success / skip,
      non-zero only when ``fail_on_error: true`` AND the script
      returned non-zero.
    """
    repo_root = Path(repo_root) if repo_root is not None else Path.cwd()
    hook = load_hook(repo_root, name)
    if hook is None:
        return 0
    if not hook.enabled:
        print(f"[hook] {name}: disabled, skipping", flush=True)
        return 0

    print(f"[hook] {name}: running ({hook.description or 'no description'})",
          flush=True)
    try:
        argv = _build_argv(hook, repo_root)
    except RuntimeError as e:
        print(f"[hook] {name}: configuration error: {e}", file=sys.stderr)
        return 1 if hook.fail_on_error else 0

    proc_env = os.environ.copy()
    if env:
        proc_env.update({k: str(v) for k, v in env.items()})
    proc_env["BRIAN_HOOK_NAME"] = name
    proc_env["BRIAN_REPO_ROOT"] = str(repo_root)

    rc = _run_subprocess(argv, cwd=repo_root, env=proc_env,
                         timeout=hook.timeout_seconds)
    if rc != 0:
        if hook.fail_on_error:
            print(f"[hook] {name}: FAILED (exit {rc}) — propagating",
                  file=sys.stderr, flush=True)
            return rc
        print(f"[hook] {name}: failed (exit {rc}) — fail_on_error=false, "
              f"continuing", file=sys.stderr, flush=True)
        return 0
    print(f"[hook] {name}: ok", flush=True)
    return 0
