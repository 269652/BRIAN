"""README template renderer.

Reads ``docs/readme_metrics.toml`` (single source of truth for all
empirical numbers) and substitutes ``${UPPER_CASE}`` placeholders in
``docs/README.template.md`` to produce ``README.md``.

Only ``${UPPER_CASE}`` patterns are treated as placeholders — lowercase
or mixed-case dollar expressions (e.g. ``$1.50/hr`` in prose, ``$VAR``
in code examples) are left intact.

CLI entry point: ``brian update-readme [--check]``
Pre-commit hook: runs ``--check`` mode and exits 1 if README is stale.
"""
from __future__ import annotations

import re
from pathlib import Path
from typing import Optional


# ── log macro regexes ─────────────────────────────────────────────────
# ${LOG_TAIL:source:N} — last-N-lines fenced block + GitHub link
# ${LOG_LINK:source}   — GitHub link only (suitable for table cells)
# source: "best" | "latest" | <TOML-key> | <literal-path>

_LOG_TAIL_RE = re.compile(r"\$\{LOG_TAIL:([^:}]+):(\d+)\}")
_LOG_LINK_RE = re.compile(r"\$\{LOG_LINK:([^}]+)\}")


# ── public exception ──────────────────────────────────────────────────

class ReadmeRenderError(Exception):
    """One or more template placeholders have no value in metrics TOML."""

    def __init__(self, missing: list[str]) -> None:
        self.missing = missing
        keys = ", ".join(sorted(missing))
        super().__init__(
            f"README template has {len(missing)} unresolved placeholder(s): {keys}\n"
            f"Add the missing key(s) to docs/readme_metrics.toml and re-run."
        )


# ── placeholder regex — ONLY ${UPPER_CASE} ───────────────────────────
# Matches ${KEY} where KEY is one or more uppercase letters, digits, or
# underscores, starting with an uppercase letter.
_PLACEHOLDER_RE = re.compile(r"\$\{([A-Z][A-Z0-9_]*)\}")


# ── public API ────────────────────────────────────────────────────────

def extract_placeholders(template: str) -> set[str]:
    """Return the set of ``${KEY}`` placeholder names found in *template*."""
    return set(_PLACEHOLDER_RE.findall(template))


def load_metrics(metrics_path: Path) -> dict[str, str]:
    """Load *metrics_path* (TOML) and return a flat ``{KEY: str}`` dict.

    TOML sections are ignored — all keys from all sections are merged
    into one flat namespace.  Numeric values are converted to strings.
    """
    if not metrics_path.exists():
        raise FileNotFoundError(f"readme_metrics.toml not found: {metrics_path}")

    try:
        import tomllib  # Python 3.11+
    except ImportError:
        try:
            import tomli as tomllib  # type: ignore[no-redef]
        except ImportError:
            # Last resort: tomllib is in the stdlib from 3.11; older envs
            # need tomli.  If neither is available use the pure-Python
            # fallback below — only needed for very old envs.
            tomllib = None  # type: ignore[assignment]

    if tomllib is not None:
        with open(metrics_path, "rb") as fh:
            raw = tomllib.load(fh)
    else:
        raw = _parse_toml_fallback(metrics_path)

    flat: dict[str, str] = {}
    for value in raw.values():
        if isinstance(value, dict):
            for k, v in value.items():
                flat[k] = _to_str(v)
        else:
            # top-level scalar (shouldn't happen with our TOML layout, but
            # handle gracefully)
            pass
    return flat


def render(template: str, metrics: dict[str, str]) -> str:
    """Substitute all ``${KEY}`` placeholders in *template*.

    Raises :class:`ReadmeRenderError` listing every missing key if any
    placeholder has no value in *metrics*.  Reports ALL missing keys at
    once so the user can fix them in one pass.
    """
    needed = extract_placeholders(template)
    missing = sorted(k for k in needed if k not in metrics)
    if missing:
        raise ReadmeRenderError(missing)

    def _sub(m: re.Match) -> str:
        return metrics[m.group(1)]

    return _PLACEHOLDER_RE.sub(_sub, template)


def resolve_log_macros(
    template: str,
    metrics: dict[str, str],
    repo_root: Optional[Path] = None,
) -> str:
    """Expand ${LOG_TAIL:src:N} and ${LOG_LINK:src} macros in *template*.

    Must be called AFTER the standard ${KEY} substitution so that log file
    content (which may contain $-signs) is never treated as a placeholder.

    Sources
    -------
    ``best``        — resolved from ``.brian/best_run.ln``
    ``latest``      — mtime-newest ``.log`` file under ``<repo_root>/logs``
    ``<TOML-KEY>``  — path read from *metrics* dict
    ``<literal>``   — treated as a path relative to *repo_root*
    """
    root = repo_root or Path(".")

    def _tail_sub(m: re.Match) -> str:
        source, n = m.group(1), int(m.group(2))
        return _render_log_tail(_resolve_log_source(source, metrics, root), n, root)

    def _link_sub(m: re.Match) -> str:
        return _render_log_link(
            _resolve_log_source(m.group(1), metrics, root), root
        )

    result = _LOG_TAIL_RE.sub(_tail_sub, template)
    result = _LOG_LINK_RE.sub(_link_sub, result)
    return result


def render_readme(
    template_path: Path,
    metrics_path: Path,
    output_path: Optional[Path] = None,
    *,
    check: bool = False,
    repo_root: Optional[Path] = None,
) -> tuple[str, bool]:
    """Render the README template and optionally write / diff the result.

    Parameters
    ----------
    template_path:
        Path to the template file (``docs/README.template.md``).
    metrics_path:
        Path to the metrics TOML (``docs/readme_metrics.toml``).
    output_path:
        Destination for the rendered README (``README.md``).
        Required unless *check* is True and you only need the rendered string.
    check:
        When True, compare the rendered result to the on-disk *output_path*
        without writing anything.  Returns ``(rendered, is_clean)`` where
        ``is_clean`` is True iff the on-disk file already matches.

    Returns
    -------
    (rendered_text, is_clean)
        ``is_clean`` is always True in write mode (non-check).

    Raises
    ------
    ReadmeRenderError
        If any placeholder in the template has no value in metrics.
    FileNotFoundError
        If template or metrics file does not exist.
    """
    template = template_path.read_text(encoding="utf-8")
    metrics = load_metrics(metrics_path)
    # Step 1: standard ${KEY} substitution — leaves ${LOG_TAIL:…} untouched
    rendered = render(template, metrics)
    # Step 2: log macro resolution — safe to run on rendered content
    rendered = resolve_log_macros(rendered, metrics, repo_root=repo_root)

    if check:
        if output_path is None or not output_path.exists():
            return rendered, False
        current = output_path.read_text(encoding="utf-8")
        return rendered, current == rendered

    if output_path is not None:
        output_path.write_text(rendered, encoding="utf-8")
    return rendered, True


# ── log macro helpers ─────────────────────────────────────────────────

def _resolve_log_source(
    source: str,
    metrics: dict[str, str],
    root: Path,
) -> Optional[Path]:
    """Translate a macro source token into a Path (or None if not resolvable)."""
    if source == "best":
        ln = root / ".brian" / "best_run.ln"
        if not ln.is_file():
            return None
        try:
            from neuroslm.log_refs import read_ref
            ref = read_ref(ln)
            p = ref.target if ref.target.is_absolute() else root / ref.target
            return p if p.is_file() else None
        except Exception:
            return None

    if source == "latest":
        return _find_latest_log(root / "logs")

    # TOML key lookup
    if source in metrics:
        p = Path(metrics[source])
        resolved = p if p.is_absolute() else root / p
        return resolved  # may not exist — callers handle that gracefully

    # Literal path fallback
    p = Path(source)
    return p if p.is_absolute() else root / p


def _find_latest_log(log_dir: Path) -> Optional[Path]:
    """Return the most-recently-modified .log file under *log_dir*, or None."""
    try:
        logs = list(log_dir.rglob("*.log"))
    except OSError:
        return None
    if not logs:
        return None
    return max(logs, key=lambda p: p.stat().st_mtime)


def _render_log_tail(log_path: Optional[Path], n_lines: int, root: Path) -> str:
    """Render a GitHub link + last-N-lines fenced code block, or a "not available" note."""
    if log_path is None or not log_path.is_file():
        return "*(log not available)*"
    try:
        rel_posix = log_path.relative_to(root).as_posix()
    except ValueError:
        rel_posix = log_path.as_posix()
    lines = log_path.read_text(encoding="utf-8", errors="replace").splitlines()
    tail = lines[-n_lines:] if len(lines) > n_lines else lines
    link = f"[`{rel_posix}`]({rel_posix})"
    return f"{link}\n\n```\n" + "\n".join(tail) + "\n```"


def _render_log_link(log_path: Optional[Path], root: Path) -> str:
    """Render a GitHub-followable markdown link, or a "not available" note."""
    if log_path is None or not log_path.is_file():
        return "*(log not available)*"
    try:
        rel_posix = log_path.relative_to(root).as_posix()
    except ValueError:
        rel_posix = log_path.as_posix()
    return f"[`{rel_posix}`]({rel_posix})"


# ── internal helpers ──────────────────────────────────────────────────

def _to_str(value: object) -> str:
    if isinstance(value, float):
        # str(66.0) → "66.0", str(6.12) → "6.12" — preserves the decimal
        return str(value)
    return str(value)


def _parse_toml_fallback(path: Path) -> dict:
    """Minimal TOML parser for the subset we use (no arrays, no inline tables).

    Only supports:
    - Section headers: ``[section]``
    - String values:   ``KEY = "value"``
    - Integer values:  ``KEY = 1234``
    - Float values:    ``KEY = 3.14``
    - Comments:        ``# ...``
    """
    result: dict = {}
    current_section: dict = {}
    current_name = "__root__"
    result[current_name] = current_section

    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("[") and line.endswith("]"):
            current_name = line[1:-1].strip()
            current_section = {}
            result[current_name] = current_section
            continue
        if "=" in line:
            key, _, val = line.partition("=")
            key = key.strip()
            val = val.strip()
            if val.startswith('"') and val.endswith('"'):
                current_section[key] = val[1:-1]
            elif "." in val:
                try:
                    current_section[key] = float(val)
                except ValueError:
                    current_section[key] = val
            else:
                try:
                    current_section[key] = int(val)
                except ValueError:
                    current_section[key] = val
    return result
