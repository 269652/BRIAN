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


def render_readme(
    template_path: Path,
    metrics_path: Path,
    output_path: Optional[Path] = None,
    *,
    check: bool = False,
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
    rendered = render(template, metrics)

    if check:
        if output_path is None or not output_path.exists():
            return rendered, False
        current = output_path.read_text(encoding="utf-8")
        return rendered, current == rendered

    if output_path is not None:
        output_path.write_text(rendered, encoding="utf-8")
    return rendered, True


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
