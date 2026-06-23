"""README template renderer with claims, citations, and metric interpolation.

A small templating language for README.md that supports:

1. **Metric interpolation:** ${METRIC_NAME} from docs/readme_metrics.toml
2. **Log citations:** $cite(logfile.log, 20, 23) - inline log excerpt
3. **Claim definitions:** $claim { hypothesis: "H1", checkpoint: "...", ... }
4. **Claim references:** ${claim.H1.ood_ppl}, ${claim.H1.checkpoint}

Template syntax:
    
    # At the top of the file, define claims:
    $claim{
        id: "H22_smollm2",
        hypothesis: "H22",
        checkpoint: "hf://.../step10000.pt",
        train_ppl: 23.6,
        ood_ppl: 155.0,
        gap_ratio: 6.55,
        back: [
            ["logs/20260615/neuroslm-full-dna-arch/175931_0_10000.log", 800, 803]
        ],
        falsify: []
    }
    
    # Later in the document:
    Best H22 run achieved ${claim.H22_smollm2.ood_ppl} OOD PPL:
    
    $cite(logs/20260615/neuroslm-full-dna-arch/175931_0_10000.log, 800, 803)
    
    Or reference the claim's backing directly:
    ${claim.H22_smollm2.back[0]}

CLI: brian update-readme [--check]
Pre-commit: runs --check and exits 1 if README is stale
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional


# ── Exceptions ────────────────────────────────────────────────────────

class ReadmeRenderError(Exception):
    """Template rendering failed."""
    pass


class MissingMetricError(ReadmeRenderError):
    """One or more ${METRIC} placeholders have no value."""
    
    def __init__(self, missing: list[str]) -> None:
        self.missing = missing
        keys = ", ".join(sorted(missing))
        super().__init__(
            f"README template has {len(missing)} unresolved metric(s): {keys}\n"
            f"Add them to docs/readme_metrics.toml"
        )


class MissingClaimError(ReadmeRenderError):
    """Referenced claim doesn't exist."""
    
    def __init__(self, claim_id: str) -> None:
        super().__init__(f"Claim '{claim_id}' referenced but not defined")


class LogNotFoundError(ReadmeRenderError):
    """$cite() references a log file that doesn't exist."""
    
    def __init__(self, path: str) -> None:
        super().__init__(f"Log file not found: {path}")


# ── Data structures ───────────────────────────────────────────────────

@dataclass
class Claim:
    """A scientific claim with evidence."""
    id: str
    hypothesis: str
    checkpoint: Optional[str] = None
    train_ppl: Optional[float] = None
    ood_ppl: Optional[float] = None
    gap_ratio: Optional[float] = None
    back: list[list[Any]] = field(default_factory=list)  # Supporting evidence
    falsify: list[list[Any]] = field(default_factory=list)  # Contradicting evidence
    metadata: dict[str, Any] = field(default_factory=dict)  # Extra fields
    
    def __getitem__(self, key: str) -> Any:
        """Allow dict-like access for templating."""
        if hasattr(self, key):
            return getattr(self, key)
        return self.metadata.get(key)


# ── Regex patterns ────────────────────────────────────────────────────

# ${METRIC_NAME} - only UPPERCASE names
_METRIC_RE = re.compile(r"\$\{([A-Z][A-Z0-9_]*)\}")

# ${claim.ID.field} or ${claim.ID.field[index]}
_CLAIM_REF_RE = re.compile(
    r"\$\{claim\.([a-zA-Z0-9_]+)\.([a-zA-Z0-9_]+)(?:\[(\d+)\])?\}"
)

# $claim{ ... } - multiline claim definition
# $claim{ ... } - claim definition (captures trailing newlines)
_CLAIM_DEF_RE = re.compile(
    r"\$claim\s*\{([^}]+(?:\{[^}]*\}[^}]*)*)\}\n*",
    re.MULTILINE | re.DOTALL
)

# $cite(path, start, end) - log citation
_CITE_RE = re.compile(
    r"\$cite\(([^,]+),\s*(\d+),\s*(\d+)\)"
)

# ${TEASE:what:arg1[:arg2]} - live ledger/log teasers
# Examples:
#   ${TEASE:runs:5}         → markdown table of 5 most recent runs
#   ${TEASE:log:best:10}    → last 10 lines of best training log
#   ${TEASE:findings:3}     → 3 most recent findings entries
_TEASE_RE = re.compile(r"\$\{TEASE:([^}]+)\}")


# ── Parsing utilities ─────────────────────────────────────────────────

def _parse_claim_body(body: str) -> dict[str, Any]:
    """Parse claim body like Python dict literal.
    
    Supports:
        id: "H22"
        train_ppl: 23.6
        back: [["file.log", 0, 10]]
        falsify: []
    """
    # Simple parser - treat as quasi-JSON/Python literal
    # Clean up: remove trailing commas, convert to JSON-compatible
    body = body.strip()
    
    # Replace Python-style keys with JSON keys
    body = re.sub(r'([a-zA-Z_][a-zA-Z0-9_]*)\s*:', r'"\1":', body)
    
    # Try to parse as JSON
    try:
        return json.loads("{" + body + "}")
    except json.JSONDecodeError:
        # Fallback: manual key-value extraction
        result = {}
        # Match key: value patterns
        for match in re.finditer(r'"([^"]+)"\s*:\s*([^,\n]+)', body):
            key = match.group(1)
            value_str = match.group(2).strip()
            
            # Parse value
            if value_str.startswith('"') and value_str.endswith('"'):
                result[key] = value_str[1:-1]  # String
            elif value_str.startswith('['):
                # Array - find matching ]
                try:
                    result[key] = json.loads(value_str)
                except:
                    result[key] = value_str
            elif value_str in ('true', 'false', 'null'):
                result[key] = json.loads(value_str)
            else:
                # Try numeric
                try:
                    if '.' in value_str:
                        result[key] = float(value_str)
                    else:
                        result[key] = int(value_str)
                except ValueError:
                    result[key] = value_str
        
        return result


def _claim_from_dict(data: dict[str, Any]) -> Claim:
    """Convert parsed dict to Claim object."""
    claim_id = data.pop('id', None)
    if not claim_id:
        raise ReadmeRenderError("Claim missing 'id' field")
    
    # Extract known fields
    hypothesis = data.pop('hypothesis', '')
    checkpoint = data.pop('checkpoint', None)
    train_ppl = data.pop('train_ppl', None)
    ood_ppl = data.pop('ood_ppl', None)
    gap_ratio = data.pop('gap_ratio', None)
    back = data.pop('back', [])
    falsify = data.pop('falsify', [])
    
    # Everything else goes to metadata
    return Claim(
        id=claim_id,
        hypothesis=hypothesis,
        checkpoint=checkpoint,
        train_ppl=train_ppl,
        ood_ppl=ood_ppl,
        gap_ratio=gap_ratio,
        back=back,
        falsify=falsify,
        metadata=data
    )


def _read_log_lines(path: str, start: int, end: int, root: Path) -> str:
    """Read lines [start, end] from log file (1-indexed, inclusive)."""
    log_path = root / path
    if not log_path.exists():
        raise LogNotFoundError(path)
    
    try:
        with open(log_path, 'r', encoding='utf-8', errors='ignore') as f:
            lines = f.readlines()
    except Exception as e:
        raise ReadmeRenderError(f"Failed to read {path}: {e}")
    
    # Convert to 0-indexed
    start_idx = max(0, start - 1)
    end_idx = min(len(lines), end)
    
    excerpt = ''.join(lines[start_idx:end_idx])
    
    # Format as code block
    return f"```\n{excerpt.rstrip()}\n```"


# ── Main rendering ────────────────────────────────────────────────────

class TemplateRenderer:
    """Renders README template with metrics, claims, and citations."""
    
    def __init__(self, repo_root: Path, metrics: dict[str, str]):
        self.repo_root = repo_root
        self.metrics = metrics
        self.claims: dict[str, Claim] = {}
    
    def parse_claims(self, template: str) -> str:
        """Extract $claim{...} definitions and remove them from template."""
        
        def _extract_claim(match: re.Match) -> str:
            body = match.group(1)
            data = _parse_claim_body(body)
            claim = _claim_from_dict(data)
            self.claims[claim.id] = claim
            return ""  # Remove claim definition from template
        
        return _CLAIM_DEF_RE.sub(_extract_claim, template)
    
    def resolve_citations(self, template: str) -> str:
        """Replace $cite(path, start, end) with log excerpts."""
        
        def _cite(match: re.Match) -> str:
            path = match.group(1).strip()
            start = int(match.group(2))
            end = int(match.group(3))
            return _read_log_lines(path, start, end, self.repo_root)
        
        return _CITE_RE.sub(_cite, template)
    
    def resolve_claim_refs(self, template: str) -> str:
        """Replace ${claim.ID.field} with claim values."""
        
        def _ref(match: re.Match) -> str:
            claim_id = match.group(1)
            field = match.group(2)
            index = match.group(3)
            
            if claim_id not in self.claims:
                raise MissingClaimError(claim_id)
            
            claim = self.claims[claim_id]
            value = claim[field]
            
            # Handle array indexing
            if index is not None:
                if not isinstance(value, list):
                    raise ReadmeRenderError(
                        f"claim.{claim_id}.{field} is not a list"
                    )
                idx = int(index)
                if idx >= len(value):
                    raise ReadmeRenderError(
                        f"claim.{claim_id}.{field}[{idx}] out of range"
                    )
                value = value[idx]
            
            # Format value
            if value is None:
                return "—"
            elif isinstance(value, float):
                return f"{value:.1f}"
            elif isinstance(value, list):
                # If it's a log citation triple, format it
                if len(value) == 3 and isinstance(value[0], str):
                    return _read_log_lines(value[0], value[1], value[2], self.repo_root)
                return str(value)
            else:
                return str(value)
        
        return _CLAIM_REF_RE.sub(_ref, template)
    
    def resolve_metrics(self, template: str) -> str:
        """Replace ${METRIC} placeholders."""
        needed = set(_METRIC_RE.findall(template))
        missing = sorted(k for k in needed if k not in self.metrics)
        if missing:
            raise MissingMetricError(missing)
        
        def _sub(match: re.Match) -> str:
            return self.metrics[match.group(1)]
        
        return _METRIC_RE.sub(_sub, template)
    
    def resolve_tease_macros(self, template: str) -> str:
        """Replace ${TEASE:what:args...} with live ledger/log content."""
        from neuroslm.cli_help import (
            parse_runs_ledger, tease_runs, format_runs_table_md,
            tease_log_tail, tease_findings, get_best_log_path,
        )

        def _tease(match: re.Match) -> str:
            spec = match.group(1)
            parts = [p.strip() for p in spec.split(":")]
            what = parts[0] if parts else ""

            if what == "runs":
                n = int(parts[1]) if len(parts) > 1 else 5
                runs_path = self.repo_root / "docs" / "runs.md"
                if not runs_path.exists():
                    return "_Run ledger not found._"
                entries = parse_runs_ledger(runs_path.read_text(encoding="utf-8"))
                recent = tease_runs(entries, n=n)
                return format_runs_table_md(recent)

            if what == "log":
                # ${TEASE:log:best:10} or ${TEASE:log:path/to/file:10}
                src = parts[1] if len(parts) > 1 else "best"
                n = int(parts[2]) if len(parts) > 2 else 10
                if src == "best":
                    log_path = get_best_log_path(self.repo_root)
                else:
                    log_path = self.repo_root / src
                if log_path is None or not log_path.exists():
                    return "_Log not available._"
                tail = tease_log_tail(log_path, n=n)
                if not tail:
                    return "_Log empty._"
                rel = log_path.relative_to(self.repo_root) if log_path.is_relative_to(self.repo_root) else log_path
                return f"_Last {n} lines of `{rel}`:_\n```\n{tail}\n```"

            if what == "findings":
                n = int(parts[1]) if len(parts) > 1 else 3
                findings_path = self.repo_root / "docs" / "findings.md"
                if not findings_path.exists():
                    return "_findings.md not found._"
                text = findings_path.read_text(encoding="utf-8")
                return tease_findings(text, n=n)

            return match.group(0)  # unknown — leave as-is

        return _TEASE_RE.sub(_tease, template)

    def render(self, template: str) -> str:
        """Full rendering pipeline."""
        # 1. Parse and remove claim definitions
        template = self.parse_claims(template)

        # 2. Resolve citations $cite(...)
        template = self.resolve_citations(template)

        # 3. Resolve claim references ${claim.ID.field}
        template = self.resolve_claim_refs(template)

        # 4. Resolve ${TEASE:...} live-ledger macros
        template = self.resolve_tease_macros(template)

        # 5. Resolve metrics ${METRIC}
        template = self.resolve_metrics(template)

        return template


# ── Public API ────────────────────────────────────────────────────────

def load_metrics(metrics_path: Path) -> dict[str, str]:
    """Load metrics TOML and return flat {KEY: value} dict."""
    if not metrics_path.exists():
        raise FileNotFoundError(f"Metrics not found: {metrics_path}")

    try:
        import tomllib
    except ImportError:
        try:
            import tomli as tomllib  # type: ignore[no-redef]
        except ImportError:
            raise ImportError("Need tomllib (Python 3.11+) or tomli package")

    with open(metrics_path, "rb") as f:
        raw = tomllib.load(f)

    # Flatten all sections into one namespace
    # Top-level keys go directly, nested sections are also flattened
    flat: dict[str, str] = {}
    
    for key, value in raw.items():
        if isinstance(value, dict):
            # It's a [section] - flatten it
            for sub_key, sub_value in value.items():
                if isinstance(sub_value, bool):
                    flat[sub_key] = str(sub_value).lower()
                elif isinstance(sub_value, (int, float)):
                    flat[sub_key] = str(sub_value)
                else:
                    flat[sub_key] = str(sub_value)
        else:
            # It's a top-level key - add directly
            if isinstance(value, bool):
                flat[key] = str(value).lower()
            elif isinstance(value, (int, float)):
                flat[key] = str(value)
            else:
                flat[key] = str(value)

    return flat


def load_arch_exports(neuro_exports_path: Path) -> dict[str, str]:
    """Load .neuro/exports.toml (arch-derived values) if it exists.

    Returns an empty dict if the file is absent — arch exports are
    optional. Static metrics in docs/readme_metrics.toml always win
    if both define the same key (caller merges exports first, then metrics).
    """
    if not neuro_exports_path.exists():
        return {}
    return load_metrics(neuro_exports_path)


def render_readme(
    template_path: Path,
    metrics_path: Path,
    output_path: Optional[Path] = None,
    *,
    check: bool = False,
    repo_root: Optional[Path] = None,
    neuro_exports_path: Optional[Path] = None,
    skip_live_collection: bool = False,
) -> tuple[str, bool]:
    """Render README template with full v2 templating (compatible with v1 API).

    Parameters
    ----------
    template_path:
        Path to template file (README.template.md)
    metrics_path:
        Path to metrics TOML (docs/readme_metrics.toml)
    output_path:
        Destination for rendered README (README.md)
        Required unless check=True
    check:
        When True, compare rendered result to on-disk output_path
        without writing. Returns (rendered, is_clean) where
        is_clean is True iff the file already matches.
    repo_root:
        Repository root for resolving log paths.
        Defaults to template_path.parent
    neuro_exports_path:
        Optional path to .neuro/exports.toml generated from arch.neuro
        # @export directives.  When present its values are merged into
        the metrics dict *before* the static TOML, so static values win.
    skip_live_collection:
        Skip live metrics collection (pytest, arch parsing).
        Useful for tests that only care about template syntax.

    Returns
    -------
    (rendered_text, is_clean)
        is_clean is always True in write mode (non-check)

    Raises
    ------
    ReadmeRenderError
        If any placeholder has no value or other template error
    FileNotFoundError
        If template or metrics file does not exist
    """
    if repo_root is None:
        repo_root = template_path.parent

    # Layer 1: Live-collected metrics (source of truth)
    if not skip_live_collection:
        try:
            from neuroslm.metrics_collector import collect_all
            live_metrics = collect_all(repo_root, skip=["model_params"])
        except Exception:
            live_metrics = {}
    else:
        live_metrics = {}

    # Layer 2: Arch exports (.neuro/exports.toml)
    if neuro_exports_path is not None:
        arch_exports = load_arch_exports(neuro_exports_path)
    else:
        arch_exports = {}

    # Layer 3: Static TOML overrides (always wins)
    static_metrics = load_metrics(metrics_path)

    # Merge: live < arch_exports < static TOML
    merged = {**live_metrics, **arch_exports, **static_metrics}

    # Load and render
    template = template_path.read_text(encoding='utf-8')
    renderer = TemplateRenderer(repo_root, merged)
    rendered = renderer.render(template)

    # Resolve ${LOG_TAIL:src:N} and ${LOG_LINK:src} macros (defined in v1 renderer)
    from neuroslm.readme_renderer import resolve_log_macros
    rendered = resolve_log_macros(rendered, merged, repo_root)

    # Check mode: compare without writing
    if check:
        if output_path is None or not output_path.exists():
            return rendered, False
        current = output_path.read_text(encoding='utf-8')
        return rendered, current == rendered

    # Write mode
    if output_path is not None:
        output_path.write_text(rendered, encoding='utf-8')
    return rendered, True

