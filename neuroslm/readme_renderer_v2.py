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
    
    def render(self, template: str) -> str:
        """Full rendering pipeline."""
        # 1. Parse and remove claim definitions
        template = self.parse_claims(template)
        
        # 2. Resolve citations $cite(...)
        template = self.resolve_citations(template)
        
        # 3. Resolve claim references ${claim.ID.field}
        template = self.resolve_claim_refs(template)
        
        # 4. Resolve metrics ${METRIC}
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
    flat: dict[str, str] = {}
    for section in raw.values():
        if isinstance(section, dict):
            for key, value in section.items():
                # Convert to string
                if isinstance(value, bool):
                    flat[key] = str(value).lower()
                elif isinstance(value, (int, float)):
                    flat[key] = str(value)
                else:
                    flat[key] = str(value)
    
    return flat


def render_readme(
    template_path: Path,
    metrics_path: Path,
    output_path: Path,
    repo_root: Optional[Path] = None
) -> None:
    """Render README.template → README.md with full templating."""
    if repo_root is None:
        repo_root = template_path.parent.parent
    
    # Load inputs
    template = template_path.read_text(encoding='utf-8')
    metrics = load_metrics(metrics_path)
    
    # Render
    renderer = TemplateRenderer(repo_root, metrics)
    output = renderer.render(template)
    
    # Write
    output_path.write_text(output, encoding='utf-8')


def check_readme_stale(
    template_path: Path,
    metrics_path: Path,
    readme_path: Path,
    repo_root: Optional[Path] = None
) -> bool:
    """Return True if README.md is stale (needs regeneration)."""
    if not readme_path.exists():
        return True
    
    if repo_root is None:
        repo_root = template_path.parent.parent
    
    # Render to temp
    template = template_path.read_text(encoding='utf-8')
    metrics = load_metrics(metrics_path)
    
    renderer = TemplateRenderer(repo_root, metrics)
    rendered = renderer.render(template)
    
    # Compare
    current = readme_path.read_text(encoding='utf-8')
    return rendered.strip() != current.strip()


def render_readme(
    template_path: Path,
    metrics_path: Path,
    output_path: Optional[Path] = None,
    *,
    check: bool = False,
    repo_root: Optional[Path] = None
) -> tuple[str, bool]:
    """Render README template with full v2 templating (compatible with v1 API).

    Parameters
    ----------
    template_path:
        Path to template file (docs/README.template.md)
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
        Defaults to template_path.parent.parent

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
        repo_root = template_path.parent.parent
    
    # Load and render
    template = template_path.read_text(encoding='utf-8')
    metrics = load_metrics(metrics_path)
    
    renderer = TemplateRenderer(repo_root, metrics)
    rendered = renderer.render(template)

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

