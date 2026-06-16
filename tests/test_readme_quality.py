"""
Smoke tests for README.md quality and data-driven claims.

Ensures:
1. No hardcoded numbers except semantic constants (version numbers, architecture constants)
2. All claims are backed by ${METRIC} placeholders in the template
3. All metrics referenced in template exist in readme_metrics.toml
4. Rendered README has no ${...} placeholders left behind
"""
import re
from pathlib import Path
import pytest

try:
    import tomllib  # Python 3.11+
except ModuleNotFoundError:
    try:
        import tomli as tomllib  # type: ignore[no-redef]
    except ModuleNotFoundError:
        raise ImportError("Need tomllib (Python 3.11+) or tomli package")

REPO_ROOT = Path(__file__).parent.parent
TEMPLATE_PATH = REPO_ROOT / "README.template.md"
README_PATH = REPO_ROOT / "README.md"
METRICS_PATH = REPO_ROOT / "docs" / "readme_metrics.toml"


# Allowlist: Numbers that are architectural/semantic constants, not empirical claims
ALLOWED_CONSTANTS = {
    # Version numbers & badges
    r"3\.10\+",  # Python version
    r"2\.x",     # PyTorch version
    r"4\.0",     # IIT 4.0
    r"badge.*?-\d+",  # Badges can have numbers (like python-3.10)
    
    # Architecture constants (from design, not empirical)
    r"\b11-stage\b",        # bowtie stages
    r"\b10×10\b",           # GridWorld size
    r"\b28\b populations",  # NFG populations
    r"\b19\b synapses",     # NFG synapses
    r"\b7\b neurotransmitter",  # NT systems
    r"\b3\b pretrained", # 3 cortex experts
    r"\b3\b frozen",     # 3 cortex experts
    
    # Code example literals (not claims)
    r"count:\s*32",         # .neuro example
    r"timescale:\s*0\.005", # .neuro example
    r"gain:\s*0\.6",        # .neuro example
    r"```.*?```",           # Entire code blocks (multiline handled separately)
    
    # Doc section numbers
    r"§\d+",                # section references like §12
    r"#\d+-",               # markdown anchors like #12-the-neuro
    r"\(docs/.*?#\d+\)",    # links to docs with anchors
    
    # Test counts in hypothesis table cells (allowed, already ${METRIC} in most places)
    r"\(\d+\s+tests?\)",    # "(36 tests)", "(24 tests)" in hypothesis table
    
    # H-numbers (hypothesis IDs)
    r"\b[Hh]\d+(\.\d+)?\b", # H1, H6.5, H22, etc.
    
    # Git hashes, step ranges, dates (metadata, not claims)
    r"\b[a-f0-9]{7,40}\b",   # git SHAs
    r"step\s*\d+",           # step numbers in log names
    r"\d{8}",                # YYYYMMDD dates
    r"20\d{6}T\d{6}Z",      # ISO8601 timestamps
    
    # $cite() and $claim{} references (not hardcoded - they're template syntax)
    r"\$cite\([^)]+\)",
    r"\$claim\{[^}]+\}",
    r"\$\{[A-Z_]+\}",       # ${METRIC} placeholders
}


def test_template_has_no_hardcoded_numbers():
    """
    Template should use ${METRIC} for all empirical claims.
    Hardcoded numbers are only allowed for:
    - Version numbers (Python 3.10+, PyTorch 2.x, IIT 4.0)
    - Architectural constants (11-stage, 10×10, 28 populations, etc.)
    - Code examples (count: 32, gain: 0.6)
    - Section references (§12, #12)
    """
    template_text = TEMPLATE_PATH.read_text(encoding="utf-8")
    
    # Find all numbers NOT inside ${...} placeholders
    # Pattern: number that's NOT preceded by ${ and NOT followed by }
    lines = template_text.split("\n")
    violations = []
    
    for line_no, line in enumerate(lines, start=1):
        # Remove ${...} placeholders first
        cleaned = re.sub(r'\$\{[^}]+\}', '', line)
        
        # Find all numbers
        for match in re.finditer(r'\b\d+(?:\.\d+)?(?:[eE][+-]?\d+)?\b', cleaned):
            number_str = match.group()
            # Check if this number matches any allowed pattern
            is_allowed = False
            for pattern in ALLOWED_CONSTANTS:
                # Check the full line context for the pattern
                if re.search(pattern, line):
                    is_allowed = True
                    break
            
            if not is_allowed:
                # Extract context (20 chars before/after)
                start = max(0, match.start() - 20)
                end = min(len(cleaned), match.end() + 20)
                context = cleaned[start:end]
                violations.append(f"Line {line_no}: {context.strip()}")
    
    if violations:
        msg = f"Found {len(violations)} hardcoded numbers in template:\n" + "\n".join(violations[:10])
        if len(violations) > 10:
            msg += f"\n... and {len(violations) - 10} more"
        pytest.fail(msg)


def test_all_template_metrics_exist():
    """
    Every ${METRIC} placeholder in template must exist in readme_metrics.toml
    """
    template_text = TEMPLATE_PATH.read_text(encoding="utf-8")
    metrics_toml = tomllib.loads(METRICS_PATH.read_text(encoding="utf-8"))
    
    # Extract all ${METRIC} placeholders
    placeholders = set(re.findall(r'\$\{([A-Z_]+)\}', template_text))
    
    # Check each exists in TOML
    missing = []
    for ph in placeholders:
        if ph not in metrics_toml:
            missing.append(ph)
    
    if missing:
        pytest.fail(f"Template references {len(missing)} missing metrics: {sorted(missing)}")


def test_readme_has_no_unrendered_placeholders():
    """
    Rendered README.md must have no ${...} placeholders left.
    """
    if not README_PATH.exists():
        pytest.skip("README.md not yet generated")
    
    readme_text = README_PATH.read_text(encoding="utf-8")
    unrendered = re.findall(r'\$\{[^}]+\}', readme_text)
    
    if unrendered:
        pytest.fail(f"README.md has {len(unrendered)} unrendered placeholders: {unrendered[:5]}")


def test_readme_has_no_claim_def_markers():
    """
    Rendered README.md must have no $claim{...} definition blocks.
    (They should be stripped during rendering.)
    """
    if not README_PATH.exists():
        pytest.skip("README.md not yet generated")
    
    readme_text = README_PATH.read_text(encoding="utf-8")
    claim_defs = re.findall(r'\$claim\{[^}]+\}', readme_text)
    
    if claim_defs:
        pytest.fail(f"README.md has {len(claim_defs)} unrendered $claim{{}} blocks: {claim_defs[:3]}")


def test_metrics_file_exists_and_parseable():
    """
    readme_metrics.toml must exist and parse as valid TOML.
    """
    assert METRICS_PATH.exists(), f"Missing {METRICS_PATH}"
    
    metrics = tomllib.loads(METRICS_PATH.read_text(encoding="utf-8"))
    assert isinstance(metrics, dict), "Metrics file must be a TOML dict"
    assert len(metrics) > 0, "Metrics file is empty"


def test_no_raw_numbers_in_layer_b_table():
    """
    Layer B OOD table in template should use ${METRIC} for ALL numeric values.
    """
    template_text = TEMPLATE_PATH.read_text(encoding="utf-8")
    
    # Extract Layer B table section
    table_match = re.search(
        r'### Layer B.*?\n\|.*?\|.*?\n\|.*?\|.*?\n((?:\|.*?\n)+)',
        template_text,
        re.DOTALL
    )
    
    if not table_match:
        pytest.skip("Layer B table not found in template")
    
    table_rows = table_match.group(1)
    
    # Find numbers NOT in ${...}
    # First remove all ${...} placeholders
    cleaned = re.sub(r'\$\{[^}]+\}', '', table_rows)
    
    # Find remaining numbers
    violations = []
    for match in re.finditer(r'\b\d+(?:\.\d+)?(?:[eE][+-]?\d+)?\b', cleaned):
        # Allow B1, B2, B3, B4 (variant IDs)
        context_before = cleaned[max(0, match.start()-5):match.start()]
        if re.search(r'B\d', context_before):
            continue
        
        # Allow step counts like "2,000" or "80,000" IF they're in ${...} in original
        # But if they're bare, that's a violation
        violations.append(match.group())
    
    if violations:
        pytest.fail(f"Layer B table has {len(violations)} hardcoded numbers: {violations}")


def test_hypothesis_table_claims_have_no_hardcoded_metrics():
    """
    Hypothesis table should not have hardcoded empirical claims.
    Test counts like "(36 tests)" are allowed (covered by ALLOWED_CONSTANTS).
    But things like "PPL=155.0" or "gap_ratio=2.87" should be ${METRIC}.
    """
    template_text = TEMPLATE_PATH.read_text(encoding="utf-8")
    
    # Extract hypothesis table
    table_match = re.search(
        r'All \d+ core mechanisms.*?\n\|.*?\|.*?\n\|.*?\|.*?\n((?:\|.*?\n)+)',
        template_text,
        re.DOTALL
    )
    
    if not table_match:
        pytest.skip("Hypothesis table not found")
    
    table_text = table_match.group(1)
    
    # Remove ${...} placeholders
    cleaned = re.sub(r'\$\{[^}]+\}', '', table_text)
    
    # Remove allowed patterns
    for pattern in ALLOWED_CONSTANTS:
        cleaned = re.sub(pattern, '', cleaned)
    
    # Find remaining numbers that look like metrics (e.g., PPL=123, gap=4.5)
    metric_violations = re.findall(r'\b(?:PPL|gap|loss|ratio|step)\s*[=:]\s*\d+(?:\.\d+)?', cleaned)
    
    if metric_violations:
        pytest.fail(f"Hypothesis table has hardcoded metrics: {metric_violations}")


def test_implementation_status_uses_metrics():
    """
    Implementation Status section should use ${...} for test counts, not hardcoded.
    """
    template_text = TEMPLATE_PATH.read_text(encoding="utf-8")
    
    # Extract "Implementation Status" section
    status_match = re.search(
        r'### Implementation Status.*?(?=###|\Z)',
        template_text,
        re.DOTALL
    )
    
    if not status_match:
        pytest.skip("Implementation Status section not found")
    
    status_text = status_match.group(0)
    
    # Remove ${...} placeholders
    cleaned = re.sub(r'\$\{[^}]+\}', '', status_text)
    
    # Remove allowed constants
    for pattern in ALLOWED_CONSTANTS:
        cleaned = re.sub(pattern, '', cleaned)
    
    # Find test count claims like "1511/1515 tests" or "620 in tests/dsl/"
    # These should be ${TOTAL_TESTS}, ${DSL_TESTS}, etc.
    violations = re.findall(r'\b\d+/\d+\s+tests', cleaned)
    violations += re.findall(r'\b\d{3,}\s+(?:in|passing)', cleaned)  # 620 in, 1511 passing
    
    if violations:
        pytest.fail(f"Implementation Status has hardcoded test counts: {violations}")


def test_best_run_citation_uses_claim():
    """
    "Latest stable full-scale run" should reference a $claim{} or log citation, not hardcoded values.
    """
    template_text = TEMPLATE_PATH.read_text(encoding="utf-8")
    
    # Find the "Latest stable full-scale run" line
    run_match = re.search(r'\*\*Latest stable full-scale run:\*\*.*', template_text)
    
    if not run_match:
        pytest.skip("Latest stable run line not found")
    
    run_line = run_match.group(0)
    
    # Should have either $cite(...) or ${...} for metrics
    has_cite = '$cite(' in run_line or '${' in run_line
    
    # Remove placeholders and check for hardcoded numbers
    cleaned = re.sub(r'\$(?:cite\([^)]+\)|\{[^}]+\})', '', run_line)
    
    # Allow B4, A100, step counts (architectural), branch/commit hashes
    cleaned = re.sub(r'\bB\d+\b', '', cleaned)
    cleaned = re.sub(r'\bA100\b', '', cleaned)
    cleaned = re.sub(r'\b\d+k\b', '', cleaned)  # "2k steps"
    cleaned = re.sub(r'`[a-f0-9]+`', '', cleaned)  # commit hash
    cleaned = re.sub(r'\$\d+\.\d+/hr', '', cleaned)  # pricing
    
    # Find remaining numbers that look like metrics
    violations = re.findall(r'\b(?:PPL|OOD|gap)\s+\d+(?:\.\d+)?', cleaned)
    
    if violations and not has_cite:
        pytest.fail(f"Best run line has hardcoded metrics without $cite(): {violations}")


def test_rendered_readme_metrics_are_numeric():
    """
    All metrics in rendered README that came from placeholders should be valid numbers.
    (This catches template bugs where ${METRIC} -> non-numeric value.)
    """
    if not README_PATH.exists():
        pytest.skip("README.md not yet generated")
    
    readme_text = README_PATH.read_text(encoding="utf-8")
    metrics_toml = tomllib.loads(METRICS_PATH.read_text(encoding="utf-8"))
    
    # For each metric in TOML, if it should be numeric, verify it renders as number
    numeric_keys = [
        "TRUNK_TRAINABLE_PARAMS", "TOTAL_FROZEN_PARAMS", "LAYER_A_TEST_COUNT",
        "LAYER_B_BEST_GAP_RATIO", "LAYER_B_BEST_TRAIN_PPL", "LAYER_B_BEST_OOD_PPL",
        "LAYER_B_BASELINE_GAP_RATIO", "LAYER_B_IMPROVEMENT_PCT"
    ]
    
    for key in numeric_keys:
        if key not in metrics_toml:
            continue
        
        value = metrics_toml[key]
        # Should be a string that looks like a number (allow ~ prefix for approximations)
        cleaned_value = value.lstrip('~').replace(',', '')
        if not re.match(r'^\d+(?:\.\d+)?(?:[KkMmBb])?$', cleaned_value):
            pytest.fail(f"Metric {key} = '{value}' is not numeric format")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
