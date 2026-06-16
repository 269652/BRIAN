"""Collect multi-cortex expert roster from arch.neuro + alias registry."""
import re
from pathlib import Path

METRICS = [
    "EXPERT_GENERAL_MODEL", "EXPERT_GENERAL_ROLE",
    "EXPERT_CODE_MODEL", "EXPERT_CODE_ROLE",
    "EXPERT_REASONING_MODEL", "EXPERT_REASONING_ROLE",
]


def collect(root: Path) -> dict[str, str]:
    """Parse expert roster from arch.neuro, resolve aliases via neuroslm.experts."""
    import sys
    metrics: dict[str, str] = {}

    # Ensure neuroslm is importable
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))

    # Find active arch.neuro
    for arch_dir in ["architectures/current", "architectures/master"]:
        arch_path = root / arch_dir / "arch.neuro"
        if arch_path.exists():
            text = arch_path.read_text(encoding="utf-8", errors="replace")
            break
    else:
        return metrics

    # Parse expert roster: { id: "alias", domain: "role" }
    expert_pattern = re.compile(
        r'\{\s*id:\s*"([^"]+)"\s*,\s*domain:\s*"([^"]+)"', re.MULTILINE
    )

    # Import alias resolver
    try:
        from neuroslm.experts import resolve_expert_alias
    except ImportError:
        resolve_expert_alias = None  # type: ignore[assignment]

    role_map = {"general": "GENERAL", "code": "CODE", "reasoning": "REASONING"}

    for match in expert_pattern.finditer(text):
        alias_or_id = match.group(1)
        domain = match.group(2)

        # Resolve to canonical HF name
        if resolve_expert_alias is not None:
            try:
                canonical = resolve_expert_alias(alias_or_id)
            except (ValueError, KeyError):
                canonical = alias_or_id
        else:
            canonical = alias_or_id

        role_key = role_map.get(domain)
        if role_key:
            metrics[f"EXPERT_{role_key}_MODEL"] = canonical
            metrics[f"EXPERT_{role_key}_ROLE"] = domain

    return metrics
