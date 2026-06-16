"""Collect architecture topology metrics from arch.neuro."""
import re
from pathlib import Path

METRICS = [
    "BOWTIE_STAGES",
    "TRUNK_POPULATIONS",
    "BIO_POPULATIONS",
    "TOTAL_POPULATIONS",
]


def collect(root: Path) -> dict[str, str]:
    """Parse populations and stage count from arch.neuro + orchestrator."""
    metrics: dict[str, str] = {}

    # Find active arch.neuro
    for arch_dir in ["architectures/current", "architectures/master"]:
        arch_path = root / arch_dir / "arch.neuro"
        if arch_path.exists():
            text = arch_path.read_text(encoding="utf-8", errors="replace")
            break
    else:
        return metrics

    # Trunk populations
    trunk_match = re.search(
        r'param_scope\s+trunk\s*\{[^}]*populations:\s*\[([^\]]+)\]',
        text, re.DOTALL
    )
    if trunk_match:
        trunk_pops = [p.strip() for p in trunk_match.group(1).split(",") if p.strip()]
        metrics["TRUNK_POPULATIONS"] = str(len(trunk_pops))

    # Bio populations
    bio_match = re.search(
        r'param_scope\s+bio\s*\{[^}]*populations:\s*\[([^\]]+)\]',
        text, re.DOTALL
    )
    if bio_match:
        bio_pops = [p.strip() for p in bio_match.group(1).split(",") if p.strip()]
        metrics["BIO_POPULATIONS"] = str(len(bio_pops))

    if "TRUNK_POPULATIONS" in metrics and "BIO_POPULATIONS" in metrics:
        metrics["TOTAL_POPULATIONS"] = str(
            int(metrics["TRUNK_POPULATIONS"]) + int(metrics["BIO_POPULATIONS"])
        )

    # Bowtie stages from orchestrator
    orc_path = root / "neuroslm" / "intelligence" / "orchestrator.py"
    if orc_path.exists():
        orc_text = orc_path.read_text(encoding="utf-8", errors="replace")
        m = re.search(r'n_stages\s*=\s*(\d+)', orc_text)
        if m:
            metrics["BOWTIE_STAGES"] = m.group(1)

    return metrics
