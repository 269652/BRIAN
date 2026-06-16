"""Collect test suite metrics by running pytest --co (collection only)."""
import subprocess
import sys
from pathlib import Path

METRICS = [
    "TOTAL_TESTS",
    "LAYER_A_TEST_COUNT",  # alias for TOTAL_TESTS (used in README)
    "DSL_TESTS",
    "TRAINING_TESTS",
    "TEST_RUNTIME_SECONDS",
]


def collect(root: Path) -> dict[str, str]:
    """Count tests via pytest --collect-only (no execution)."""
    result = subprocess.run(
        [sys.executable, "-m", "pytest", "tests/", "--co", "-q",
         "--ignore=tests/test_feature_flag_ablation.py"],
        capture_output=True, text=True, cwd=str(root), timeout=120
    )

    lines = [l for l in result.stdout.splitlines()
             if "::" in l and not l.startswith("=")]

    dsl = [l for l in lines if "tests/dsl/" in l or "tests\\dsl\\" in l]
    training = [l for l in lines if "tests/training/" in l or "tests\\training\\" in l]

    total = str(len(lines))

    # Estimate runtime: ~0.12s per test on CPU (empirical)
    est_seconds = int(len(lines) * 0.125)
    runtime = f"~{est_seconds}" if est_seconds < 600 else f"~{est_seconds // 60} min"

    return {
        "TOTAL_TESTS": total,
        "LAYER_A_TEST_COUNT": total,
        "DSL_TESTS": str(len(dsl)),
        "TRAINING_TESTS": str(len(training)),
        "TEST_RUNTIME_SECONDS": runtime,
    }
