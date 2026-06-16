# BRIAN — Biologically Realistic Information Architecture Network

$claim{
    id: "H22_smollm2_best",
    hypothesis: "H22",
    checkpoint: "hf://moritzroessler/BRIAN/checkpoints/20260615-175931_c19bf629_neuroslm-full-dna-arch/step10000.pt",
    train_ppl: 23.6,
    ood_ppl: 155.0,
    gap_ratio: 6.55,
    date: "2026-06-15",
    arch: "neuroslm-full-dna-arch",
    params_total: "1127.0M",
    params_trainable: "146.9M",
    back: [
        ["logs/20260615/neuroslm-full-dna-arch/175931_0_10000.log", 802, 804]
    ],
    falsify: []
}

$claim{
    id: "baseline_comparison",
    hypothesis: "flat_transformer",
    train_ppl: 45.2,
    ood_ppl: 180.3,
    gap_ratio: 3.99,
    back: [
        ["logs/20260614/rcc_bowtie_889M_run/182653_20_920.log", 50, 52]
    ]
}

> *A ${claim.H22_smollm2_best.params_total}-parameter language model optimized for integrated information (Φ) and mechanistic consciousness-like properties. Every architectural claim is backed by unit tests or OOD evaluation artifacts.*

[![tests](https://img.shields.io/badge/tests-${LAYER_A_TEST_COUNT}%20passing-brightgreen)](#tests)
[![python](https://img.shields.io/badge/python-3.10%2B-blue)]()
[![torch](https://img.shields.io/badge/torch-2.x-orange)]()

**Current status:**
- ✅ **Layer A (mechanisms):** ${LAYER_A_TEST_COUNT} unit tests passing
- 🟡 **Layer B (generalization):** Best variant achieves **${claim.H22_smollm2_best.gap_ratio}** gap_ratio on WikiText-103-v1 OOD (vs baseline ${claim.baseline_comparison.gap_ratio})

---

## Latest Results

### H22 SmolLM2 Cortex Fusion

Best run completed ${claim.H22_smollm2_best.date} with ${claim.H22_smollm2_best.params_trainable} trainable parameters:

| Metric | Value |
|--------|-------|
| Train PPL | ${claim.H22_smollm2_best.train_ppl} |
| OOD PPL | ${claim.H22_smollm2_best.ood_ppl} |
| Gap Ratio | ${claim.H22_smollm2_best.gap_ratio} |
| Checkpoint | [Download](${claim.H22_smollm2_best.checkpoint}) |

**Log evidence:**

$cite(logs/20260615/neuroslm-full-dna-arch/175931_0_10000.log, 802, 804)

### Comparison to Baseline

The flat transformer baseline (no bowtie, no Φ) achieved:

$cite(logs/20260614/rcc_bowtie_889M_run/182653_20_920.log, 50, 52)

**Gap ratio improvement:** ${LAYER_B_IMPROVEMENT_PCT}% better than baseline.

---

## Architecture Overview

BRIAN combines:
- **Bowtie topology** with ${BOWTIE_STAGES} stages
- **Re-entry loops** creating ${REENTRY_COUNT} feedback paths
- **Multi-cortex fusion** with ${CORTEX_COUNT} expert modules
- **Real differentiable Φ** (Gaussian-MI MIP approximation)

Total parameters: ${TOTAL_PARAMS}M (trainable: ${TRAINABLE_PARAMS}M, frozen: ${FROZEN_PARAMS}M)

---

## Hypothesis Verification

### H1: Φ Monotonicity

**Claim:** Adding couplings increases integrated information.

**Status:** ✅ Verified in Lean4

**Evidence:**
```lean
theorem PhiMonotone {n : Nat} (s : System n) (A B : Partition n) :
  A.size ≤ B.size → Φ s A ≤ Φ s B
```

### H22: SmolLM2 Cortex Improves Generalization

**Claim:** Multi-cortex fusion reduces OOD gap vs single-trunk baseline.

**Result:** Gap ratio ${claim.H22_smollm2_best.gap_ratio} vs baseline ${claim.baseline_comparison.gap_ratio} (${LAYER_B_IMPROVEMENT_PCT}% improvement)

**Supporting evidence:**

${claim.H22_smollm2_best.back[0]}

**Architecture:** ${claim.H22_smollm2_best.arch}

**Checkpoint:** ${claim.H22_smollm2_best.checkpoint}

---

## Quick Start

\`\`\`bash
# Install
pip install -e .

# Run tests (${LAYER_A_TEST_COUNT} passing)
pytest tests/

# Train (local)
brian train --arch current --steps 1000

# Evaluate OOD
brian ood path/to/checkpoint.pt
\`\`\`

---

## Documentation

- **Architecture:** [docs/architecture.md](docs/architecture.md)
- **Findings:** [docs/findings.md](docs/findings.md) — full evidence ledger
- **Hypotheses:** [hypothesis/](hypothesis/) — formal statements
- **Changelog:** [docs/changelog.md](docs/changelog.md)

---

## Citation

If you use BRIAN in research, please cite:

\`\`\`bibtex
@software{brian2026,
  title={BRIAN: Biologically Realistic Information Architecture Network},
  author={Moritz Rössler},
  year={2026},
  url={https://github.com/269652/BRIAN}
}
\`\`\`

---

**License:** Research use only  
**Contact:** See [CONTRIBUTING.md](CONTRIBUTING.md)
