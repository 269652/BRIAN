# Contributing to NeuroSLM

This guide is for contributors and future AIs working on NeuroSLM.

**Table of Contents**
1. [Core Principles](#core-principles)
2. [Development Setup](#development-setup)
3. [Code Organization](#code-organization)
4. [The `.neuro` DSL](#the-neuro-dsl)
5. [Testing & Evidence](#testing--evidence)
6. [Documentation & Synchronization](#documentation--synchronization)
7. [Adding Architectural Changes](#adding-architectural-changes)
8. [Training & Evaluation](#training--evaluation)

---

## Core Principles

These apply to ALL work in this repo. See CLAUDE.md for the full context; this is the contributor summary.

### 1. Test-Driven Development (TDD)

Any non-trivial code change starts with a **failing test**:

```
1. Find or create the right test file under tests/
2. Write the test that captures desired behavior
3. Run it and confirm it fails (for the right reason)
4. Implement the minimal change to make it pass
5. Run the full test suite
6. Commit the test + implementation together
```

**Example:** Adding a new neuromodulator (e.g., histamine):

```python
# tests/test_neurochem.py
def test_histamine_concentration_tracks_targets():
    brain = Brain.from_preset("30m_p4")
    assert "histamine" in brain.transmitters.nt_levels
    # ... actual test logic
```

Then implement `TransmitterSystem.histamine` with test passing.

### 2. Architecture-First, Code-Second

Changes to the model architecture go in **`.neuro` files first**, not in Python:

```neuro
neurotransmitter histamine {
    base_concentration: 0.10,
    release_rate: 0.15,
    reuptake_rate: 0.85
}
```

This declarative approach means:
- The architecture is git-diff-readable
- Parameter tweaks don't require code changes
- DSL → Python compilation is automatic

### 3. Claims Require Evidence

Every architectural claim must be tied to an artifact:

- **Layer A (mechanism):** Named test in `tests/` that passes.
- **Layer B (performance):** Result JSON in `results/` with metrics.

See [Evidence Standards](#evidence-standards) below.

### 4. Synchronize Docs When Code Changes

If you modify `arch.neuro`:

```
arch.neuro change
    ↓
Update architecture.md (if explaining the change)
    ↓
Update technical_report.md (if headline metrics change)
    ↓
Run: python scripts/maintain_technical_report.py --fix
    ↓
Commit together
```

The script archives stale docs and warns about drift.

---

## Development Setup

### Prerequisites

```bash
# Python 3.9+
python --version

# Clone and install
git clone https://github.com/your-fork/neuroslm.git
cd neuroslm

# Dependencies
pip install -r requirements.txt
pip install -e .  # editable install

# Verify
py -3 -m pytest tests/test_phi.py::test_phi_higher_for_coupled_outputs -v
# Should pass in ~1s
```

### IDE Setup (VS Code)

```json
// .vscode/settings.json
{
  "python.defaultInterpreterPath": ".venv/bin/python",
  "python.linting.enabled": true,
  "python.linting.pylintEnabled": true,
  "python.testing.pytestEnabled": true,
  "editor.formatOnSave": true,
  "python.formatting.provider": "black"
}
```

### Quick Test

```bash
# All Layer A tests (mechanisms)
py -3 -m pytest tests/ -v

# Quick smoke test (one file)
py -3 -m pytest tests/test_phi.py -v

# Run with coverage
py -3 -m pytest tests/ --cov=neuroslm --cov-report=html
```

---

## Code Organization

```
neuroslm/
├── dsl/                    # DSL parser, compiler, code generator
│   ├── nn_lang.py         # DSL parser (neuro syntax)
│   ├── multifile.py       # Multi-file compilation
│   ├── codegen.py         # Emit nn.Module from AST
│   └── training_config.py # Load training blocks from .neuro
│
├── modules/               # Core PyTorch nn.Module implementations
│   ├── language.py        # LanguageCortex (trunk transformer)
│   ├── workspace.py       # GlobalWorkspace (bowtie bottleneck)
│   ├── consciousness.py   # Φ (integrated information) computation
│   ├── *_cortex.py        # Expert cortices (math, reasoning, etc.)
│   └── common.py          # Shared utilities
│
├── intelligence/          # High-level systems
│   ├── orchestrator.py    # 11-stage forward pass + re-entry
│   ├── narrative_store.py # Episodic memory
│   ├── causal_head.py     # Causal inference
│   └── latent_program_bus.py
│
├── neurochem/             # Plasticity, homeostasis
│   ├── growth.py          # Trophic system (BDNF-driven growth)
│   ├── transmitters.py    # 7 NT systems
│   └── hebbian.py         # Hebbian fast weights
│
├── harness.py             # BRIANHarness (training wrapper)
├── train_dsl.py           # DSL-driven training entrypoint
└── train.py               # Hand-written Brain training (legacy)

tests/
├── test_phi.py            # Φ computation, MIP algorithm
├── test_brain_forward.py  # Forward pass correctness
├── test_neurochem.py      # Plasticity, trophic growth
├── test_narrative_memory.py # Causal rules, contradiction detection
├── test_cognitive_closure.py # Embodied survival, identity
├── test_pct_smoke.py      # Predictive coding trunk
├── test_stabilization.py  # Gradient isolation, ReZero gates
└── test_recursive_reasoning.py # Recursive expert loops

architectures/rcc_bowtie/
├── arch.neuro             # Main config (training, presets, topology)
├── modules/               # Region definitions (.neuro files)
│   ├── sensory.neuro
│   ├── thalamus.neuro
│   ├── gws.neuro
│   ├── *_cortex.neuro
│   └── ...
└── nfg.png                # Graph visualization (auto-generated)

results/
└── ood_*.json             # OOD evaluation results (Layer B evidence)

docs/
├── technical_report.md    # This report (auto-maintained)
├── architecture.md        # Detailed spec (§0–12 of the full design)
├── findings.md            # Hypothesis ledger (Layer A + Layer B results)
├── history.md             # Session notes + decisions (auto-maintained)
├── changelog.md           # Git-derived (auto-maintained)
└── archive/               # Stale docs (timestamped)
    └── YYYY-MM-DD_*.md
```

---

## The `.neuro` DSL

### Minimal Example

```neuro
architecture my_brain {
    d_sem: 256,
    dt: 0.01
}

training {
    preset: "small",
    loss_clipping: { enabled: true, method: "per_sample", factor: 3.0 },
    dropout: 0.1,
    optimizer: "adamw",
    learning_rate: 0.0003,
    batch_size: 16,
    seq_len: 1024,
    steps: 10000,
    
    scales: {
        small: {
            d_model: 256,
            depth: 4,
            n_heads: 4,
            batch_size: 16
        }
    }
}

param_scope trunk {
    populations: [lang_cortex, pfc],
    gradient: "normal"
}

neurotransmitter dopamine {
    base_concentration: 0.10,
    release_rate: 0.20,
    reuptake_rate: 0.80
}

import { lang_cortex } from "@/modules/language"
import { pfc } from "@/modules/prefrontal"

synapse lang_cortex -> pfc {
    weight: 0.8,
    neurotransmitter: "glutamate",
    equation: "y = weight * (x_pre @ W)"
}

modulation dopamine -> pfc {
    effect: "multiplicative",
    gain: 0.6,
    equation: "y = output * (c * gain)"
}
```

### Key Constructs

| Construct | Purpose | Example |
|-----------|---------|---------|
| `architecture {}` | Global settings | `d_sem: 256` |
| `training {}` | Hyperparameters + presets | `optimizer: "adamw"`, `batch_size: 16` |
| `param_scope name {}` | Gradient isolation groups | `trunk` (full grad), `bio` (detached) |
| `neurotransmitter name {}` | Define NT system | 7 classical NT systems |
| `import { a, b } from "path"` | Load regions from subfiles | `import { gws } from "@/modules/gws"` |
| `synapse A -> B {}` | Synaptic projection | Wiring diagram edge |
| `modulation NT -> region {}` | Neuromodulator effect | Gain/bias scaling |
| `formal_spec name {}` | Constraint (informational) | `rule: "integrated_information"` |

### Best Practices

1. **Keep presets simple:** One line per preset with dims/depth/batch.
2. **Comments are OK:** Use `#` for clarification, not narrative.
3. **One architecture per folder:** `architectures/rcc_bowtie/` is self-contained.
4. **Validate syntax:** `python -c "from neuroslm.dsl.nn_lang import parse; parse(open('arch.neuro').read())"`.

---

## Testing & Evidence

### Layer A Tests (Mechanism Confirmation)

Test that a module **behaves as specified**, not whether it improves the model.

**Template:**

```python
import pytest
import torch
from neuroslm import Brain
from neuroslm.modules.consciousness import _compute_phi_mip

def test_new_mechanism_computes_correctly():
    """
    [CLAIM: NewMechanism computes XYZ]
    
    Test that the mechanism implements its spec faithfully.
    Should be:
    - Deterministic (no randomness in forward pass)
    - Fast (~<100ms on CPU for a single forward)
    - Gradient-passable (∂L/∂input is real)
    """
    # Setup
    brain = Brain.from_preset("tiny")
    
    # Forward
    output = brain.new_mechanism(torch.randn(2, 256))
    
    # Assert
    assert output.shape == (2, 256)
    assert not torch.isnan(output).any()
    assert output.requires_grad
    
    # Backward
    loss = output.sum()
    loss.backward()
    assert brain.new_mechanism.weight.grad is not None
```

**Evidence artifact:** The test itself (e.g., `tests/test_new.py::test_new_mechanism_computes_correctly`)

### Layer B Tests (Architectural Performance)

Evaluate on OOD corpus (WikiText-103-v1). This is the expensive part.

**Recipe:**

```bash
# 1. Train a checkpoint (on vast.ai or local GPU)
python -m neuroslm.train_dsl \
    --arch architectures/rcc_bowtie \
    --scale 30m_p4 \
    --steps 10000 \
    --ckpt_dir lfs_checkpoints

# 2. Evaluate OOD
bash scripts/vast_ood_eval.sh \
    CKPT=lfs_checkpoints/neuroslm_pct_30m_68M_adamw_mix_best.pt \
    BRANCH=arch/my-feature \
    ROLE_TAG=my-feature-run

# 3. Copy result to results/
cp results_remote.json results/ood_my_feature_68M_step4000.json

# 4. Update findings.md and technical_report.md
# Link the new result JSON and describe the hypothesis
```

**Evidence artifact:** `results/ood_my_feature_*.json` with metrics.

### Running Tests

```bash
# All Layer A tests
py -3 -m pytest tests/ -v

# Specific test
py -3 -m pytest tests/test_phi.py::test_phi_higher_for_coupled_outputs -v

# With output capture (show print statements)
py -3 -m pytest tests/ -v -s

# Parallel (if tests are independent)
py -3 -m pytest tests/ -n auto
```

---

## Documentation & Synchronization

### The Three Documentation Layers

| Layer | File | Purpose | Update Rule |
|-------|------|---------|------------|
| **Spec** | `architecture.md` | Detailed module specs, IIT theory, formulas | When you change the design (§0–12) |
| **Ledger** | `findings.md` | Hypothesis table + Layer A/B evidence links | When you run new evals |
| **Report** | `technical_report.md` | Executive summary + current state | Auto-regenerated; manual edits for narrative |

### Synchronization Workflow

**Before committing an architecture change:**

```bash
# 1. Update the .neuro file
# 2. Update architecture.md if necessary (detailed spec)
# 3. Run the audit script
python scripts/maintain_technical_report.py --verbose

# 4. Check that it reports no errors
# If there are issues, fix them
python scripts/maintain_technical_report.py --fix

# 5. Commit all three together
git add architectures/rcc_bowtie/arch.neuro \
        docs/architecture.md \
        docs/technical_report.md
git commit -m "arch: add new mechanism to Φ computation

- Add [NEW_MECHANISM] to consciousness.py
- Update arch.neuro lines X–Y with new params
- Update architecture.md §3 with formula
- Technical report updated via maintain script
"
```

**Before committing a training/eval result:**

```bash
# 1. Run OOD eval → results/ood_my_variant_68M_step4000.json
# 2. Update findings.md with new row + links + caveats
# 3. Technical report will be re-read for external AI context

git add results/ood_my_variant_*.json docs/findings.md
git commit -m "results: PCT-30M achieves 4.51 gap_ratio (H10 partial)

[EVIDENCE: results/ood_pct-30m_68M_step4000.json]
[EVIDENCE: tests/test_pct_smoke.py::*]

See findings.md::H10 for caveat on cross-scale comparison.
"
```

### Archive Old Docs

When a doc becomes stale (session summary, old methodology note, abandoned experiment log):

```bash
# Move to archive with timestamp
mv docs/OLD_SESSION_NOTES.md docs/archive/2026-06-01_old_session_notes.md

# Or use the maintain script
python scripts/maintain_technical_report.py --fix
```

### External AI Context

If you're using external AIs (NotebookLM, Perplexity, ChatGPT) to analyze this project:

1. **Start with:** `technical_report.md` (comprehensive, all sections linked)
2. **Dive deeper with:** `architecture.md` (detailed formulas, module specs)
3. **Check evidence with:** `findings.md` (Layer A/B results, caveats)
4. **Verify results:** Check `results/` JSONs directly (commit SHAs linked)

The technical report is designed to be self-contained for external consumption.

---

## Adding Architectural Changes

### Example: Add a New Neuromodulator

**Step 0: Design & Planning**

Before writing code, ask yourself:
- What mechanism does it implement (biologically)?
- What is its concentration/level equation?
- Which regions does it modulate, and with what effect (gain)?
- Can it be tested in isolation?

**Step 1: Write the Mechanism Test**

```python
# tests/test_new_neuromodulator.py
def test_new_nt_concentration_follows_equation():
    """
    [CLAIM: NewNT concentration = f(activity, baseline, τ)]
    
    Verify the new NT follows the specified kinetics.
    """
    from neuroslm.neurochem.transmitters import TransmitterSystem
    
    ts = TransmitterSystem(nt_systems={"new_nt": {...}})
    
    # Set initial concentration
    ts.nt_levels["new_nt"] = 0.5
    
    # Step with zero activity
    ts.step(nt_release={"new_nt": 0.0})
    
    # Should decay toward baseline
    assert 0.3 < ts.nt_levels["new_nt"] < 0.5
```

Run it, watch it fail:

```bash
py -3 -m pytest tests/test_new_neuromodulator.py::test_new_nt_concentration_follows_equation -v
# FAILED: NewNT not implemented
```

**Step 2: Implement the Mechanism**

```python
# neuroslm/neurochem/transmitters.py

class TransmitterSystem:
    def __init__(self, nt_systems: Dict):
        # ... existing code ...
        if "new_nt" in nt_systems:
            self.tau_decay["new_nt"] = 0.85
            self.base_conc["new_nt"] = 0.20
    
    def step(self, nt_release: Dict):
        # ... existing steps ...
        if "new_nt" in nt_release:
            level = self.nt_levels["new_nt"]
            tau = self.tau_decay["new_nt"]
            base = self.base_conc["new_nt"]
            new_level = level * tau + base * (1 - tau) + nt_release["new_nt"]
            self.nt_levels["new_nt"] = torch.clamp(new_level, 0, 1)
```

**Step 3: Test Again**

```bash
py -3 -m pytest tests/test_new_neuromodulator.py::test_new_nt_concentration_follows_equation -v
# PASSED
```

**Step 4: Add to `.neuro` File**

```neuro
# architectures/rcc_bowtie/arch.neuro
neurotransmitter new_nt {
    base_concentration: 0.20,
    release_rate: 0.15,
    reuptake_rate: 0.85,
    diffusion_rate: 0.02
}

# And add modulation rules
modulation new_nt -> some_region {
    effect: "multiplicative",
    gain: 0.5,
    equation: "y = output * (c * gain)"
}
```

**Step 5: Update Docs**

```markdown
<!-- docs/architecture.md, new section -->

### NewNT Modulation

NewNT implements [describe mechanism].

| Property | Value | Rationale |
|----------|-------|-----------|
| base_concentration | 0.20 | Baseline activity level |
| release_rate | 0.15 | Sensitivity to trigger signal |
| reuptake_rate | 0.85 | Fast decay (τ ≈ 13 steps) |
```

**Step 6: Commit**

```bash
git add tests/test_new_neuromodulator.py \
        neuroslm/neurochem/transmitters.py \
        architectures/rcc_bowtie/arch.neuro \
        docs/architecture.md

git commit -m "feat: add NewNT neuromodulator system

- Implements kinetic equation: c_new = c*τ + base*(1-τ) + release
- Base concentration 0.20, reuptake rate 0.85
- Modulates sensorimotor regions with gain 0.5
- Test: test_new_nt_concentration_follows_equation PASSED

[EVIDENCE: tests/test_new_neuromodulator.py::*]
[SPEC: docs/architecture.md section NewNT]
"
```

---

## Training & Evaluation

### Local Testing

```bash
# Smoke test: 10 steps on synthetic data
python -m neuroslm.train_dsl \
    --arch architectures/rcc_bowtie \
    --scale tiny \
    --steps 10 \
    --device cpu

# Should complete in ~10s, no crashes
```

### Training on GPU (Local)

```bash
# 30M preset, 1000 steps (≈30 min on A100)
python -m neuroslm.train_dsl \
    --arch architectures/rcc_bowtie \
    --scale 30m_p4 \
    --steps 1000 \
    --device cuda:0 \
    --ckpt_dir checkpoints/local_run
```

### Training on Cloud (vast.ai)

```bash
# Set up environment
export ARCH_PATH=architectures/rcc_bowtie
export SCALE=30m_p4
export NUM_STEPS=10000

# Deploy via vast.ai CLI
vast create instance \
    --image pytorch/pytorch:2.0-cuda11.8-cudnn8-runtime \
    --disk 50 \
    --gpu-count 1 \
    --preset 'GPU_NAME in [A100_SXM4] AND num_gpus=1 AND reliability>0.99'

# Then in the instance:
bash scripts/vast_train_dsl_loop.sh
```

### Evaluation & Result Archival

```bash
# Evaluate a checkpoint
bash scripts/vast_ood_eval.sh \
    CKPT=lfs_checkpoints/neuroslm_30m_p4_step10000.pt \
    BRANCH=my-feature \
    ROLE_TAG=my-feature-eval

# Copy result locally
cp results_remote.json results/ood_my_feature_68M_step10000.json

# Link in findings.md
# Update table, add new row with link to JSON
```

---

## Common Workflows

### "I want to add a new test"

```bash
# Create test file
touch tests/test_my_feature.py

# Add test function
cat > tests/test_my_feature.py << 'EOF'
import pytest
import torch
from neuroslm import Brain

def test_my_feature_computes():
    """[CLAIM: MyFeature does XYZ]"""
    brain = Brain.from_preset("tiny")
    # ...

EOF

# Run test
py -3 -m pytest tests/test_my_feature.py -v

# If it fails, implement the feature
# If it passes, commit
git add tests/test_my_feature.py neuroslm/...
```

### "I want to tune hyperparameters"

```bash
# Edit arch.neuro preset
vim architectures/rcc_bowtie/arch.neuro

# Update technical_report.md
python scripts/maintain_technical_report.py --verbose

# Commit
git add architectures/rcc_bowtie/arch.neuro docs/technical_report.md
git commit -m "tuning: adjust dropout 0.10→0.12 for OOD (P4 push)"
```

### "I want to run a full OOD eval"

```bash
# Train a checkpoint (either locally or on vast.ai)
# Then OOD eval it on WikiText-103-v1
bash scripts/vast_ood_eval.sh \
    CKPT=lfs_checkpoints/neuroslm_30m_p4_best.pt \
    BRANCH=my-feature

# Update findings.md with new row + evidence link
vim docs/findings.md

# Commit
git add results/ood_my_feature_*.json docs/findings.md
git commit -m "results: [VARIANT] achieves [METRICS]

[EVIDENCE: results/ood_my_feature_68M_step4000.json]
See findings.md::H[N] for interpretation.
"
```

---

## Questions?

- **For implementation details:** Read `architecture.md` (§0–12 of the full design)
- **For past results:** Check `findings.md` (hypothesis ledger + Layer A/B artifacts)
- **For training commands:** See `scripts/vast_*.sh` (real, executed recipes)
- **For CLAUDE.md rules:** See repo root `CLAUDE.md` (style + process rules)

---

**Last updated:** 2026-06-01  
**Maintainer:** NeuroSLM project team  
**For AI contributors:** All sections above apply equally to both human and AI agents working in this codebase.
