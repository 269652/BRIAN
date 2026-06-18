# BRIAN — Biologically Realistic Information Architecture Network

> *146.9M trainable-param bowtie trunk · ~980M frozen cortex experts · exploring integrated information (Φ).*

[![tests](https://img.shields.io/badge/tests-3518%20passing-brightgreen)](#tests)
[![python](https://img.shields.io/badge/python-3.10%2B-blue)]()
[![torch](https://img.shields.io/badge/torch-2.x-orange)]()
[![license](https://img.shields.io/badge/license-research-lightgrey)]()
[![cortex-fusion](https://img.shields.io/badge/cortex--fusion-KL%20+%20NT--gated-blueviolet)]()
[![formal-gate](https://img.shields.io/badge/improvement--gate-Welch's%20t-9cf)]()

BRIAN is a research prototype that bets on **topology, Φ-coupled plasticity, and closed-loop embodiment** instead of raw parameter count. The core question: does a strategically-wired 146.9M-param trunk outgeneralize a flat 100M transformer on OOD tasks?

**Current verdict:** 🟡 inconclusive — best variant B4 achieves **2.87 gap\_ratio** (53% better than flat-transformer baseline at 6.12), but matched-compute comparison is still pending.

---

## What it does

BRIAN combines five mechanisms into a single differentiable training loop:

| Pillar | What it is | Verified |
|--------|-----------|---------|
| **11-stage bowtie + re-entry loops** | Two re-entry paths enforce non-zero integrated information Φ | ✅ H1 |
| **Differentiable Φ objective** | Gaussian-MI MIP pushes gradients toward integrated states | ✅ H2–H3 |
| **Sheaf H¹ contradiction detection** | Narrative memory detects and resolves conflicting beliefs | ✅ H4–H5 |
| **Embodied survival loop** | GridWorld 10×10 with homeostatic drive shapes qualia and policy | ✅ H6.5 |
| **Multi-cortex fusion** | 3 frozen LM experts distil into the bowtie trunk via KL + NT-gated α | ✅ H16–H21 |

Full architecture spec and tensor shapes: [`docs/architecture.md`](docs/architecture.md).

---


![Neural Flow Graph — current architecture](.neuro/nfg.svg)

*The NFG is generated from `arch.neuro` → Hypergraph IR → PyTorch. Re-render with:*

```powershell
brian compile nfg --current          # → .neuro/nfg.png
brian compile nfg --current --heat heatmap.json   # → .neuro/nfg.heat.png
```

---

## The `.neuro` DSL

Architecture is specified declaratively in `.neuro` files — math-first ODEs and modulation rules that compile to byte-equivalent PyTorch:

```neuro
export population amygdala {
    count: 32,
    ode: "dV/dt = (-V + x) / tau",
    timescale: 0.005
}

modulation dopamine -> pfc {
    effect: "multiplicative", gain: 0.6,
    equation: "y = output * (c * gain)"
}
```

The compiler (`neuroslm/compiler/module_bundler.py`, `ribosome.py`) produces modules with **source maps** and **byte-identity round-trip** verification. 956 tests in `tests/dsl/` guard codegen correctness.

Full reference: [`docs/dsl.md`](docs/dsl.md).

---

## Multi-Cortex Fusion

Three frozen causal-LM experts sit above the bowtie trunk and fuse logits at the LM head:

| Domain | Expert | Params | Tokenizer bridge |
|--------|--------|--------|-----------------|
| General | `smollm2_360m` | ~360M | cross-vocab retokenise |
| Code | `microsoft/CodeGPT-small-py` | ~125M | direct (shared BPE) |
| Reasoning | `Qwen/Qwen2.5-0.5B` | ~500M | cross-vocab retokenise |

|--------|--------|--------|-----------------|
| General | `smollm2_360m` | ~360M | cross-vocab retokenise |
| Code | `microsoft/CodeGPT-small-py` | ~125M | direct (shared BPE) |
| Reasoning | `Qwen/Qwen2.5-0.5B` | ~500M | cross-vocab retokenise |

Three interlocking mechanisms govern the fusion:

1. **`cortex_pre_head_norm`** — LayerNorm before the tied head suppresses GPT-2's rogue dimension (std ≈ 24, 82× median). Without it, CE at step 0 = 21.3 nats (above the uniform-distribution ceiling of 10.82). With it: **10.6 nats**.

2. **Per-position abstain logit (H21)** — Unmapped vocab slots are filled with `max(mapped_logits) − ln(V_trunk)` instead of a flat `−1e4`. This single fix dropped standalone-cortex CE from **17.37 → 4.03 nats** and unlocked the entire multi-cortex pathway:

   | Metric | broken (vast 40923107) | fixed (vast 40925851) | Δ |
   |--------|--------|-------|---|
   | train PPL @ 2,000 steps | 1444 | **102.9** | -93% |
   | OOD PPL (WikiText-103-v1) | 4655 | **295.9** | -94% |
   | `α_eff` | 0.000 (collapsed) | **0.500** (stable) | fusion alive |
   | `cortex_loss_ema` | ~0.001 | **~0.5** | fusion alive |

3. **NT-mediated α gating** — Once the trunk surpasses the cortex, an EMA inhibitory signal drives `α_eff → 0` so cortex experts retire automatically. The reverse also holds: cortex contribution resumes if the trunk regresses.

KL distillation runs in parallel: `L_KL = T² · KL(cortex.detach()/T ‖ trunk/T)` with a gap-ramped λ that saturates at 1.0 when the trunk lags and shuts off when it leads.

---

## Evidence

### Layer A — Mechanism Verification ✅

3518 unit tests across `tests/` confirm every mechanism computes as specified:

| Hypothesis | Result |
|-----------|--------|
| H1 — Φ > 0 for coupled outputs | ✅ Gaussian-MI MIP verified |
| H2 — Φ gradient is real | ✅ ‖∂L/∂θ‖ increases measurably |
| H3 — BDNF grows high-Φ paths preferentially | ✅ kernel rank expands on hot paths |
| H4 — Sheaf H¹ detects contradictions | ✅ "likes coffee" vs "hates coffee" → SUPERSEDES edge |
| H5 — Causal generalization from narratives | ✅ P(Joy\|Gift) > 0.8 from 10 examples |
| H6 — Personality survives weight reload | ✅ identity vector stable across checkpoints |
| H16 — `cortex_pre_head_norm` kills init loss | ✅ CE: 21.3 → 10.6 nats |
| H19 — `ImprovementGate` (Welch's t) | ✅ p-values within 1e-6 of scipy; mutation blocked without significance |
| H21 — Per-position abstain unblocks fusion | ✅ -93% train-PPL / -94% OOD-PPL vs broken precursor |

Run all: `py -3 -m pytest tests/ -v` (~~439s on CPU).

### Layer B — OOD Generalization 🟡

Evaluated on WikiText-103-v1 held-out set. **gap\_ratio = OOD\_ppl / train\_ppl** (lower is better):

| Variant | Params | Steps | train\_ppl | OOD\_ppl | gap\_ratio |
|---------|--------|-------|-----------|---------|-----------|
| Flat Transformer (Baseline) | 106.9M | 80,000 | 66.0 | 404.0 | **6.12** |
| BRIAN B1 (trunk + recursive) | 108.2M | 5,000 | 216.5 | 1372.8 | 6.34 |
| BRIAN B2 (trunk + ReZero) | 107.8M | 7,000 | 258.8 | 1351.5 | 5.22 |
| BRIAN B3 (PCT trunk) | 69.2M | 4,000 | 400.9 | 1806.6 | 4.51 |
| **BRIAN (abstain-fix + multi-cortex, B4)** | **889.6M** | **2,000** | **102.9** | **295.9** | **2.87** |

B4 is the first variant under 3.0 gap\_ratio — a 53% improvement over the flat baseline — achieved at 40x fewer steps. Absolute OOD PPL (295.9) still trails the baseline (404.0), but the baseline ran 80,000 steps. Matched-compute comparison is the immediate next experiment.

> ⚠️ gap\_ratio drifts upward within B4 (2.05 → 2.87 from step 500 → 2,000). The 10k follow-up run will distinguish plateau from accelerating overfit. See [`docs/findings.md#H21`](docs/FINDINGS.md#h21--per-position-abstain-logit-fixes-catastrophic-cortex-ce-2026-06-14).

---

## Quick Start

```bash
python -m venv .venv
# Windows:
.\.venv\Scripts\Activate.ps1
# Linux/Mac:
source .venv/bin/activate

pip install -r requirements.txt
# Install torch separately to match your accelerator:
pip install torch --index-url https://download.pytorch.org/whl/cu121

# CPU sanity run (~~27M params)
brian train --preset=tiny --steps=2000

# A100 full run (~~240M params, bf16)
brian train --preset=xl --steps=100000 --device=cuda

# Resume latest checkpoint
python -m neuroslm.train --resume latest

# Flat transformer ablation at matched params
python -m neuroslm.train --preset xl --baseline

# Interactive generation
python -m neuroslm.generate --prompt "Once upon a time"
```

Full Colab workflow (clone → ablation → training → benchmarks): [`colab_run.ipynb`](colab_run.ipynb).

---

## Parameter Presets

| Preset | Params | Accelerator | VRAM | Notes |
|--------|--------|-------------|------|-------|
| `tiny` | ~~27M | CPU | — | sanity / CI |
| `small` | ~~93M | CPU | — | local dev |
| `medium` | ~~389M | T4 | 16 GB | |
| `large` | ~~100M | T4 | 15 GB | |
| `xl` | ~~240M | A100 | 40 GB | standard research run |
| `xxl` | ~~10B | 4×A100 | 320 GB | |

Add `--baseline` for a parameter-matched flat transformer ablation.

---

## Configuration (`brian.toml`)

Single source of truth for the active architecture:

```toml
[current]
arch = "architectures/rcc_bowtie"   # active architecture
dna  = ""                            # .dna path for evolutionary training

[nfg]
output = ".neuro/nfg.png"
format = "png"                       # png | svg | pdf | dot
engine = "dot"
```

Override per-run with env vars: `BRIAN_ARCH`, `BRIAN_DNA`, `BRIAN_NFG_OUTPUT`. Contract locked by 27 tests in [`tests/test_project_config.py`](tests/test_project_config.py).

---

## Real-Time Architecture Evolution

BRIAN can mutate its own architecture during training. Mutations are gated by `ImprovementGate` (Welch's t-test) — no structural change lands without statistically significant fitness gain:

```python
from neuroslm.utils import EvolutionaryTrainingContext

with EvolutionaryTrainingContext("dna/base.dna", "checkpoints/") as ctx:
    harness = BRIANHarness(ctx.arch_path, resume_from=ctx.resume_step)
    for step in range(ctx.resume_step, 10000):
        loss = harness.train_step(batch)
        if step % 1000 == 0:
            harness.checkpoint_mutations()   # emits step_XXXXX.patch.dna
```

- RAID-5 protected DNA (triple redundancy)
- Incremental patches only — not full model state
- Hot paths (ρ > 0.7) grow via BDNF; cold paths (ρ < 0.1) prune
- Fault-tolerant: patch stack replayed from any checkpoint

---

## Loss Composition

| Term | Source | Weight |
|------|--------|--------|
| `lm_loss` | mesolimbic-gain-modulated cross-entropy | 1.0 |
| `phi_loss` | `−tanh(Φ/3)·3` from MIP estimator | 0.02 × maturation |
| `world_loss` | predicted vs target world embedding (MSE) | 0.3 × maturation |
| `pred_coding_loss` | per-layer next-layer prediction | 0.1 × maturation |
| `cortex_kl_loss` | `T²·KL(cortex.detach()/T ‖ trunk/T)` | λ\_t (gap-ramped, max 1.0) |
| motor, CPC, RSSM, novel aux | embodied + optional modules | 0.05–0.1 × maturation |

The maturation gate `_aux_w_scale ∈ [0.001, 1.0]` suppresses all aux losses until step 5000 (or lm\_loss < 7.5), so the LM gradient dominates during early training.

---

## Introspection

```python
model.intelligence_metrics.snapshot()   # Φ, identity drift, causal density, self-reference rate
model.consciousness_metrics.per_tick()  # γ (binding), θ (memory), α (idling), coherence, ignition
model.narrative_stack.query_rules()     # discovered causal patterns with support counts
model.personality_vector               # tensor(5) — stable across checkpoints
```

---

## Tests

```bash
py -3 -m pytest tests/                                              # full suite (3518 tests, ~~439s)
py -3 -m pytest tests/test_phi.py -v                               # H1–H3: integrated information
py -3 -m pytest tests/test_narrative_memory.py -v                  # H4–H5: memory & causation
py -3 -m pytest tests/test_cognitive_closure.py -v                 # H6–H6.5: identity & embodiment
py -3 -m pytest tests/training/test_cortex_pre_head_norm.py -v     # H16: catastrophic-loss fix
py -3 -m pytest tests/training/test_cortex_distillation_and_gating.py -v  # H17–H18: KL + NT gating
py -3 -m pytest tests/verification/test_improvement_gate.py -v     # H19: Welch's t admission gate
py -3 -m pytest tests/dsl/ -v                                       # 956 DSL codegen + byte-equivalence
```

---

## Checkpoints

Pushed to `moritzroessler/BRIAN` on HuggingFace Hub every 2500 steps. Configure in `architectures/*/config.neuro`:

```neuro
checkpoint {
    push_backend: "hf"
    hf_repo_id: "moritzroessler/BRIAN"
    hf_token_env: "HF_TOKEN"
    save_every: 500
    push_every: 2500
    push_optimizer: false    # strips Adam state, ~2/3 size saving
}
```

**Skip LFS on laptops** (recommended):
```bash
git lfs install --local --skip-smudge   # fetch stubs only
git lfs pull --include="lfs_checkpoints/neuroslm_xl_adamw_mix_800.pt"  # pull one when needed
```

---

## Docs

| Document | Contents |
|----------|----------|
| [`docs/findings.md`](docs/findings.md) | Hypothesis ledger H1–H14: test files, result JSONs, raw logs. Source of truth for what's proven vs open. |
| [`docs/architecture.md`](docs/architecture.md) | Full spec: 11-stage forward pass, tensor shapes, equations, IIT 4.0 theory. |
| [`docs/formal_framework.md`](docs/formal_framework.md) | Normative math contract: sheaf ontology, H¹ guard, Φ guard, RAID-5 DNA, ImprovementGate spec, Lean roadmap. |
| [`docs/dsl.md`](docs/dsl.md) | `.neuro` syntax, macro system, compile pipeline, module bundling, source maps. |
| [`docs/technical_report.md`](docs/technical_report.md) | Executive summary: proven claims, open questions, all 7 pillars. |

---

*Open research. Issues, stars, and PRs welcome.*
