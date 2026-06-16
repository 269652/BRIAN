# BRIAN — Biologically Realistic Information Architecture Network

> A research prototype testing whether biologically-motivated topology — bowtie re-entry loops, Φ-coupled plasticity, and multi-cortex distillation — reduces the generalization gap vs a flat transformer at matched parameters. Every empirical claim below cites the log or checkpoint that produced it.

[![tests](https://img.shields.io/badge/tests-${TEST_COUNT}%20passing-brightgreen)](#running-tests)
[![python](https://img.shields.io/badge/python-3.13-blue)]()
[![torch](https://img.shields.io/badge/torch-2.x-orange)]()
[![license](https://img.shields.io/badge/license-research-lightgrey)]()

---

## What this is testing

The central bet: a **strategically wired small model** (bipartitioned bowtie, re-entry loops, real Φ gradient) generalizes better out-of-distribution than a flat transformer at the same parameter count, even with far fewer training steps.

That bet is not yet settled. See [Layer B](#layer-b--ood-generalization-the-open-question-) for the current state of evidence.

Full design rationale and tensor-level spec: [`docs/architecture.md`](docs/architecture.md).  
Hypothesis ledger with artifact links: [`docs/findings.md`](docs/findings.md).

---

## Evidence

$claim{
  id: "B6_BEST_OOD",
  train_ppl: "23.6",
  ood_ppl: "155.0",
  gap_ratio: "6.55",
  steps: "10,000",
  desc: "B6 achieved the best absolute OOD PPL in the arc (155.0) at 10k steps with SmolLM2 cortex, but gap_ratio regressed to 6.55 vs B4's 2.87"
}

$claim{
  id: "B4_GAP_BREAKTHROUGH",
  train_ppl: "102.9",
  ood_ppl: "295.9",
  gap_ratio: "2.87",
  steps: "2,000",
  improvement: "First BRIAN variant under gap_ratio 3.0",
  desc: "B4 broke the gap_ratio barrier at 2.87 through the abstain-fix enabling multi-cortex fusion"
}

$claim{
  id: "H21_ABSTAIN_FIX",
  ce_before: "17.37",
  ce_after: "4.03",
  train_improvement: "14×",
  ood_improvement: "17×",
  desc: "Per-position abstain logit fix reduced standalone-cortex CE from 17.37 to 4.03 nats, enabling stable multi-cortex fusion"
}

### Layer A — mechanisms work as specified ✅

These tests assert a constructed `Brain` carries out the mechanism described in `docs/architecture.md`. They prove the primitive exists and computes as written. They say nothing about whether the primitive improves OOD generalization — that is Layer B's job.

| Hypothesis | Test file | What is verified |
|---|---|---|
| **H1** — Φ non-zero for coupled outputs | `tests/test_phi.py::test_phi_higher_for_coupled_outputs` | Gaussian-MI MIP produces Φ > 0 for rank-coupled outputs |
| **H2** — Φ objective injects real gradient | `tests/test_brain_forward.py::test_phi_objective_increases_total_gradient` | ‖∂L/∂θ‖ increases measurably with `phi_loss` term |
| **H3** — BDNF growth biased toward high-Φ paths | `tests/test_neurochem.py::test_trophic_phi_boosts_growth` | High-Φ pathways grow kernel rank preferentially |
| **H4** — Sheaf H¹ detects narrative contradictions | `tests/test_narrative_memory.py::test_sheaf_contradiction_detection` | "Alice likes coffee" vs "hates coffee" → SUPERSEDES edge |
| **H5** — Causal generalization from few-shot narratives | `tests/test_narrative_memory.py::test_causal_generalization` | 10 (Gift→Joy) episodes → P(Joy\|Gift) > 0.8 without gradient updates |
| **H6** — Personality persists across checkpoint reload | `tests/test_cognitive_closure.py::test_autobiographical_personality_consistency` | Identity vector survives weight reload within tolerance |
| **H6.5** — Embodied survival reshapes qualia and policy | `tests/test_cognitive_closure.py::test_survival_*` | Energy drop produces latent-space warp; +RPE updates work |
| **H7** — Trunk gradient isolation prevents post-awakening collapse | `tests/test_stabilization.py` | Detaching `sem` before aux pipeline prevents step-5k divergence |
| **H16** — `cortex_pre_head_norm` removes catastrophic init loss | `tests/training/test_cortex_pre_head_norm.py` (8) | GPT-2 rogue dim (std≈24, 82× median) → CE=13.84 without norm; 10.82±0.5 with it |
| **H17** — KL distillation from cortex to trunk | `tests/training/test_cortex_distillation_and_gating.py::TestDistillation*` (11) | Gradient flows trunk-only; cortex logits detached |
| **H18** — NT-mediated α gating retires cortex when trunk wins | `tests/training/test_cortex_distillation_and_gating.py::TestEffectiveAlpha*` (11) | `α_eff → 0` as `cortex_loss_ema > lm_loss_ema` |
| **H19** — `ImprovementGate` uses Welch's t-test for mutation admission | `tests/verification/test_improvement_gate.py` (16) | p-values within 1e-6 of scipy reference; rejects unless `p < α ∧ effect > min_effect` |
| **H21** — Per-position abstain logit fixes catastrophic cortex CE | `tests/training/test_lm_expert_abstain_safety.py` (5) | Standalone-cortex CE: ${H21_ABSTAIN_CE_BEFORE} → ${H21_ABSTAIN_CE_AFTER} nats on random batch |

Run the full suite: `brian test full` (~8 min). Smoke check: `brian test fast` (~5 s, 30 fastest tests).

---

### Layer B — OOD generalization (the open question) 🟡

Evaluated on WikiText-103-v1 held-out set. Common eval harness: `scripts/vast_ood_eval.sh` calling `brian_ood_test.py` — 200 windows, stride 512 / seq_len 1024, GPT-2 BPE tokenizer (vocab 50257).

`gap_ratio = OOD_ppl / train_ppl` — measures how much worse the model does out-of-distribution relative to in-distribution. Lower is better.

Params split: **trainable** (trunk only, in checkpoint) / **frozen** (expert weights, not saved).

| Row | Arch | Steps | Trainable | Frozen | train_ppl | OOD_ppl | gap_ratio | Artifact |
|---|---|---|---|---|---|---|---|---|
| **B0** flat baseline | vanilla transformer | ${B0_STEPS} | ${B0_TRAINABLE} | — | **${B0_TRAIN_PPL}** | **${B0_OOD_PPL}** | ${B0_GAP_RATIO} | [`${B0_ARTIFACT}`](${B0_ARTIFACT}) |
| **B1** trunk-iso + recursive | bowtie | ${B1_STEPS} | ${B1_TRAINABLE} | — | ${B1_TRAIN_PPL} | ${B1_OOD_PPL} | ${B1_GAP_RATIO} | [`${B1_ARTIFACT}`](${B1_ARTIFACT}) |
| **B2.fix** trunk-iso + ReZero | bowtie | ${B2FIX_STEPS} | ${B2FIX_TRAINABLE} | — | ${B2FIX_TRAIN_PPL} | ${B2FIX_OOD_PPL} | ${B2FIX_GAP_RATIO} | [`${B2FIX_ARTIFACT}`](${B2FIX_ARTIFACT}) |
| **B3** PCT trunk | bowtie | ${B3_STEPS} | ${B3_TRAINABLE} | — | ${B3_TRAIN_PPL} | ${B3_OOD_PPL} | ${B3_GAP_RATIO} | [`${B3_ARTIFACT}`](${B3_ARTIFACT}) |
| **B4** abstain-fix + multi-cortex | bowtie + GPT-2/CodeGPT/Qwen | ${claim.B4_GAP_BREAKTHROUGH.steps} | **${B4_TRAINABLE}** | ${B4_FROZEN} | ${claim.B4_GAP_BREAKTHROUGH.train_ppl} | ${claim.B4_GAP_BREAKTHROUGH.ood_ppl} | **${claim.B4_GAP_BREAKTHROUGH.gap_ratio}** | vast `${B4_VAST_ID}` @ `${B4_GIT_SHA}` — [log](logs/vast/) `${B4_LOG}` |
| **B5** B4 rerun, 10k (mid-run step 3k†) | same as B4, GPT-2 roster | ${B5_STEPS} | ${B5_TRAINABLE} | ${B5_FROZEN} | ${B5_TRAIN_PPL} | ${B5_OOD_PPL} | ${B5_GAP_RATIO} | `${B5_LOG}` |
| **B6** SmolLM2 `general` upgrade | bowtie + SmolLM2/CodeGPT/Qwen | ${claim.B6_BEST_OOD.steps} | **${B6_TRAINABLE}** | ${B6_FROZEN} | **${claim.B6_BEST_OOD.train_ppl}** | **${claim.B6_BEST_OOD.ood_ppl}** | ${claim.B6_BEST_OOD.gap_ratio} | `${B6_LOG}`; ckpt HF `${B6_HF_CKPT}` |

†B5 is a mid-run snapshot; final 10k numbers pending.

**What the table says:**

1. **${claim.B4_GAP_BREAKTHROUGH.improvement}** (${claim.B4_GAP_BREAKTHROUGH.gap_ratio} vs ≥${B3_GAP_RATIO} for B0–B3), driven entirely by the per-position abstain fix (H21) unblocking multi-cortex fusion. The broken precursor (vast `${H21_BROKEN_VAST_ID}`, identical arch) ran with `α_eff=0, cortex_loss_ema≈${H21_BROKEN_CX_EMA}` — fusion collapsed, trunk trained alone, giving train PPL ${H21_BROKEN_TRAIN_PPL} and OOD PPL ${H21_BROKEN_OOD_PPL}. The fix dropped those to ${claim.B4_GAP_BREAKTHROUGH.train_ppl} and ${claim.B4_GAP_BREAKTHROUGH.ood_ppl} (${claim.H21_ABSTAIN_FIX.train_improvement} and ${claim.H21_ABSTAIN_FIX.ood_improvement} respectively). Pinned by `tests/training/test_lm_expert_abstain_safety.py`.

2. **B5 (mid-run, step 3k)** shows a gap_ratio plateau: ${B5_GAP_RATIO}, essentially unchanged from B4's ${claim.B4_GAP_BREAKTHROUGH.gap_ratio}, while absolute PPL continues improving on both axes (train ${B5_TRAIN_PPL}, OOD ${B5_OOD_PPL}). Longer training helps absolute quality without widening the gap — at the GPT-2 roster scale.

3. **B6 (SmolLM2 upgrade, H22):** H22 set four explicit targets. Three were met — train PPL ≤ ${B6_H22_TRAIN_TARGET} ✅ (${claim.B6_BEST_OOD.train_ppl}), OOD PPL ≤ ${B6_H22_OOD_TARGET} ✅ (${claim.B6_BEST_OOD.ood_ppl}, the **best absolute OOD in the arc**), throughput within 15% ✅ — but gap_ratio ≤ ${B6_H22_GAP_TARGET} ❌ (got ${claim.B6_BEST_OOD.gap_ratio}). The absolute OOD result is genuine progress; what regressed is the *ratio*: train PPL fell much faster than OOD PPL, meaning the trunk memorizes the training distribution more aggressively with a stronger teacher. findings.md records this as "falsified on gap_ratio." GFD v2 (commit `${GFD_V2_COMMIT}`) is designed to address the mechanism — stronger teachers should improve generalization proportionally, not just in-distribution fit. See findings.md H22 and run-${H22_POSTMORTEM_VAST_ID} post-mortem for the root-cause chain.

4. **BRIAN has not been shown to outperform the flat baseline at matched compute.** B0 used ${B0_STEPS} steps; B1–B4 used 2–7k steps. The 3–4× absolute-PPL gap at this snapshot is within the range that compute asymmetry alone can explain — a 100M trunk at step 7k is nowhere near converged. gap_ratio (where BRIAN wins by 15–50%) is the only fair comparison axis given the step-count mismatch. Matched-compute baseline (step-7k flat transformer) remains the missing experiment; see findings.md H12.

5. **GFD v2** (commit `${GFD_V2_COMMIT}`, 2026-06-15) adds prior-residual sparsification and pointwise-K from teacher PMI to the distillation funnel. The goal: prevent stronger teachers from accelerating corpus-specific memorization disproportionately. Unit tests pass (${GFD_V2_TEST_COUNT}/${GFD_V2_TEST_COUNT} in `tests/training/test_cfd_distillation.py`); no full training eval yet.

---

## System Architecture

BRIAN is an **11-stage bowtie** with two re-entry loops and five functional subsystems:

```
┌─────────────────────────────────────────┐
│   Sensory → Thalamus → State Models     │
│        ↓                                 │
│   Qualia + Hopfield + Cortical Ignition │  ← within-pass re-entry
│        ↓                                 │
│   Memory + Cognition + Executive        │
│        ↓                                 │
│   Motor Output                          │
│        ↓                                 │
│   [PFC + GWS] ──→ Thalamic crosspass    │  ← cross-pass re-entry
└─────────────────────────────────────────┘
         ↕ (bidirectional)
  ┌─ Narrative Stack (Sheaf H¹)
  ├─ Causal Rule Store
  └─ Personality Vector
```

Every box is a learnable module. Every arrow is a documented tensor operation. Full spec: [`docs/architecture.md`](docs/architecture.md).

**Visual blueprint:** The full bowtie with all populations, synapses, and neurotransmitter systems is rendered in the Neural Flow Graph:

![Neural Flow Graph — current architecture](.neuro/nfg.svg)

Re-render after editing `arch.neuro`:

```powershell
brian compile nfg --current     # writes to .neuro/nfg.svg
```

---

## The `.neuro` DSL

BRIAN's brain architecture is specified declaratively in `.neuro` files — math-first equations that compile to PyTorch at runtime:

```neuro
# architectures/master/modules/amygdala.neuro
export population amygdala {
    count: 32,
    ode: "dV/dt = (-V + x) / tau",
    timescale: 0.005
}

# architectures/master/arch.neuro
modulation dopamine -> pfc {
    effect: "multiplicative", gain: 0.6,
    equation: "y = output * (c * gain)"
}
```

The DSL codegen produces torch modules with byte-equivalent forward passes (verified by 620+ DSL tests in `tests/dsl/`) and supports symbolic analysis. The DNA layer (`neuroslm/compiler/module_bundler.py`, `ribosome.py`) supports module bundling with source maps and byte-identity round-trip verification.

Full reference: [`docs/dsl.md`](docs/dsl.md).

---

## Multi-Cortex Fusion

BRIAN stacks three frozen causal-LM "cortex" experts above the bowtie trunk and fuses their logits at the LM head. Current roster (`architectures/master/arch.neuro`, active as of `${GFD_V2_COMMIT}`):

| Domain | Expert | Tokenizer vs trunk | Path |
|---|---|---|---|
| `general` | `smollm2_360m` (HuggingFaceTB/SmolLM2-360M, ~360M) | different (~49 152 BPE) | bridge (exact-end char-offset align) |
| `code` | `microsoft/CodeGPT-small-py` (~125M) | same (gpt2 BPE) | fast path |
| `reasoning` | `Qwen/Qwen2.5-0.5B` (~500M) | different | bridge |

**SmolLM2 status:** The H22 run (`vast ${B6_VAST_ID}`, log `${B6_LOG}`) FALSIFIED the naive upgrade: SmolLM2 regressed gap_ratio from ${B4_GAP_RATIO} → ${B6_GAP_RATIO} despite better absolute train PPL (${B6_TRAIN_PPL}). Root cause is the `reduction='batchmean'` distillation bug (making KL ~263× larger than LM loss at step 500) combined with cross-family imitation infeasibility. SmolLM2 is re-armed in the current DSL behind the **GFD v2** funnel, which is designed to fix both failure modes. GFD v2 has unit test coverage but no full eval yet.

### Mechanism: per-position abstain logit (H21) ✅

Trunk-vocab IDs the cortex never produces must be filled with an abstain value. The legacy flat `_ABSTAIN_LOGIT = -1e4` poisoned standalone-cortex CE (~10,000 nats per unmapped slot), triggering the NT-mediated inhibition and collapsing `α_eff → 0`. Fix (`${H21_FIX_COMMIT}`):

```
abstain_t = max(mapped_logits_t) − ln(V_trunk)
```

Standalone-cortex CE: **${claim.H21_ABSTAIN_FIX.ce_before} → ${claim.H21_ABSTAIN_FIX.ce_after} nats** on a random batch. Training impact on `30m_p4` preset (2k steps, A100 SXM4, `vast ${B4_VAST_ID}` vs broken precursor `${H21_BROKEN_VAST_ID}`):

| | broken (${H21_BROKEN_VAST_ID}) | fixed (${B4_VAST_ID}) |
|---|---|---|
| train PPL @ step 2000 | ${H21_BROKEN_TRAIN_PPL} | **${claim.B4_GAP_BREAKTHROUGH.train_ppl}** |
| OOD PPL (WikiText-103) | ${H21_BROKEN_OOD_PPL} | **${claim.B4_GAP_BREAKTHROUGH.ood_ppl}** |
| `α_eff` | 0.000 (collapsed) | **0.500** (stable) |

Pinned by `tests/training/test_lm_expert_abstain_safety.py` (5 contracts). Code: `neuroslm/experts.py::LMExpertEnsemble._project_to_trunk_vocab`.

### Mechanism: GFD v2 — Generalization-Focused Distillation ✅ (unit tests only)

The `reduction='batchmean'` KL bug (commit `${GFD_V2_COMMIT}`, Followup F1) is fixed in the GFD path. GFD adds two new stages on top of CFD v1's top-K + entropy-matched temperature:

**M2 — prior-residual sparsification.** Subtract `γ · log p_uni` from teacher logits before top-K. Removes the unigram-marginal component from the distillation gradient (Theorem V, `docs/formal_framework.md §14`). `γ=0` is bit-identical to CFD v1.

**M4 — pointwise-K from teacher PMI.** Per-position `K(t) = clip(K_max · exp(−PMI(t)/scale), K_min, K_max)`. High-PMI positions get small K (concentrate signal on rare-prior peaks); low-PMI positions get large K (soft regulariser on common tokens).

Unit tests: `tests/training/test_cfd_distillation.py` (${GFD_V2_TEST_COUNT} tests, `TestCFDv2PriorResidual`, `TestCFDv2PointwiseK`, `TestCFDv2VariableKTopK`, `TestCFDv2BackCompat`). V1 path is bit-identical when v2 knobs are at defaults.

**No full eval yet.** GFD v2 is the current research frontier. B7 will be the first training run with this path active.

### Mechanism: KL distillation (Slot A) ✅

Per step:

```
L_KL = T² · KL(softmax(cortex.detach()/T) ‖ softmax(lm_logits/T))
λ_t  = λ_max · clip((gap_t − floor) / (ceiling − floor), 0, 1)
```

Cortex logits are detached (gradient into trunk only). When trunk loss is much worse than cortex loss, λ → `lambda_max`; when the trunk catches up, λ → 0. Defaults: `T=4.0`, `λ_max=1.0`, `gap_floor=0.1`, `gap_ceiling=2.0`. Under GFD v2 the reduction is `mean` (per-token) not `batchmean`.

### Mechanism: NT-mediated α gating (Slot C) ✅

```
inhibition_t = (1−β)·inhibition_{t−1} + β·σ((cortex_loss_ema − lm_loss_ema) / T_inh)
α_eff = α · (1 − inhibition_t)
```

When the trunk surpasses the cortex, `inhibition → 1` and `α_eff → 0` — the cortex retires. Reverse is also true. Code: `neuroslm/harness.py::_update_cortex_inhibition`, `_effective_alpha`.

---

## Real-Time Architecture Evolution

Architectural mutations are gated by `ImprovementGate` (Welch's t-test): no mutation lands without statistically-significant fitness gain over a baseline window. Code: `neuroslm/evolution/`. Tests: `tests/verification/test_improvement_gate.py` (16).

DNA compilation round-trips through `neuroslm/compiler/ribosome.py` with byte-identity verified by `tests/test_dna_roundtrip_byte_identity.py`.

---

## Project Configuration (`brian.toml`)

Single source of truth for which architecture / DNA every training, deploy, and eval targets:

```toml
[current]
arch = "architectures/master"   # active architecture
dna  = ""                        # set to a .dna path for DNA-loop training

[nfg]
output = ".neuro/nfg.svg"
format = "svg"
engine = "dot"
```

Contract locked by 27 tests in [`tests/test_project_config.py`](tests/test_project_config.py).

---

## Quick start

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -e .[ml,dev]

# Smoke-check the test suite
brian test fast

# Local CPU training (small preset, ~15M params)
brian train --steps 2000

# Full-scale training on A100 (requires vast.ai deploy)
brian deploy --steps 10000

# OOD eval on a checkpoint
brian ood eval lfs_checkpoints/<checkpoint>.pt
```

---

## Running Tests

```powershell
brian test full                    # canonical sweep (~8 min)
brian test fast                    # 30 fastest tests — smoke check (~5 s)
brian test quick                   # 30 most-recently-edited test files

# Targeted runs
brian test tests/test_phi.py       # H1–H3: integrated information
brian test tests/test_narrative_memory.py   # H4–H5: memory and causation
brian test tests/training/test_cortex_distillation_and_gating.py   # H17–H18
brian test tests/training/test_cfd_distillation.py                  # GFD v1/v2
```

---

## Parameter presets

Checkpoints contain only **trainable** (trunk) parameters. Frozen expert weights are loaded at runtime from HuggingFace and are not checkpointed.

| Preset | Trainable (trunk, in ckpt) | Frozen experts (runtime) | Accelerator |
|---|---|---|---|
| `tiny` | ~5M | ~750M | CPU |
| `small` | ~15M | ~750M | CPU |
| `medium` | ~80M | ~750M | T4 16GB |
| `rcc_bowtie_30m_p4` | **~147M** (B4/B5/B6 measured) | ~743M (GPT-2 roster) / ~980M (SmolLM2 roster) | A100 40GB |

---

## Loss composition

| Term | Source | Weight | Gating |
|---|---|---|---|
| `lm_loss` | trunk cross-entropy (mesolimbic-gain modulated) | `w_lm = 1.0` | always |
| `phi_loss` | `−tanh(Φ/3)·3` from Gaussian-MI MIP | `w_phi = 0.02` | × `_aux_w_scale` |
| `pred_coding_loss` | per-layer next-layer prediction | `w_pred_coding = 0.1` | × `_aux_w_scale` |
| `world_loss` | MSE on world embedding | `w_world = 0.3` | × `_aux_w_scale` |
| `motor_loss` | speak/silent cross-entropy | `w_motor = 0.05` | × `_aux_w_scale` |
| `cortex_kl_loss` | GFD-funneled KL(cortex.detach() ‖ lm_logits) | `λ_t` (gap-ramped, max 1.0) | `cfd_enabled` only |

`_aux_w_scale ∈ [0.001, 1.0]` — topological maturation gate. Aux losses are suppressed during early training so LM gradient dominates while representations form.

---

## Documentation

| Document | Contents |
|---|---|
| [`docs/findings.md`](docs/findings.md) | Hypothesis ledger H1–H22+: every claim tied to a test file, result JSON, or raw log. Source of truth for what is proven vs open. |
| [`docs/architecture.md`](docs/architecture.md) | Full spec: 11-stage forward pass, tensor shapes, equations, module diagrams. |
| [`docs/formal_framework.md`](docs/formal_framework.md) | THSD mathematical framework, GFD theorems (§13–14), ImprovementGate admission spec, Lean roadmap. |
| [`docs/technical_report.md`](docs/technical_report.md) | Executive summary for external collaborators; synced with findings.md. |
| [`docs/dsl.md`](docs/dsl.md) | DSL syntax, macro system, symbol resolution, compile pipeline. |

---

## Open questions

The active research questions, in priority order:

1. **Does GFD v2 fix the SmolLM2 gap_ratio regression?** B7 (GFD v2 + SmolLM2 + 10k steps) is the next planned run.
2. **Does the gap_ratio plateau at ~2.9 or does it drift upward?** B5 at step 3k shows ${B5_GAP_RATIO} ≈ B4's ${claim.B4_GAP_BREAKTHROUGH.gap_ratio}, consistent with a floor. Full B5 (10k) needed to confirm.
3. **Can BRIAN beat the flat baseline at matched compute?** B0 used ${B0_STEPS} steps; no BRIAN variant has been trained that long at this scale. A step-7k flat baseline eval would give the true matched-compute comparison for H12.
4. **Does the `reduction='batchmean'` bug (Followup F1) explain why GPT-2 experts work while SmolLM2 doesn't?** GFD v2 fixes the reduction in the new path; we need an ablation that isolates F1 from M2/M4.

See [`docs/findings.md`](docs/findings.md) for the full backlog.
