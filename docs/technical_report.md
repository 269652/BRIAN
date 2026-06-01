# NeuroSLM Technical Report — Project Overview & Current State

> **Last Updated:** 2026-06-01  
> **Reporting Period:** Inception through OOD regularization push (P4)  
> **Status:** Active research; training stable to ~10k steps at 30M scale  
> **Next Checkpoint:** Matched-compute baseline eval (H12), full PCT integration  

---

## Executive Summary

NeuroSLM is a research project exploring whether **biologically-inspired topology** can achieve better generalization at lower parameter counts than flat transformer baselines. The core claim: a 30M–240M parameter model with cortical-grade modular structure, plasticity, and homeostatic regulation outperforms vanilla transformers at matched compute on out-of-distribution tasks.

**Current evidence:**
- **Layer A (mechanism):** 15 unit tests confirm that core modules (consciousness, plasticity, narrative memory, survival loop) behave as specified. ✅ CONFIRMED
- **Layer B (architecture):** Best variant (PCT-30M, B3) achieves **4.51 OOD gap_ratio** vs baseline 6.12 — **26% better generalization signature**, though at lower parameter count and earlier training step. Still under-converged and cross-scale. 🟡 PARTIAL
- **Training stability:** Reaches step 10k cleanly on 30M P4 preset. Diverges at step 7-10k on 100M baseline without loss clipping.

**Public claim vs reality:**
- README states "measurably better at matched FLOPs than a flat 230M dense transformer" (H12).
- Actual snapshot: flat baseline at 80k steps beats BRIAN variants by ~3-4× on absolute PPL, but BRIAN wins **gap_ratio** by 15% (5.22 vs 6.12). Baseline got 11× more training steps — **compute asymmetry breaks the comparison.**
- Resolution: Pending step-7000 baseline eval (~$3-5).

---

## 1. Project Charter & Research Questions

### 1.1 Hypothesis

**Core claim:** Topological innovation in module structure, connectivity, and homeostatic regulation can compress intelligence density. A biologically-grounded 30–240M model should match or exceed vanilla 1B+ parameter transformers on reasoning and generalization tasks.

**Sub-hypotheses tested:**
1. **Consciousness-first design (Φ):** Maximizing integrated information → better internal binding → improved generalization. [Status: ✅ mechanism confirmed, 🟡 architectural payoff unclear]
2. **Trunk gradient isolation (§5.2):** Detaching bio-module outputs from the shared trunk prevents post-awakening divergence. [Status: ✅ CONFIRMED]
3. **Recursive reasoning depth:** Weight-sharing expert loops add reasoning depth at zero parameters. [Status: 🟡 in-distribution win, no OOD payoff]
4. **Predictive coding trunk (PCT):** Top-down generative predictors suppress train→OOD gap. [Status: 🟡 PARTIAL — 26% improvement but not 2× threshold]
5. **Phased maturation & ReZero gates:** Soft awakening via learnable scalar gates prevents the catastrophic PPL jump at convergence. [Status: 🟡 removes discontinuity, no OOD win]

### 1.2 Non-goals

This project is **not** attempting to:
- Achieve SOTA on any benchmark (compute, data, time budgets are research-scale).
- Replace production LLMs (no RLHF, no instruction-tuning, no alignment).
- Prove IIT 4.0 is correct (IIT is a source of inspiration, not ground truth).

It **is** attempting to falsify or validate the five sub-hypotheses above via systematic ablation and to report the findings **with caveats intact** rather than spin them into victories.

---

## 2. What NeuroSLM Tries to Prove: The Three Pillars

### 2.1 Pillar 1: Integrated Information (Φ) drives generalization

**Claim:** Models trained to maximize integrated information Φ — the minimum information partition across module outputs — develop internal structure that generalizes better than models optimized only for next-token loss.

**Operationalization:**
- Train two 100M models, identical except: (A) includes Φ loss term, (B) Φ-weighted BDNF regulation off.
- Evaluate both at matched steps on WikiText-103-v1 OOD.
- Hypothesis: Model A has lower OOD perplexity and higher Φ at convergence.

**Current status:** Φ is computed correctly (test: `test_phi.py`), drives BDNF growth (test: `test_neurochem.py`), and contributes real gradient (`test_brain_forward.py`). But it does **not** appear in B0–B3 ablation results as a clear OOD lever — gap_ratio improvements come from architectural tweaks (PCT, isolation, recursion), not from Φ alone. [🟡 PARTIAL — mechanism confirmed, payoff unclear]

**Evidence link:** `tests/test_phi.py::test_phi_higher_for_coupled_outputs` [✅ CONFIRMED]

### 2.2 Pillar 2: Gradient hierarchy prevents catastrophic forgetting

**Claim:** When multiple objectives (language modeling + world model + action + consciousness + causal reasoning) compete for the shared trunk representation, their uncontrolled gradient flow corrupts the main loss. Detaching the bio pipeline from the trunk (§5.2: "Trunk Gradient Isolation") and gating module→trunk contributions via ReZero scalars (§5.3) prevents this.

**Operationalization:**
- Train a 30M brain with detach ON and OFF.
- Measure: (a) post-awakening divergence rate, (b) final-step train/OOD PPL.
- Hypothesis: detach=ON trains cleanly to 10k; detach=OFF diverges by 7k.

**Current status:** ✅ CONFIRMED. B1 (recursive) and B2.fix (ReZero) both train to step 5-7k cleanly *with* isolation. Prior runs without isolation diverged at ~5-6k. Test: `test_stabilization.py` shows trunk gradient is invariant to aux-loss weights when isolation is on.

**Evidence link:** `results/ood_recursive_108M_step5000.json`, `tests/test_stabilization.py::test_trunk_gradient_invariance` [✅ CONFIRMED]

### 2.3 Pillar 3: Biologically-grounded topology is more data-efficient

**Claim:** A brain with cortical-grade specialization, plasticity (BDNF), and homeostasis needs fewer training examples to reach the same loss as a flat transformer. Measured as: OOD gap_ratio = OOD_ppl / train_ppl. Lower gap_ratio indicates less distribution shift vulnerability.

**Operationalization:**
- Compare BRIAN best variant (B3, 69M, gap_ratio 4.51) against baseline (107M, gap_ratio 6.12) at matched OOD corpus.
- Hypothesis: BRIAN gap_ratio < baseline gap_ratio (improved generalization fingerprint despite lower parameter count).

**Current status:** ✅ gap_ratio claim holds: 4.51 (PCT-30M) < 6.12 (flat baseline). But cross-scale (69M vs 107M) and cross-step (4000 vs 80000) confounds are present. Matched-params, matched-steps variant pending.

**Evidence link:** `results/ood_pct-30m_68M_step4000.json`, `results/ood_baseline-80k_107M_step80000.json`, `findings.md::H10` [🟡 PARTIAL]

---

## 3. Architecture Overview & DSL Compilation

### 3.1 The `.neuro` DSL — Declarative Neural Architecture

NeuroSLM uses a custom **architecture description language** (`.neuro` syntax) to separate the *declaration* of connectivity, neuromodulation, and parameters from their PyTorch implementation. This approach provides:

- **Reproducibility:** The `.neuro` file is the canonical spec. Git diff on `arch.neuro` immediately shows what changed.
- **Separation of concerns:** Parameters, topology, and loss weights live in one human-readable file, not scattered across Python modules.
- **Compilation pipeline:** `neuroslm.dsl.compile_folder()` parses the `.neuro` files, builds a computation graph, and emits a `nn.Module` ready for training.

**Location & format:**
- `architectures/rcc_bowtie/arch.neuro` — entry point, declares `architecture`, `training`, `param_scope`, `neurotransmitter`, module `import`s, `synapse` (wiring), and `modulation` (neuromodulator routing).
- Each region (cortex, hippocampus, amygdala, etc.) has a subfile under `architectures/rcc_bowtie/modules/`.

**Core sections in arch.neuro (lines 20–570):**

| Section | Purpose | Example |
|---------|---------|---------|
| `architecture {}` | Global dimensionality (`d_sem=256`, timestep `dt=0.01`) | Line 20 |
| `training {}` | Hyperparameters: LR, batch, seq_len, dropout, loss clipping, optimizer, regularization | Lines 31–248 |
| `param_scope {}` | Gradient isolation rules; `trunk` gets full LM gradient, `bio` gets detached (§5.2 isolation) | Lines 258–265 |
| `neurotransmitter {}` (×7) | Dopamine, NE, 5HT, ACh, eCB, Glutamate, GABA; base conc, release/reuptake rates | Lines 271–318 |
| `import` | Pulls in module regions from disk | Lines 324–335 |
| `synapse` | Bowtie bottleneck wiring + re-entry loops | Lines 342–457 |
| `modulation` | NT → region effect tables (dopamine→PFC gain=0.6, etc.) | Lines 467–550 |
| `formal_spec` | IIT constraints (phi, bowtie narrowing) | Lines 556–569 |

**Design:** The `.neuro` file is a **declarative** constraint; the harness then operationalizes it. This lets us reason about what *should* happen (mathematically) separately from what *does* happen (in gradient flows).

### 3.2 Compilation Pipeline

```
arch.neuro (+ modules/*.neuro)
    ↓
neuroslm.dsl.multifile.compile_folder()
    ├─ parse .neuro files
    ├─ build symbol table (regions, NTs, synapses)
    ├─ resolve imports + wiring
    ├─ construct computation graph
    └─ emit nn.Module
        ↓
    Brain (nn.Module subclass)
        ├─ LanguageCortex (trunk)
        ├─ GlobalWorkspace (bottleneck)
        ├─ Orchestrator (11-stage forward pass + re-entry)
        ├─ Neurotransmitter levels
        ├─ Homeostasis gates
        └─ Plasticity (BDNF, trophic, Hebbian)
```

**Train loop integration:**
1. Load `.neuro` via `load_training_config_from_arch()` → extracts `training {}` block.
2. Compile modules via `CodeGenerator` → `nn.Module` + parameter scopes.
3. Wrap in `BRIANHarness` → applies loss clipping, grad accumulation, aux-loss ramp, maturity gates.
4. Standard PyTorch training: forward → loss → backward → optimizer.step().

---

## 4. The Training Harness (`BRIANHarness`)

The harness wraps the compiled `Brain` and implements:
- **Loss clipping** (robust per-sample clipping per §P4, line 36 arch.neuro)
- **Gradient accumulation** over micro-batches
- **Auxiliary loss scheduling** (Φ, world model, motor, narrative, etc.)
- **Maturity gates** (soft awakening via phase gates at MAT levels 0.35, 0.45, 0.55, 0.60)
- **OOD evaluation loop** (periodic WikiText-103-v1 perplexity checks)
- **Early-exit rules** ("OOD trend stalls" → save & exit)

**Key hyperparameters in arch.neuro (P4 preset, lines 35–89):**

| Param | Value | Rationale |
|-------|-------|-----------|
| `loss_clipping: factor=3.0` | Clip per-sample loss to [0, 3.0] | Prevents gradient spikes that cause divergence post-awakening. |
| `dropout: 0.12` | Embedding + per-block + pre-head | Single biggest OOD lever (0.0 → 13× OOD gap blowout). |
| `pct_trunk: 0.4, pct_strength: 0.3` | 40% of samples use top-down predictor, strength=0.3 weight | Predictive coding loss in addition to CE. |
| `stochastic_depth: 0.1` | Random block skip prob at training | Regularizes trunk, helps OOD. |
| `flooding_level: 4.0` | Prevent train loss from collapsing below floor | Maintains useful gradients, stops memorization. |
| `batch_size: 16, seq_len: 2048` | 32k tokens/step | Long-range context for WikiText prose. |
| `warmup_steps: 300, min_lr_ratio: 0.1` | Ramp to 3e-4, decay to 3e-5 | Smooth early training, long-tail learning. |

### 4.1 The Maturity-Phase Gating (Soft Awakening)

During infancy (MAT < 0.30), all auxiliary losses are off: the model is a vanilla transformer. As MAT increases, auxiliary subsystems engage via smooth sigmoid gates centered at different MAT levels:

```
MAT ∈ [0, 1]  (performance proxy: 1 - loss/L_random)
  ↓
For each aux_loss ∈ {pred_coding, world, motor, Φ, kl, cpc, ...}:
  phase_gate(MAT, center, width=0.10) = 0.5 * (1 + tanh((MAT-center)/width))
  final_loss_weight = base_weight * phase_gate(MAT, center) * aux_w_scale
  ↓
Subsystem      | Phase center | Onset logic
pred_coding    | 0.35         | Cheap internal supervision; engages earliest
world          | 0.45         | World model grounding once LM bootstraps
motor/forward  | 0.50         | Action objectives need working world model
novel_aux/cpc  | 0.55         | Contrastive objectives
kl_world/Φ     | 0.60         | Heaviest objectives; last to engage
```

**Why soft gates matter:**
- Pre-§0.10: single on/off switch at MAT=0.30 caused catastrophic awakening collapse (all 8+ objectives slammed on simultaneously).
- Post-§0.10: per-system phase windows distribute the subsystem load, preventing gnorm spike.

**Implementation:** `Brain._phase_gate(mat, center, width=0.10)` in forward pass, multiplies each aux-loss term.

---

## 5. Current Model State: 30M P4 Preset

The active baseline is the **30M P4 (Price 4, loss-clipping experiment)** preset, defined in `arch.neuro` lines 163–172:

```python
30m_p4: {
    d_model:       512      # embedding + trunk hidden dim
    depth:         8        # # of transformer blocks
    n_heads:       8        # attention heads
    max_ctx:       2048     # context window
    batch_size:    16       # per-GPU micro-batch
    seq_len:       2048     # training sequence length
    grad_accum:    1        # gradient accumulation steps
    approx_params: "30M"    # informational
}
```

**Training command:**
```bash
python -m neuroslm.train_dsl \
    --arch architectures/rcc_bowtie \
    --scale 30m_p4 \
    --steps 40000 \
    --device cuda:0
```

**Current results (from findings.md::B3, committed 2026-05-25):**
- **Train PPL:** 400.9 @ step 4000 (best checkpoint)
- **OOD PPL (WikiText-103-v1):** 1806.6
- **gap_ratio:** 4.51 ← **best achieved to date**
- **Status:** Reaches step 10k cleanly; training stable.

**Checkpoint:** `lfs_checkpoints/neuroslm_pct_30m_68M_adamw_mix_best.pt`

---

## 6. Evaluation Harness & Evidence Standards

### 6.1 Layer A — Unit Tests (Mechanism Confirmation)

15 tests in `tests/` verify that individual modules compute correctly. Each test constructs a `Brain`, calls a method, and asserts the output matches spec.

**All passing as of 2026-06-01:** ✅

| # | Test | Spec section | Status |
|---|------|--------------|--------|
| H1 | `test_phi.py::test_phi_higher_for_coupled_outputs` | §3.3 MIP algorithm | ✅ |
| H2 | `test_brain_forward.py::test_phi_objective_increases_total_gradient` | §2.2 Φ injects gradient | ✅ |
| H3 | `test_neurochem.py::test_trophic_phi_boosts_growth` | §6.2 BDNF scaling | ✅ |
| H4 | `test_narrative_memory.py::test_sheaf_contradiction_detection` | §10.1 H¹ cocycles | ✅ |
| H5 | `test_narrative_memory.py::test_causal_generalization` | §10.2–3 causal rules | ✅ |
| H6 | `test_cognitive_closure.py::test_autobiographical_personality_consistency` | §10.5 identity persistence | ✅ |
| H6.5 | `test_cognitive_closure.py::test_survival_imperative_qualia_shift` | §11 embodied survival | ✅ |

**Run all:** `py -3 -m pytest tests/ -v` (~7s on CPU)

### 6.2 Layer B — Architecture Evaluation (OOD Perplexity)

A trained checkpoint is evaluated on a held-out OOD corpus (WikiText-103-v1) using a sliding-window eval:

```
Training corpus: FineWeb-Edu + OpenHermes (mixed, ~409k tokens @ batch 16 × seq_len 2048)
OOD corpus:      WikiText-103-v1 (academic prose, ~103k tokens)
Eval metric:     Perplexity via cross-entropy loss on token prediction
Eval script:     scripts/vast_ood_eval.sh → brian_ood_test.py
Harness:         GPT-2 BPE tokenizer (vocab 50257)
```

**Ablation table (from findings.md):**

| Variant | Params | Train PPL | OOD PPL | gap_ratio | Commits | Verdict |
|---------|--------|-----------|---------|-----------|---------|---------|
| B0 (baseline) | 106.9M | 66.0 | 404.0 | 6.12 | `stabilize/trunk-grad-isolation` @ 80k | ✅ Reference |
| B1 (recursive) | 108.2M | 216.5 | 1372.8 | 6.34 | `stabilize/recursive-reasoning` @ 5k | 🟡 In-dist win, no OOD |
| B2.fix (ReZero) | 107.8M | 258.8 | 1351.5 | 5.22 | `stabilize/trunk-grad-isolation` @ 7k | 🟡 -17% gap but underdone |
| B3 (PCT-30M) | 69.2M | 400.9 | 1806.6 | **4.51** | `arch/predictive-coding-trunk` @ 4k | 🟡 Best ratio, cross-scale caveat |

**Key caveat:** B3 is 69M (30% smaller) and step 4000 (not fully converged). Matched-params B3 variant pending.

---

## 7. The Five Core Architectural Fixes (§5 of architecture.md)

### 7.1 Trunk Gradient Isolation (§5.2)

**Problem:** Bio modules (world model, motor, consciousness, narrative) were attached to the shared trunk, so their random-init losses sent large, uncorrelated gradients into the representation the LM head depends on. At awakening (MAT > 0.30), all 8+ objectives suddenly engaged, causing a gnorm spike → divergence.

**Solution:** `sem = sem.detach()` before feeding to bio pipeline. The trunk gradient now comes *only* from the LM loss. Bio losses shape *their own* modules, not the trunk.

**Evidence:** `results/ood_recursive_108M_step5000.json` (trains cleanly to 5k), `tests/test_stabilization.py` (trunk gradient invariant to aux weights). [✅ CONFIRMED]

### 7.2 ReZero Learnable Scalar Gates (§5.3)

**Problem:** Module outputs (motor, memory, thought) were gated onto the trunk via maturity phase-gates (smooth sigmoid ramp at MAT=0.5). This caused a discontinuity at awakening: the LM head, trained on `h`, suddenly saw `h + 0.5*motor_bias`, a different function → PPL jump (~90 → 370).

**Solution:** Replace phase-gates with zero-init learnable scalars λ. At init, all λ=0 → modules are identities. LM gradient self-discovers the blend.

**Evidence:** gap_ratio 5.22 (ReZero) vs 6.34 (plain recursive), `tests/test_stabilization.py`. [🟡 PARTIAL — removes discontinuity, no OOD win]

### 7.3 Recursive Reasoning Loops (§5.4)

**Problem:** Token-level experts (MathCortex, ReasoningCortex) only run once per token. Some tasks (ARC, multi-step) need iterative refinement.

**Solution:** Loop the expert blocks `N=4` times with weight-sharing (Universal Transformer style). Effective depth becomes `4 × 3 = 12` layers of reasoning, zero added parameters.

**Evidence:** train PPL 216.5 (recursive) < 258.8 (ReZero), but OOD tied. [🟡 PARTIAL — in-dist win only]

### 7.4 Predictive Coding Trunk (§5.5)

**Problem:** ERM on a fixed mix structurally rewards shortcuts. Every param is pulled toward minimizing next-token loss; no signal about what representations *should* look like. Result: 4–6× train→OOD gap.

**Solution:** Add top-down generative predictors `g_n: h_{n+1} → ĥ_n` (direction flip from standard deep supervision). Deeper layers forced to be invertible → forces compositionality.

**Evidence:** gap_ratio 4.51 (PCT) vs 6.12 (baseline), but at 69M, step 4k (underparameterized & undertrained). [🟡 PARTIAL — directional improvement, caveats present]

### 7.5 Phased Maturation & Auto-Recovery (§0.10–0.17)

**Problem:** Soft awakening (per-subsystem phase gates) fixed the gnorm spike, but trophic pruning (structural capacity removal) was still triggering at peak performance, and GWS ignition gate was over-firing (saturated).

**Solutions:**
1. **Phased maturation gates** (phase centers at MAT 0.35, 0.45, 0.55, 0.60) distribute subsystem startup.
2. **Trophic auto-recovery** — reactivate pruned projections when active fraction drops below 60%.
3. **Adaptive GWS ignition threshold** — tracks per-slot activity EMA instead of fixed threshold.

**Evidence:** Run 38469631 (P4) completes to step 10k with stable loss trajectory. [✅ CONFIRMED operationally]

---

## 8. DSL Language & Module Structure

### 8.1 Module Declaration (Example: thalamus.neuro)

Modules are `.neuro` files defining **populations** (pools of neurons), **local wiring** (within-module connections), and output projections:

```neuro
population sensory {
    type: "TextSensoryCortex",
    d_out: 256,
    has_plasticity: true
}

population thalamus {
    type: "ThalamicHub",
    d_out: 256,
    streams: ["language", "math", "reasoning", "spatial", "social"]
}

internal_wiring thalamus {
    # Thalamus → GWS (via synapses declared in arch.neuro)
    # Local re-entry: thalamus ← GWS feedback (next step)
}
```

### 8.2 Synapse Definition

Synapses declare **pre→post connectivity**, **neurotransmitter**, and **equation**:

```neuro
synapse thalamus -> gws {
    weight: 0.8,
    neurotransmitter: "glutamate",
    equation: "y = weight * (x_pre @ W)"
}
```

This is **declarative:** the harness generates the actual PyTorch projection layers.

### 8.3 Neuromodulation

Each synapse's output can be scaled by neurotransmitter concentration:

```neuro
modulation dopamine -> pfc {
    effect: "multiplicative",
    gain: 0.6,
    equation: "y = output * (c * gain)"
}
```

Concentrations are computed in `TransmitterSystem`, updated each step based on aux-loss signals (Φ high → dopamine release, etc.).

---

## 9. Key Mechanisms & Their Evidence

### 9.1 Consciousness (Integrated Information Φ)

**Mechanism:** Compute Gaussian mutual information MIP over module outputs, differentiable w.r.t. parameters.

**Implementation:** `neuroslm/modules/consciousness.py :: _compute_phi_mip`
- For ≤8 modules: exhaustive 2^(n-1) bipartition enumeration
- For >8: spectral bisection via Laplacian Fiedler vector

**Training signal:** `L_Φ = -tanh(Φ/8) · 8` (bounded, pushes Φ higher)

**Evidence:** `tests/test_phi.py`, `tests/test_brain_forward.py::test_phi_objective_increases_total_gradient` [✅ CONFIRMED]

**Caveat:** Φ does not appear as a dominant OOD lever in B0–B3 ablations. May be important for *causal structure* (tested in Layer A) rather than generalization directly.

### 9.2 Plasticity (BDNF + Trophic Growth)

**Mechanism:** High-Φ attractors get more trophic support → NeuralGeometryAdapter rank grows → integrated pathways deepen.

**Implementation:** `neuroslm/neurochem/growth.py :: TrophicSystem`
- Trophic factor released proportional to Φ
- Rank increase gated by maturity + Fiedler boost (graph connectivity)
- Auto-recovery: reactivate pruned projections below 60% active fraction

**Evidence:** `tests/test_neurochem.py::test_trophic_phi_boosts_growth` [✅ CONFIRMED]

### 9.3 Homeostasis & Neuromodulation

**Mechanism:** Seven neurotransmitter systems (dopamine, NE, 5HT, ACh, eCB, Glu, GABA) maintain steady-state activity targets via feedback control.

**Implementation:** `neuroslm/neurochem/transmitters.py :: TransmitterSystem`
- Each region has NT receptor gains
- NT levels updated based on activity: `c_new = c * τ_decay + baseline * (1 - τ_decay) + release`
- Clamping + saturation scavenging (fast reuptake) prevent saturation

**Evidence:** Implicit in successful training; no dedicated test yet.

### 9.4 Narrative Memory & Causal Inference

**Mechanism:** Store experiences as `(subject, predicate, object, context)` tuples. Detect contradictions via sheaf H¹ cohomology. Learn causal rules: `Gift → Joy` observed twice → generalizes to unseen Gift trials.

**Implementation:** `neuroslm/intelligence/narrative_store.py`, `neuroslm/intelligence/causal_head.py`

**Evidence:** `tests/test_narrative_memory.py::test_sheaf_contradiction_detection`, `::test_causal_generalization` [✅ CONFIRMED]

### 9.5 Survival Loop & Embodied Homeostasis

**Mechanism:** Energy state perturbation → qualia warp → basal ganglia policy shift via dopamine+RPE updates.

**Implementation:** GridWorld environment, VAE sensory encoder, homeostatic targets.

**Evidence:** `tests/test_cognitive_closure.py::test_survival_imperative_qualia_shift` [✅ CONFIRMED]

---

## 10. Training Dynamics & Failure Modes

### 10.1 Post-Awakening Collapse (§5.2 — Fixed)

**Symptom:** Clean training until MAT ≈ 0.30, then:
- Gradient norm spikes to ~14 (vs steady-state ~1.5)
- Train loss jumps: 4.1 → 6.0
- PPL stalls at ~250, oscillates, diverges by step 7-8k

**Root cause:** Aux losses inject gradient into the shared trunk the LM head depends on. At awakening, 8+ random-init objectives fire simultaneously, causing gradient collision.

**Fix (§5.2):** Detach trunk before bio pipeline. **Result:** Trains cleanly to 10k+.

**Commits:** `cce9be0` (recursive fix), `2dd893b` (trunk isolation enabled by default)

### 10.2 Awakening Discontinuity (§5.3 — Partially Fixed)

**Symptom:** PPL smooth until MAT ≈ 0.45, then:
- Motor/memory gates suddenly engage (maturity phase-gate switches on)
- LM head was trained on `h`, now sees `h + 0.5*motor_bias`
- PPL jump: ~90 → ~370

**Root cause:** Maturity phase-gates are all-or-nothing around their center. A small MAT shift causes a large coefficient shift.

**Fix (§5.3):** ReZero zero-init λ scalars. LM gradient self-discovers the blend. **Result:** Discontinuity removed; modest gap_ratio improvement (5.22 vs 6.34).

**Commits:** `5ff02c8` (ReZero gates default ON)

### 10.3 OOD Gap Still High (§5.5 PCT — In Progress)

**Symptom:** All B0–B3 variants show gap_ratio ≥ 4.5 (benchmark: <2.0 for no overfitting).

**Root cause:** ERM on FineWeb-Edu + OpenHermes mix — the model learns dataset-specific shortcuts.

**Partial fix (§5.5):** Predictive coding trunk. Top-down predictors force compositionality. **Result:** gap_ratio 4.51 (PCT) vs 6.12 (baseline), but caveats: smaller (69M), earlier best (step 4k), undertrained.

**Commits:** `arch/predictive-coding-trunk` branch

**What's needed next:** Same-params PCT variant at step 7000+, to answer whether the improvement is real or artifact of scale/training time.

---

## 11. Current Results Summary & Artifact Links

### 11.1 Reference Table (All committed to `results/`)

| Variant | Scale | Params | Steps | Train PPL | OOD PPL | gap_ratio | Verdict | Artifact |
|---------|-------|--------|-------|-----------|---------|-----------|---------|----------|
| Baseline | 107M | 106.9M | 80k | 66.0 | 404.0 | 6.12 | ✅ strong baseline | `ood_baseline-80k_107M_step80000.json` |
| Recursive | 108M | 108.2M | 5k | 216.5 | 1372.8 | 6.34 | 🟡 in-dist win | `ood_recursive_108M_step5000.json` |
| ReZero | 108M | 107.8M | 7k | 258.8 | 1351.5 | 5.22 | 🟡 modest gap improvement | `ood_rezero-fixed_107M_step7000.json` |
| **PCT-30M** | **30M** | **69.2M** | **4k** | **400.9** | **1806.6** | **4.51** | 🟡 best ratio (caveat: cross-scale) | `ood_pct-30m_68M_step4000.json` |

### 11.2 Unit Test Status

All 15 Layer-A tests passing:
```bash
py -3 -m pytest tests/test_phi.py tests/test_brain_forward.py \
                tests/test_neurochem.py tests/test_narrative_memory.py \
                tests/test_cognitive_closure.py tests/test_pct_smoke.py -v
```

**Output:** `15 passed in 7.23s` ✅

### 11.3 Recent Training Runs

**Run 38469631 (2026-06-01):**
- Architecture: RCC BoWTie P4 (30M)
- Completed: 10,000 steps
- Final PPL: 242.1
- Status: ✅ Stable, no divergence
- Checkpoint: `lfs_checkpoints/neuroslm_rcc_bowtie_30m_p4_step10000.pt`

---

## 12. What's Proven, What's Pending, What's Broken

### 12.1 Confirmed (✅)

- Φ (integrated information) computes correctly and injects gradient.
- Trunk gradient isolation prevents post-awakening collapse.
- Recursive reasoning adds depth at zero parameters.
- Phased maturation enables stable post-awakening training.
- BDNF plasticity couples to Φ and reshapes the connectivity graph.
- Narrative memory detects contradictions and learns causal rules.
- The system can train cleanly to 10k+ steps at 30M scale.

### 12.2 Partially Proven (🟡)

- **gap_ratio improvement:** BRIAN best (4.51) < baseline (6.12), but at different scales and steps.
- **ReZero gates:** Remove PPL discontinuity at awakening, modest gap win, no absolute-OOD win.
- **PCT:** Lowers gap_ratio by 26%, but target was 2× (not met). Underdone on cross-scale basis.

### 12.3 Pending (🟠)

- Same-params PCT eval (30M to step 7k+) vs 30M ReZero baseline.
- Baseline at step 7k (matched-compute comparison for H12).
- Full SRC-TEH wall-clock benchmarks (H11).
- ARC / reasoning benchmarks (non-PPL eval).

### 12.4 Falsified (❌)

- "BRIAN measurably better at matched FLOPs than flat 1B baseline" (H12) — current snapshot: baseline wins by 3-4× absolute PPL, but asymmetric compute (11×). Inconclusive.

### 12.5 Unverified Overclaims (⚠)

- README line 276: "measurably better at matched FLOPs than a flat 230M dense transformer" — backed by H12 verdict above (inconclusive).

---

## 13. How to Extend This Report

This file should be kept synchronized with the codebase via a `brian ai document` command (see CONTRIBUTING.md). Key rules:

1. **Every claim here must cite evidence:** A test name, a result JSON, or a code line.
2. **Update in the same change set:** Architecture change → update this file + `architecture.md` + commit together.
3. **Archive old findings:** Stale results (OOD_PUSH_STAGES.md, old session summaries) → `docs/archive/YYYY-MM-DD_*.md`.
4. **Distinguish active from historical:** Current state (sections 4–11) vs completed experiments (findings.md).

---

## 14. For External AIs (NotebookLM, Perplexity, etc.)

This report is designed to be self-contained. You have:
- **Project charter** (§1): what we're trying to prove
- **Architecture spec** (§3–8): how it works (with references to `docs/architecture.md` for details)
- **Evidence** (§6–11): Layer A (mechanisms) + Layer B (OOD performance)
- **Caveats** (§12): what's proven, what's not, what's pending
- **Reproducibility** (findings.md): exact commands to re-run each result

**To stay current:** Check for a newer version in the git history. This file is regenerated periodically via `brian ai document` (see CONTRIBUTING.md).

---

## References & Artifacts

- **Architecture.md** — Detailed module specs, DSL semantics, IIT foundations
- **findings.md** — Hypothesis ledger + Layer A/B evidence + reproducibility recipes
- **changelog.md** — Commit-derived changelog (auto-maintained)
- **arch.neuro** — Canonical DSL source (training config, hyperparameters, topology)
- **results/\*.json** — OOD evaluation results, all archival
- **tests/\*.py** — Layer A mechanism tests
- **logs/vast/\*.log** + **logs/analyzed/\*.md** — Raw vast.ai training logs + LLM analysis

---

**Last verified:** 2026-06-01  
**Next review:** After matched-compute baseline eval or new OOD results commit  
**Maintainer:** User instructions in CLAUDE.md §documentation-sync
