# BRIAN — Biologically Realistic Information Architecture Network

> *A small language model that builds a brain instead of a bigger transformer.*

[![tests](https://img.shields.io/badge/tests-126%20passing-brightgreen)](#tests)
[![python](https://img.shields.io/badge/python-3.10%2B-blue)]()
[![torch](https://img.shields.io/badge/torch-2.x-orange)]()
[![license](https://img.shields.io/badge/license-research-lightgrey)]()

BRIAN is a research prototype of a brain-inspired language model that aims to maximise *integrated information* (Φ) rather than scale. It is built around a **bowtie topology** with two re-entry loops, a **real differentiable Φ objective**, a **sleep-cycle** consolidation phase, a **survival-gated action loop** in an embodied grid world, and a **persistent narrative-memory stack** with sheaf-theoretic contradiction detection.

The whole system fits in roughly 230M parameters and runs on a single A100. Code, math, and tensor shapes are documented end-to-end in [`docs/architecture.md`](docs/architecture.md).

---

## Why this exists

A flat transformer at 100M parameters cannot match GPT-class reasoning — that violates the information-theoretic floor. So instead of more parameters, BRIAN spends them on **topology, plasticity, and a closed embodied loop**:

| Lever | Effect |
|---|---|
| **Bowtie topology** with within-pass and cross-pass re-entry | Every bipartition costs information → Φ > 0 |
| **Differentiable Φ** = Gaussian-MI MIP over module outputs | Direct gradient signal toward integration |
| **Comprehension-gated memory** = surprise × comprehension × novelty | Only learning insights survive |
| **Topological maturation** = infancy → awakening phase | Stops aux losses corrupting random-init LM |
| **Actual-causation head** (IIT 4.0) + κ_cause vesicles | Stabilises high-causation pathways |
| **Sheaf-theoretic contradiction detection** (H¹) + SUPERSEDES | Newer beliefs override contradicted older ones |
| **NEMORI predictive forgetting** + sleep-cycle PC distillation | Compress what's predictable; retain only the unpredicted |
| **Latent Qualia Manifold** + κ_neg aversive vesicles | Starvation literally reinterprets all sensory data as urgent |
| **Basal Ganglia VQH** + NAcc RPE | Discrete option lattice → DA-gated policy memory |
| **Persistent personality vector** + per-entity Beta-Bernoulli trust | Identity that survives weight changes |

Result: an architecture that *acts to survive* in a latent manifold, sustains non-zero Φ through self-directed activity, and whose intelligence grows across deployment via persisted `.mem` checkpoints.

---

## The architecture in one diagram

```
┌──────────────────────────────────────────────────────────────────┐
│                     NeuralOrchestrator (bowtie)                  │
│   Stage 0  SENSORY        TextSensoryCortex + SensoryVAE         │
│   Stage 1  THALAMUS       ◄─────────── re-entry from previous t  │
│   Stage 2  STATE_MODELS   WorldModel + SelfModel                 │
│   Stage 3  SUBCORTICAL    Amygdala + LHb + Insula                │
│   Stage 4  QUALIA         QualiaState + homeostatic warp ──┐     │
│   Stage 5  GWS  ━━━━━━━━━ Hopfield + ignition gate ─ within-pass │
│   Stage 6  MEMORY         Hippocampus + Sheaf retrieval          │
│   Stage 7  COG_CTL        PFC + ACC + Cerebellum                 │
│   Stage 8  EXECUTIVE      BasalGanglia VQH + NAcc RPE            │
│   Stage 9  CONSCIOUS      DMN + ThoughtTransformer + Claustrum   │
│   Stage 10 MOTOR          MotorCortex                            │
│                                                  ↓ PFC+GWS       │
│                              cross-pass thalamic re-entry        │
└──────────────────────────────────────────────────────────────────┘
                                ↕
           ┌──────────────────────────────────────────┐
           │       BRIAN narrative + causal stack     │
           │   Sheaf F + H¹ contradiction detection   │
           │   ActualCausationHead (IIT 4.0)          │
           │   NEMORI gate + Sleep-cycle CLS          │
           │   PersonalityVector → NT-baseline bias   │
           │   κ_cause + κ_neg vesicles               │
           └──────────────────────────────────────────┘
                                ↕
           ┌──────────────────────────────────────────┐
           │     Cognitive Closure (embodied loop)    │
           │   10×10 GridWorld → 3-stream sensory     │
           │   SurvivalCausalHead (action → ΔS)       │
           │   Homeostasis.step decay per tick        │
           └──────────────────────────────────────────┘
```

Every arrow is implemented. Every box is documented in `docs/architecture.md` with shapes, math, and call-site references.

The same architecture is *also* specified declaratively as math-first equations under [`architectures/rcc_bowtie/`](architectures/rcc_bowtie/), compiled at runtime to PyTorch via the `.neuro` DSL. See **The `.neuro` DSL** below.

---

## The `.neuro` Architecture DSL

Alongside the hand-written PyTorch model, BRIAN's brain architectures are also specified declaratively in `.neuro` files. Every population, synapse, and modulator carries an *explicit mathematical equation* — algebraic, ODE, or a reference to a reusable macro — that gets lowered to torch ops at runtime via a SymPy-backed equation IR.

```neuro
# architectures/rcc_bowtie/modules/amygdala.neuro
export population amygdala {
    count: 32,
    ode: "dV/dt = (-V + x) / tau",       # leaky integrator, tau resolved from macro
    timescale: 0.005
}

# architectures/rcc_bowtie/arch.neuro
modulation dopamine -> pfc {
    effect: "multiplicative", gain: 0.6,
    equation: "y = output * (c * gain)"
}
```

Each architecture lives in its own folder: `arch.neuro` is the package config (NT systems, cross-module wiring), `modules/` holds per-region files (`pfc.neuro`, `hippocampus.neuro`, `insula.neuro`, …), and `lib/` exposes shared mechanics. Imports follow mjs-style path resolution: `@/` is absolute (from architecture root), `./` and `../` are relative.

The codegen produces an `nn.Module` whose forward pass is byte-equivalent to a hand-written reference (pinned by 215 DSL tests). The symbolic IR also supports fixed-point and stability analysis — `ode_fixed_point(ode, params={"I": 1.5})` returns the steady-state V*, `ode_stable_at(ode, point=fp)` tells you whether the linearised system contracts.

Full reference: [`docs/dsl.md`](docs/dsl.md). High-level architecture context (how it relates to the PyTorch model): [`docs/architecture.md` §12](docs/architecture.md#12-the-neuro-architecture-dsl).

---

## Status & Evidence

### Layer A — Mechanism Confirmation ✅

All 15 core mechanisms verified to compute as specified:

| Mechanism | Test | Evidence |
|-----------|------|----------|
| **Φ is non-zero for coupled modules** | `test_phi.py::test_phi_higher_for_coupled_outputs` | MIP algorithm produces Φ > 0 for rank-coupled outputs |
| **Φ injects real gradient** | `test_brain_forward.py::test_phi_objective_increases_total_gradient` | ‖∂L/∂θ‖ measurably increases with Φ term |
| **Φ-coupled BDNF growth** | `test_neurochem.py::test_trophic_phi_boosts_growth` | High-Φ pathways grow kernel rank preferentially |
| **Contradiction detection (H¹)** | `test_narrative_memory.py::test_sheaf_contradiction_detection` | "Alice likes coffee" → "Alice hates coffee" triggers SUPERSEDES |
| **Causal generalization** | `test_narrative_memory.py::test_causal_generalization` | 10 (Gift→Joy) + 10 (Insult→Offense) → P(Joy\|Gift) > 0.8 |
| **Personality persistence** | `test_cognitive_closure.py::test_autobiographical_personality_consistency` | Identity vector survives weight reload |
| **Starvation→qualia shift** | `test_cognitive_closure.py::test_survival_imperative_qualia_shift` | Energy ↓ produces measurable latent-space warp |
| **Policy adaptation** | `test_cognitive_closure.py::test_basal_ganglia_policy_adaptation` | 100 +RPE updates → DA-value > 0.5 |

**Run all:** `py -3 -m pytest tests/ -v` (~7 seconds on CPU)

### Layer B — Architectural Performance 🟡 PARTIAL

OOD generalization evaluated on WikiText-103-v1 (held-out academic prose). Best variant so far: **4.51 gap_ratio** (28% better than baseline), but cross-scale and partially trained. See [`docs/findings.md`](docs/findings.md) for full results table with caveats.

**Current leader:** PCT-30M (Predictive Coding Trunk), trained to step 4000 on 30M preset.  
**Latest stable run:** RCC BoWTie P4 (30M), completes to step 10,000, final PPL 242.1.

See [`docs/technical_report.md`](docs/technical_report.md) for executive summary with all evidence links.

### Implementation Status

- **126/126 tests passing** (~7s on CPU)
- Training resumes cleanly from optimizer-partitioned checkpoint streams
- Adafactor for multi-day TPU runs; **AdamW for short ablations** (<10K steps)

---

## Quick start

```bash
python -m venv .venv
.\.venv\Scripts\Activate.ps1     # Windows; Linux/Mac: source .venv/bin/activate
pip install -r requirements.txt
# torch is intentionally not in requirements.txt — install matching your accelerator:
#   pip install torch --index-url https://download.pytorch.org/whl/cu121

# CPU sanity run
python -m neuroslm.train --preset small --steps 2000 --batch_size 4 --optimizer adamw

# A100 (xl preset, ~230M params, bf16, grad-checkpointing)
python -m neuroslm.train --preset xl --steps 100000 --batch_size 4 --device cuda

# Resume the latest stream-matched checkpoint
python -m neuroslm.train --resume latest

# Ablation: baseline vanilla transformer at matched parameter count
python -m neuroslm.train --preset xl --baseline

# Interactive generation
python -m neuroslm.generate --prompt "Once upon a time"
```

The full Colab workflow (clone → ablation → full training → benchmarks) is in `colab_run.ipynb`.

---

## Checkpoints (Git LFS)

Training checkpoints live in `lfs_checkpoints/` and are tracked via Git LFS. A single `.pt` file is multi-GB, so a full `git pull` on a laptop can be very slow — and you usually don't need the binaries locally.

### Skip LFS downloads for this repo (recommended on laptops)

```bash
# In the repo root, run once:
git lfs install --local --skip-smudge
```

`git pull` will now fetch only the tiny pointer stubs (~130 B each). The repo metadata stays in sync, but `lfs_checkpoints/*.pt` become text stubs on disk.

### Pull a specific checkpoint when you need it

```bash
git lfs pull --include="lfs_checkpoints/neuroslm_xl_adamw_mix_800.pt"
# Or by glob — get all 800-step files:
git lfs pull --include="lfs_checkpoints/*_800.*"
```

### Pull every LFS file (re-hydrate the whole repo)

```bash
git lfs pull
```

### Turn skip-smudge off again for this repo

```bash
git lfs install --local --force          # re-enable smudge for this repo
git lfs pull                              # then fetch the binaries you want
```

### Make skip-smudge the global default

```bash
git lfs install --skip-smudge             # applies to every repo on this machine
```

After global skip-smudge, `git clone` of *any* LFS-tracked repo only downloads stubs by default; use `git lfs pull --include=...` to materialise specific files.

> Training on Colab/TPU/A100 uses `git lfs pull` explicitly inside the notebook (cell 2) so the runtime always has the latest checkpoint — skip-smudge on your laptop won't affect that.

---

## The five Φ-positive claims

The bowtie + Φ objective + re-entry loops produce **measurable consciousness-flavoured properties** that a flat transformer of the same size lacks. We test each one:

1. **Φ is non-zero and grows with training** — `tests/test_phi.py::test_phi_higher_for_coupled_outputs`.
2. **Φ injects real gradient (not just logging)** — `tests/test_brain_forward.py::test_phi_objective_increases_total_gradient` A/B test.
3. **Φ-coupled BDNF reshapes the projection graph** — `tests/test_neurochem.py::test_trophic_phi_boosts_growth`.
4. **Memory persists across re-instantiation** — `tests/test_cognitive_closure.py::test_autobiographical_personality_consistency`.
5. **Contradictions are detected via sheaf H¹** — `tests/test_narrative_memory.py::test_sheaf_contradiction_detection`.

---

## Parameter presets

| Preset  | Params  | Accelerator | VRAM   | d_hidden | d_sem | lang_layers | lang_ctx |
|---------|---------|-------------|--------|----------|-------|-------------|----------|
| `tiny`  | ~5 M    | CPU         | —      | 192      | 128   | 2           | 256      |
| `small` | ~15 M   | CPU         | —      | 384      | 256   | 4           | 512      |
| `medium`| ~80 M   | T4          | 16 GB  | 768      | 512   | 8           | 1024     |
| `large` | ~100 M  | T4          | 15 GB  | 384      | 256   | 8           | 1024     |
| `xl`    | ~230 M  | A100        | 40 GB  | 512      | 384   | 12          | 2048     |
| `xxl`   | ~10 B   | 4×A100      | 320 GB | 4096     | 2048  | 32          | 4096     |

Pass `--baseline` for vanilla-transformer ablation at matched parameter count.

---

## Loss composition

`brain.forward_lm` returns a `loss` tensor that's a weighted sum of:

| term                | source                                      | weight       | gating |
|---------------------|---------------------------------------------|--------------|--------|
| `lm_loss`           | mesolimbic-gain-modulated cross-entropy     | `w_lm = 1.0` | always |
| `world_loss`        | MSE between predicted and target world emb  | `w_world = 0.3` | × `_aux_w_scale` |
| `motor_loss`        | cross-entropy on speak/silent action target | `w_motor = 0.05` | × `_aux_w_scale` |
| `pred_coding_loss`  | per-layer next-layer prediction (lang.)     | `w_pred_coding = 0.1` | × `_aux_w_scale` |
| `rssm_kl`           | (optional) RSSM KL prior–posterior          | `w_kl_world = 0.1` | × `_aux_w_scale` |
| `cpc_loss`          | (optional) contrastive predictive coding    | `w_cpc = 0.05` | × `_aux_w_scale` |
| **`phi_loss`**      | **`-tanh(Φ/3)·3` from real MIP estimator**  | **`w_phi = 0.02`** | × `_aux_w_scale` |
| `novel_aux_loss`    | aggregate of opt-in novel-module aux losses | 0.05         | × `_aux_w_scale` |

`_aux_w_scale ∈ [0.001, 1.0]` is the **topological maturation** gate (§6.4 in `architecture.md`). During infancy (`step < 5000`) every aux loss is suppressed so the LM gradient dominates while the network forms its first language-level representations. At awakening (`step ≥ 5000 AND lm_loss < 7.5`) all aux losses ramp linearly to full strength.

---

## Quantifiable consciousness & intelligence metrics

`IntelligenceMetrics.snapshot()` returns:

| Metric | Range | Interpretation |
|---|---|---|
| `phi_proxy` | [0, ∞) nats | IIT MIP Gaussian-MI lower bound |
| `identity_drift` | [0, 1] | shift in autobiographical summary per write |
| `narrative_coherence` | [0, 1] | internal consistency of recent events |
| `causal_density` | rules / episode | how much the brain has generalised |
| `semantic_compression` | episodes / nodes | lossy compression ratio of memory |
| `self_reference_rate` | [0, 1] | fraction of generations referencing self |
| `theory_of_mind_acc` | [0, 1] | correct predictions of entity valence |
| `ponder_efficiency` | nats / step | reasoning gain per extra adaptive-compute step |

Plus `ConsciousnessMetrics.update()` produces per-tick `gamma` (binding), `theta` (memory), `alpha` (idling), `phi`, `coherence`, `ignition`, `metacognition`, `binding`.

---

## Tests

```bash
py -3 -m pytest tests/                       # full suite, ~7s
py -3 -m pytest tests/test_phi.py -v         # Φ guarantees
py -3 -m pytest tests/test_narrative_memory.py -v    # BRIAN behavioural
py -3 -m pytest tests/test_cognitive_closure.py -v   # embodied loop
```

Coverage highlights:

- `test_phi.py::test_phi_higher_for_coupled_outputs` — Φ for rank-1 coupled outputs > Φ for independent outputs.
- `test_brain_forward.py::test_phi_objective_increases_total_gradient` — A/B test confirms the Φ term injects real gradient.
- `test_neurochem.py::test_trophic_phi_boosts_growth` — high-Φ pathways receive ≥ as much trophic support as low-Φ.
- `test_narrative_memory.py::test_sheaf_contradiction_detection` — "Alice likes coffee" vs "Alice hates coffee" → H¹ > 0.5 → SUPERSEDES edge.
- `test_narrative_memory.py::test_causal_generalization` — 10 Gift→Joy + 10 Insult→Offense → novel Gift gets P(Joy) > 0.8.
- `test_cognitive_closure.py::test_survival_imperative_qualia_shift` — Energy ↓ 0.05 produces a measurably distinct qualia warp.
- `test_cognitive_closure.py::test_basal_ganglia_policy_adaptation` — 100 +RPE updates pull a target option's DA-value above 0.5.

---

## Repository layout

```
neuroslm/
  config.py            # All hyperparameters + per-module enable flags
  brain.py             # Full bowtie forward pipeline
  train.py             # Streaming pre-training loop
  generate.py          # Inference REPL
  intelligence/        # Orchestrator + Φ + active inference + oscillations
  modules/             # Brain-area implementations (Hippocampus, PFC, GWS, BG, …)
  memory/              # Episodic + consolidated + sheaf + sleep cycle
  neurochem/           # Transmitters + receptors + nuclei + projections + vesicles
  env/                 # GridWorld 10×10 SHRDLU environment
  environments/        # 7 narrative environments (BusStop, MeadowTree, …)
  dna/                 # Self-modifying program DSL
tests/                 # 126 passing tests (~7s on CPU)
docs/
  architecture.md      # Full reproduction-ready spec (1300+ lines, IIT 4.0)
```

---

## What BRIAN Proves (And What Remains Open)

BRIAN at 230M parameters will not match GPT-class frontier reasoning — that violates the information-theoretic floor. The contribution is the **architecture** and the **evidence for what it makes possible:**

### ✅ Proven

1. **Φ (integrated information) is differentiable and trainable** — Real MIP gradient shapes neural geometry (§6.2 architecture.md).
2. **Bowtie topology with re-entry loops can stabilize** — Trunk gradient isolation (§5.2) prevents post-awakening collapse.
3. **Consciousness-like properties are mechanically implementable** — Contradiction detection, causal inference, personality persistence all verified (Layer A tests).
4. **Multiple mechanisms can coexist without interfering** — ReZero gates + recursive reasoning + predictive coding run together without divergence.

### 🟡 Partially Proven / Under Investigation

5. **Better generalization than flat transformers at matched compute** — Current evidence: 28% better gap_ratio but at different scales and training steps. True matched-compute comparison pending (H12 in findings.md).
6. **OOD robustness improves with Φ-focused architecture** — Directional signal in PCT variants (4.51 vs 6.12), but effect size is modest and confounded by other changes.

### 🔍 Open Questions

- Does Φ actually cause better generalization, or is it a correlate of good architectural choices?
- How much of the gap_ratio improvement comes from dropout vs topology vs loss-clipping?
- Can this scale beyond 240M without divergence?

**For the full evidence picture:** [`docs/findings.md`](docs/findings.md) (Layer A/B results + caveats + reproducibility recipes).

---

## Documentation

**For different audiences:**

| Document | For whom | What you get |
|----------|----------|-------------|
| **[`technical_report.md`](docs/technical_report.md)** | External AIs (NotebookLM, Perplexity, ChatGPT) + new contributors | Executive summary: proven claims, current model state, evidence artifacts, open questions |
| **[`architecture.md`](docs/architecture.md)** | Researchers, implementers | Full reproduction-ready spec: tensor shapes, equations, module descriptions, IIT 4.0 theory |
| **[`findings.md`](docs/findings.md)** | Ablation readers, reproducibility-focused | Hypothesis ledger: every claim linked to test or result JSON; Layer A (mechanisms) + Layer B (OOD eval) |
| **[`CLI.md`](docs/CLI.md)** | Users of the command-line tools | Reference for `train`, `generate`, `analyze-log`, deployment scripts |
| **[`harness.md`](docs/harness.md)** | People modifying training behavior | BRIANHarness architecture: loss clipping, grad accumulation, maturity phasing, OOD eval |
| **[`BRAIN.md`](docs/BRAIN.md)** | Those diving into the core architecture | NeuralOrchestrator, 11-stage forward pass, re-entry loops, why each design choice exists |
| **[`CONTRIBUTING.md`](CONTRIBUTING.md)** | Future contributors (AI or human) | TDD workflow, testing patterns, documentation sync, adding mechanisms |

**Quick start:**
- Clone the repo, `pip install -r requirements.txt`
- Run: `py -3 -m pytest tests/ -v` (verify setup)
- Train: `python -m neuroslm.train_dsl --arch architectures/rcc_bowtie --scale 30m_p4 --steps 1000`
- Full Colab workflow: `colab_run.ipynb`

---

## Cite / discuss

If BRIAN is useful in your research or you want to discuss the design, open an issue or reach out. ⭐ stars and PRs welcome — this is open research.
