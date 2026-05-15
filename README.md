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

---

## Status

- **126/126 tests passing** (~7s on CPU): every module, every neurochem dynamic, every metric, every memory subsystem, plus the 10 new behavioural tests for BRIAN + Cognitive Closure.
- **Two new test suites** validate the named pass-criteria from the design spec:
  - `tests/test_narrative_memory.py` — causal generalisation, JSON autobiographical coherence, ToM trust divergence, H¹ contradiction detection + SUPERSEDES, predictive-forgetting gain.
  - `tests/test_cognitive_closure.py` — world-model causal predictivity, starvation→qualia shift, BG policy adaptation, personality-trust persistence, GWS ignition selectivity.
- Training resumes cleanly from optimizer-partitioned checkpoint streams (`neuroslm_xl_adamw_*.pt`, `neuroslm_xl_adafactor_*.pt` coexist on disk).
- Adafactor for multi-day TPU runs; **AdamW recommended for short ablations** (~10K steps or less).

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

## Honest scope

BRIAN at 230M parameters will not match GPT-class frontier reasoning — that violates the information-theoretic floor of what fits in 230M weights. The contribution is the **architecture** and what it makes possible:

- **Measurably better at matched FLOPs** than a flat 230M dense transformer.
- An intelligence that **grows over deployment** via persisted `.mem` checkpoints + evolved DNA.
- A **closed embodied loop** where the model acts to survive in its own latent manifold.
- A **falsifiable Φ objective** with five behavioural pass-criteria, not just better next-token loss.

---

## Further reading

- [`docs/architecture.md`](docs/architecture.md) — full reproduction-ready spec.
- `colab_run.ipynb` — Colab notebook for the full ablation → training → benchmarks workflow.

---

## Cite / discuss

If BRIAN is useful in your research or you want to discuss the design, open an issue or reach out. ⭐ stars and PRs welcome — this is open research.
