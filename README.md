# NeuroSLM — A Neuroanatomically Inspired Small Language Model

A research prototype of a brain-inspired small language model. Three claims set
NeuroSLM apart from a flat transformer of the same parameter budget:

1. **Topology over scale.** A bowtie architecture (sensory → thalamus → GWS
   bottleneck → memory → executive → motor) with both within-pass and
   cross-pass re-entry loops. Feedforward chains have Φ = 0 by IIT
   construction; re-entry makes every bipartition costly.
2. **A real, differentiable Φ objective.** Φ (integrated information) is
   computed as the Gaussian-MI lower bound across the minimum information
   partition of the live module outputs and added to the loss as
   `-w_phi · tanh(Φ/3) · 3`. Backprop pushes module outputs toward
   configurations where every bipartition retains high MI.
3. **Persistent, comprehension-gated memory.** Surprise × comprehension ×
   novelty decides what enters episodic memory; consolidation extracts
   causal rules; memory ships as `.mem` sidecars next to `.pt` checkpoints
   and is transferable across architectures.

## Status

- All major modules implemented and importable as standalone PyTorch
  `nn.Module`s.
- Cognitive loop (DMN + GWS + PFC + BG + motor) wired end-to-end through a
  `NeuralOrchestrator` with stage-typed routing and HomeostaticGate
  stability control at every edge.
- Streaming pre-training on Cosmopedia (open Phi-style dataset). Adafactor
  by default for TPU-native memory footprint; AdamW fallback for CUDA debug.
- 116-test suite covers every module, every metric, every neurochem dynamic
  (`py -3 -m pytest tests/` — ~14s on CPU).
- **Not yet competitive with production SLMs.** This is research
  scaffolding. A 100M-param model will not literally match frontier
  reasoning — it violates the information-theoretic floor of what fits in
  100M parameters. The point is the architecture: measurably better than a
  flat 100M dense transformer at the same FLOPs, and an intelligence that
  *grows* over deployment via persisted memory.

## Architectural highlights

### Bowtie topology with bidirectional re-entry

```
┌────────────────────────────────────────────────────────────────────┐
│                      NeuralOrchestrator                            │
│  Stage 0  SENSORY      TextSensoryCortex, AssociationCortex        │
│  Stage 1  THALAMUS     Thalamus  ◄──────────── re-entry bias       │
│  Stage 2  STATE_MODELS WorldModel, SelfModel                       │
│  Stage 3  SUBCORTICAL  Amygdala, LHb, Insula                       │
│  Stage 4  QUALIA       QualiaState                                 │
│  Stage 5  GWS  ━━━━━━━ GlobalWorkspace ─── within-pass broadcast   │
│  Stage 6  MEMORY       Hippocampus, Entorhinal, HyperGraph         │
│  Stage 7  COG_CTL      PFC, ACC, Cerebellum                        │
│  Stage 8  EXECUTIVE    BasalGanglia, ForwardModel                  │
│  Stage 9  CONSCIOUS    DMN, ThoughtTransformer, Claustrum          │
│  Stage 10 MOTOR        MotorCortex                                 │
│                                            PFC+GWS → next thalamus │
└────────────────────────────────────────────────────────────────────┘
```

Two re-entry loops:
- **Within-pass** (cortico-cortical): after Stage 5, the GWS broadcast is
  injected as a zero-init learnable residual into every subsequent module.
- **Cross-pass** (thalamo-cortical): the PFC+GWS output is EMA-smoothed and
  added to next forward's thalamic input. Sigmoid-gated mixing coefficient
  learns how strongly to apply it.

### Real Φ computation

`neuroslm/intelligence/orchestrator.py::phi_tensor` builds the n×n Gram
covariance from mean-pooled module outputs:

```
Σ = M_c · M_c^T / (d − 1)        # M_c = mean-centred outputs, (n, d)
```

For n ≤ 8 it enumerates all 2^(n-1)-1 bipartitions; for n > 8 it falls back
to Fiedler spectral bisection. MI(A;B) is the Gaussian lower bound:

```
MI = ½ · (logdet Σ_A + logdet Σ_B − logdet Σ_AB)
```

The minimum over bipartitions is Φ. Implementation uses `torch.linalg.slogdet`
which is differentiable, so `phi_tensor()` flows gradient back into every
contributing module. The training loss adds `-w_phi · tanh(Φ/3) · 3` (bounded
saturation prevents Φ from dominating). A controlled A/B test shows the
objective injects ~100 grad-units of language-cortex gradient that's absent
without it.

### Bio-inspired plasticity & neurochemistry

- **TrophicSystem** (BDNF/NGF) — Φ-gated structural plasticity. When the
  Fiedler spectral gap drops below 0.3, BDNF is boosted along the cut to
  rewire the fault line (Cheeger's inequality).
- **NeuromodulatorSystem** — DA, NE, 5-HT, ACh, eCB, Glu, GABA as
  learned scalar gain signals. Modulates attention, memory writes, and
  learning rates without adding per-decision parameters.
- **GPCRBank** — metabotropic modulation. Sustained ACh widens DG winners
  (more pattern encoding); high NE blocks CALM early-exit (full-depth
  processing under arousal).
- **HebbianFastWeights** — second tier of weights that update within a
  single forward pass at expert cortex outputs (stages 7, 8). Low-rank
  factorisation keeps memory bounded.

### Comprehension-gated memory pipeline

```
forward_lm → predicted next-emb, surprise, sem vec
       │
       ▼  surprise × comp × novelty > τ ?
ComprehensionGate
       │ write
       ▼
EpisodicMemory  (bounded buffer)
       │ every N steps
       ▼
ConsolidatedMemory  (graph nodes, cluster + edge)
       │ + CausalRuleStore (act,ctx)→outcome
       ▼
memory/store.py  →  lfs_checkpoints/*.mem  (Git LFS)
```

`.mem` checkpoints are independent of model weights and transferable to a
fresh model — load them onto a re-architected NeuroSLM and you keep the
learned episodes, causal rules, and narrative streams.

## Parameter presets

| Preset  | Params  | Accelerator | VRAM   | d_hidden | d_sem | lang_layers | lang_ctx |
|---------|---------|-------------|--------|----------|-------|-------------|----------|
| `tiny`  | ~5 M    | CPU         | —      | 192      | 128   | 2           | 256      |
| `small` | ~15 M   | CPU         | —      | 384      | 256   | 4           | 512      |
| `medium`| ~80 M   | T4          | 16 GB  | 768      | 512   | 8           | 1024     |
| `large` | ~100 M  | T4          | 15 GB  | 384      | 256   | 8           | 1024     |
| `xl`    | ~258 M  | A100        | 40 GB  | 512      | 384   | 12          | 2048     |
| `xxl`   | ~10 B   | 4×A100      | 320 GB | 4096     | 2048  | 32          | 4096     |

Pass `--baseline` for vanilla-transformer ablation at matched parameter count.

## Layout

```
neuroslm/
  config.py                  # All hyperparameters; per-module enable flags
  tokenizer.py               # tiktoken / GPT-2 BPE wrapper
  brain.py                   # The full forward pipeline
  train.py                   # Streaming pre-training loop
  generate.py                # Inference REPL
  interactive.py             # Chat-style interactive mode

  intelligence/              # Intelligence-density mechanisms
    orchestrator.py          # NeuralOrchestrator, HomeostaticGate, Φ
    flow.py                  # AdaptiveComputeBlock + PonderController
    mixture.py               # Sparse MoE (top-2, load-balanced)
    memory_attention.py      # MemoryCrossAttention (retrieval into bank)
    metrics.py               # IntelligenceMetrics + IdentityDriftTracker
    reflection.py            # SpontaneousReflection (self/ToM heads)
    active_inference.py      # Free-energy / predictive coding
    contrastive_predictive_coding.py
    oscillations.py          # δ/θ/α/β/γ band tracking via FFT

  modules/                   # Brain-area implementations
    workspace.py             # Hopfield GWS with ignition phase transition
    language.py              # DiffAttn + MoD-interleaved transformer
    pfc.py                   # Prefrontal cortex (selection + gating)
    dmn.py                   # Default Mode Network
    hippocampus.py           # DG/CA3/CA1 + sparse novelty
    basal_ganglia.py         # Go/NoGo action gating
    motor.py                 # Action selector + lang-bias
    thalamus.py              # Routing + re-entry injection
    qualia.py                # Phenomenal state representation
    consciousness.py         # ConsciousnessMetrics, estimate_fiedler
    cortical_column.py       # Cortical columns + minicolumns
    entorhinal.py            # Grid-cell module
    claustrum.py             # Cross-modal binding
    cerebellum.py            # Efference-copy predictor
    neural_geometry.py       # Meta-trainable manifold reshaping
    fast_weight.py           # Hebbian fast-weight memory
    forward_model.py         # Next-state predictor (cerebellum)
    evaluator.py             # ACC/OFC value head
    amygdala.py              # Fear conditioning, emotional tagging
    anterior_cingulate.py    # Conflict monitoring
    insula.py                # Interoception
    thought_transformer.py   # Sustained recurrent working memory
    world_model.py / self_model.py
    [+ many opt-in extensions: phase-modulated attn, HTM, dyn-MoE,
     active dendrite, neurogenesis, theory-of-mind, …]

  memory/                    # Persistent memory subsystem
    episodic.py              # Short-term episodic buffer
    consolidated.py          # Long-term semantic graph
    causal.py                # (action, context) → outcome rules
    narrative.py             # Autobiographical / world streams
    relational_graph.py      # Knowledge-triple insight store
    hypergraph.py            # N-ary hyperedge memory
    entity_store.py          # Per-entity style fingerprints
    hippocampal.py           # Multi-dimensional recall enrichment
    mesolimbic.py            # DA-tagged reward memory
    comprehension_gate.py    # Adaptive write filter
    consolidation.py         # Cluster + extract causal rules
    store.py                 # .mem checkpoint format (Git LFS)

  neurochem/                 # Neurotransmitter dynamics
    transmitters.py          # 7-channel NT state machine
    receptors.py             # ReceptorBank + GPCRBank
    nuclei.py                # VTA, NAcc, LC, Raphe, NBM, SN, PAG, HypoCRH
    projections.py           # ProjectionGraph
    gated_projections.py     # Vesicle-gated projections
    growth.py                # TrophicSystem (BDNF/NGF)
    vesicles.py              # Neuro-vesicle content packets
    homeostasis.py           # Per-NT setpoint regulation
    lateral_habenula.py      # Anti-reward / aversion learning
    mesolimbic_circuit.py    # Wanting/liking/consolidation
    plasticity.py            # PlasticityGate
    reuptake.py / desensitization.py

  environments/
    virtual_world.py         # 7 environments: BusStop, MeadowTree,
                             # RainyWindow, OceanCliff, Library, Campfire,
                             # GridWorld (action-conditioned)

  dna/                       # Self-modifying program DSL (research)
    dsl.py / compiler.py / evolve.py / latent_program.py

tests/                       # 116 unit + integration tests (~14s on CPU)
  test_phi.py                # IIT MIP estimator + differentiability
  test_brain_forward.py      # End-to-end forward/backward
  test_orchestrator.py       # Gates, routing, re-entry inplace-safety
  test_consciousness.py      # ConsciousnessMetrics, Fiedler
  test_intelligence_modules.py
  test_modules.py            # All brain areas
  test_memory.py             # Episodic / consolidated / causal / narrative
  test_neurochem.py          # Transmitters, trophic Φ-gating, vesicles
  test_environments.py       # All envs + GridWorld dynamics
  test_config_presets.py
  conftest.py                # Shared fixtures

docs/
  architecture.md            # Full reproduction-ready spec (IIT 4.0)
  refactor.md                # Intelligence-density refactor notes
```

## Quick start

```bash
# Windows PowerShell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
# torch is intentionally not in requirements.txt — install matching your
# accelerator, e.g.:
#   pip install torch --index-url https://download.pytorch.org/whl/cu121

# Train a small CPU-trainable model
python -m neuroslm.train --preset small --steps 2000 --batch_size 4

# XL preset on A100 with Adafactor (TPU-native), bf16, grad-checkpointing
python -m neuroslm.train --preset xl --steps 100000 --batch_size 4 \
    --optimizer adafactor --device cuda

# Resume the latest checkpoint (auto-finds in checkpoints/ and lfs_checkpoints/)
python -m neuroslm.train --resume latest

# Mixed text + chat training (recommended once base LM is decent)
python -m neuroslm.train --mode mix --chat_ratio 0.75

# Baseline ablation: vanilla transformer at matched params, no bio modules
python -m neuroslm.train --preset xl --baseline

# Interactive generation
python -m neuroslm.generate --prompt "Once upon a time"
```

## Loss composition

`brain.forward_lm` returns a `loss` tensor that's a weighted sum of:

| term                | source                                      | weight       |
|---------------------|---------------------------------------------|--------------|
| `lm_loss`           | mesolimbic-gain-modulated cross-entropy     | `w_lm`       |
| `world_loss`        | MSE between predicted and target world emb  | `w_world`    |
| `motor_loss`        | cross-entropy on speak/silent action target | `w_motor`    |
| `pred_coding_loss`  | per-layer next-layer prediction (lang.)     | `w_pred_coding` |
| `rssm_kl`           | (optional) RSSM KL prior–posterior          | `w_kl_world` |
| `cpc_loss`          | (optional) contrastive predictive coding    | `w_cpc`      |
| **`phi_loss`**      | **`-tanh(Φ/3)·3` from real MIP estimator**  | **`w_phi`**  |
| `novel_aux_loss`    | aggregate of opt-in novel-module aux losses | 0.05         |
| `forward_reg`       | small magnitude regulariser on fwd model    | `w_forward`  |

Aux weights ramp from 0.1× to 1.0× over the first 20% of training to let
the LM loss stabilise first.

## Quantifiable consciousness & intelligence metrics

`IntelligenceMetrics.snapshot()` (intelligence/metrics.py) returns:

| Metric                  | Range                | Interpretation                              |
|-------------------------|----------------------|---------------------------------------------|
| `phi_proxy`             | [0, ∞) nats          | IIT MIP Gaussian-MI lower bound             |
| `identity_drift`        | [0, 1] cos-distance  | shift in autobiographical summary per write |
| `narrative_coherence`   | [0, 1]               | internal consistency of recent events       |
| `causal_density`        | rules / episode      | how much the brain has *generalized*        |
| `semantic_compression`  | episodes / nodes     | lossy compression ratio of memory           |
| `self_reference_rate`   | [0, 1]               | fraction of generations referencing self    |
| `theory_of_mind_acc`    | [0, 1]               | correct predictions of entity valence       |
| `ponder_steps_ema`      | steps                | mean adaptive compute per token             |
| `reasoning_gain`        | nats                 | LM-loss(easy) − LM-loss(hard) margin        |
| `ponder_efficiency`     | nats / step          | reasoning gain per extra ponder step        |

Plus `ConsciousnessMetrics.update()` reports per-tick `gamma` (binding),
`theta` (memory), `alpha` (idling), `phi`, `coherence`, `ignition`,
`metacognition`, `binding`, all updated on every forward pass during
training.

## Tests

```bash
py -3 -m pytest tests/        # full suite, ~14s
py -3 -m pytest tests/test_phi.py -v   # just the Φ guarantees
```

Coverage highlights:
- `test_phi.py::test_phi_higher_for_coupled_outputs` — Φ for rank-1 coupled
  outputs > Φ for independent outputs (not a tautology, a property of the
  estimator).
- `test_phi.py::test_phi_tensor_is_differentiable` — backward through
  `slogdet` reaches the inputs.
- `test_brain_forward.py::test_phi_objective_increases_total_gradient` —
  A/B test confirms the Φ term injects real gradient (not just logging).
- `test_neurochem.py::test_trophic_phi_boosts_growth` — high Φ pathways
  receive at least as much trophic support as low Φ.
- `test_orchestrator.py::test_reentry_bias_safe_under_inplace_update` —
  guards the buffer-clone fix that lets re-entry coexist with backward.

## Honest scope

A 100M-param model will not match GPT-5-class frontier reasoning — that
violates the information-theoretic floor. NeuroSLM pushes *every known
parameter-efficiency lever simultaneously*: adaptive compute, sparse
mixture-of-experts, retrieval-augmented attention over persistent memory,
comprehension-gated learning, Φ-objective training, and evolutionary
self-modification of the DNA-encoded algorithms. The result is a model
that, at 100M params, should reason measurably better than a flat 100M
dense transformer, and whose intelligence *grows over deployment* via the
persisted `.mem` files + evolved DNA.

## Further reading

- `docs/architecture.md` — full reproduction-ready spec including IIT 4.0
  postulate mapping, tensor shapes, XLA/TPU notes, and pseudocode for
  every algorithm.
- `docs/refactor.md` — intelligence-density refactor rationale and the
  open items deliberately left for follow-up PRs.
- `COLAB_README.md` / `COLAB_DPO_README.md` — Colab-specific launch and
  DPO post-training notes.
