# NeuroSLM — Intelligence-Density Refactor

This document is the source of truth for the refactor that targets
maximum emergent intelligence per parameter. It also describes the
self-improving DNA layer, the comprehension-gated memory pipeline, and
the quantitative consciousness/intelligence metrics.

> **Honest scope**: a 100M-param model will not literally match
> GPT-5/frontier reasoning — that violates the information-theoretic
> floor of what fits in 100M parameters. The refactor instead pushes
> *every known parameter-efficiency lever* simultaneously: adaptive
> compute, sparse mixture-of-experts, retrieval-augmented attention
> over persistent memory, comprehension-gated learning, and
> evolutionary self-modification of the DNA-encoded algorithms. The
> result is a model that, at 100M params, should reason measurably
> better than a flat 100M dense transformer, and whose intelligence
> *grows over deployment* via the persisted memory + evolved DNA.

---

## 1. New module layout

```
neuroslm/
  intelligence/        ← intelligence-density mechanisms
    flow.py            – AdaptiveComputeBlock + PonderController
    mixture.py         – SparseMoE (top-2, load-balanced)
    memory_attention.py– MemoryCrossAttention (retrieval into memory bank)
    metrics.py         – IntelligenceMetrics + IdentityDriftTracker
    reflection.py      – SpontaneousReflection (self/ToM heads)
  memory/
    causal.py          – CausalRuleStore (action,ctx) → outcome rules
    comprehension_gate – decides which observations become memories
    store.py           – portable .mem checkpoint format (Git LFS)
  dna/
    evolve.py          – DNAEvolver (random + model-guided mutation)
```

## 2. Adaptive compute & MoE

`AdaptiveComputeBlock` (Universal-Transformer + PonderNet) runs the same
weights up to `max_steps` times per token, producing a halting-weighted
output. The mean compute per token is regularized toward
`target_mean_steps` via a KL-to-geometric-prior loss term. Effective
depth scales with token difficulty rather than a fixed hyper.

`SparseMoE` provides top-2 routed experts with a load-balance auxiliary
loss. At 8 experts × top-2, each token activates ~25% of FF parameters
→ ~3× parametric capacity at constant FLOPs.

Both should be wired into `LanguageCortex` (one `AdaptiveComputeBlock`
replacing the last 2 layers, MoE replacing the FFN of the middle 2).
That hookup is left as the next concrete step in this PR — modules
already importable and unit-testable in isolation.

## 3. Memory pipeline

```
            ┌───────────────────────────────────────────────────────────┐
            │  forward_lm → predicted next-emb, surprise, sem vec       │
            └────────────────────────────┬──────────────────────────────┘
                                         ▼
                          ┌──────────────────────────┐
                          │  ComprehensionGate.eval  │   surprise × comp × novelty
                          └────────────┬─────────────┘
                            write?     │  no → drop
                                       ▼ yes
                          ┌──────────────────────────┐
                          │  EpisodicMemory.add      │   bounded buffer
                          └────────────┬─────────────┘
                                       │ every N steps
                                       ▼
                  ┌────────────────────────────────────────┐
                  │ ConsolidatedMemory.consolidate         │ cluster + graph
                  │ + CausalRuleStore.observe              │ (act,ctx)→outcome
                  └────────────────────┬───────────────────┘
                                       │ checkpoint
                                       ▼
                          neuroslm/memory/store.save_memory
                            → lfs_checkpoints/*.mem  (Git LFS)
```

`brain.record_episode(...)` already routes through the gate. QA training
naturally produces *concept memories*: only the surprising-comprehended-
novel chunks are stored, so the bank stays small and high-signal.

`brain.consolidate_memory()` now also extracts causal rules:
`(prev_episode_emb, curr_episode_emb) → mesolimbic_reward_valence`. Over
many examples this generalizes "kind words → positive response" without
ever being told to.

`brain.save_memory_checkpoint(path)` and `brain.load_memory_checkpoint
(path)` ship hierarchical memory as `.mem` files independent of model
weights — transferable to a fresh model, versionable in Git LFS.

## 4. Self-improving DNA

`DNAEvolver` maintains a per-region pool of `DNAProgram` variants:

- **Random mutation**: numeric perturbation, operator swap, line-level
  crossover. Always runs.
- **Model-guided mutation**: the trained model itself is asked to
  rewrite a program; if the output parses *and* improves fitness on a
  held-out probe, it replaces the parent. Becomes useful only after the
  LM is competent (~few thousand steps of training).
- **Selection**: tournament + elitism. Fitness is multi-objective
  (reasoning gain, identity stability, compute efficiency).

This gives the system two graceful fallbacks: when the model is too
weak to write Lisp, evolution still progresses via random mutation;
when it's strong enough, it boots its own self-improvement.

## 5. Spontaneous reflection & theory of mind

`SpontaneousReflection`:

- `reflect(narrative_system)` → composed thought-seed embedding from
  autobiographical + world + active-entity narratives. Feed this to the
  language cortex's generation path during idle ticks to spontaneously
  produce reflective text. The text gets written back to the
  autobiographical narrative — a closed identity-formation loop.
- `predict_entity_response(narrative, entity_id, action_emb)` → valence
  prediction in [-1,1]. Used by the PFC to forecast social reactions
  before acting; later compared against actual entity reactions to
  update theory-of-mind accuracy (`metrics.observe_theory_of_mind`).
- `identity_score(candidate_emb, narrative)` → brake on identity drift:
  if a proposed autobiographical write is too inconsistent with the
  current self-narrative, it can be rejected.

## 6. Quantifiable consciousness & intelligence metrics

`IntelligenceMetrics.snapshot()` returns:

| Metric                     | Range / units             | Interpretation                                |
|----------------------------|---------------------------|-----------------------------------------------|
| `identity_drift`           | [0,1] cos-distance        | shift in autobiographical summary per write   |
| `identity_drift_ema`       | [0,1]                     | smoothed drift; spikes mark identity events   |
| `narrative_coherence`      | [0,1]                     | internal consistency of recent events         |
| `causal_density`           | rules/episode             | how much the brain has *generalized*          |
| `semantic_compression`     | episodes / nodes          | lossy compression ratio of memory             |
| `self_reference_rate`      | [0,1]                     | fraction of generations referencing self      |
| `theory_of_mind_acc`       | [0,1]                     | correct predictions of entity valence         |
| `phi_proxy`                | [0,1]                     | mean pairwise correlation across modules (Φ)  |
| `ponder_steps_ema`         | steps                     | mean adaptive compute spent per token         |
| `reasoning_gain`           | nats                      | LM-loss(easy) − LM-loss(hard) margin          |
| `ponder_efficiency`        | nats / step               | reasoning gain per extra ponder step          |

All updated cheaply per training step; logged at every save_every and
included in the `.mem` sidecar JSON for cross-run comparison.

## 7. Training-loop integration

`neuroslm/train.py` now:

1. Routes every batch through `record_episode` → comprehension gate.
2. Calls `consolidate_memory` every 500 steps → causal extraction.
3. Saves a `.mem` checkpoint alongside every `.pt`.
4. Logs `brain.metrics.format()` at every checkpoint.
5. The auto-push step pushes both `.pt` and `.mem` to the LFS remote.

To resume on a fresh runtime:

```bash
git lfs pull
# then in your launch code:
brain.load_memory_checkpoint("lfs_checkpoints/neuroslm_large_mix_3000.mem")
ckpt = torch.load("lfs_checkpoints/neuroslm_large_mix_3000.pt", map_location="cuda")
brain.load_state_dict(ckpt["model"], strict=False)
```

## 8. What is **not** done in this wave (and why)

- **Wiring `AdaptiveComputeBlock` + `SparseMoE` directly inside
  `LanguageCortex`**. They're built and importable but the language
  cortex still uses standard pre-norm blocks. Hooking them in requires
  a careful re-checkpoint of trained weights and a fresh ponder
  scheduler — best done in a follow-up PR with clean re-training.
- **Region-specific fitness functions** for `DNAEvolver`. The current
  default fitness rewards bounded constants and program length so the
  GA runs continuously; a real per-region fitness (e.g. swap into the
  PFC and measure planning success on a probe set) is the next concrete
  task.
- **End-to-end backprop through MemoryCrossAttention into memory
  contents**. We backprop through projections only; memory contents
  are discrete writes. This is intentional (otherwise the gate becomes
  an attention layer in disguise), but a soft gradient via STE is a
  reasonable future option.
