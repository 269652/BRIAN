# OOD Push — Staged Implementation Plan

Goal: drive the DSL run's OOD `gap_ratio` from the current ~7 down to
below 2.0, by implementing the cited mechanisms **fully in `.neuro`
DSL** (no Brain-side hacks).

Baseline (run 38569395, `dsl_arch_step10000.pt`):
- train PPL 118.97 · OOD PPL 837.64 · gap_ratio **7.04** (STRONG OVERFITTING)

Each stage ships independently, with arch.neuro flags so we can A/B
isolate the contribution of each mechanism. Pushed mid-OOD evals every
3000 steps give us an early signal per run.

| # | Mechanism | Status | DSL surface | Expected gap_ratio |
|---|---|---|---|---|
| 0 | Regularization push (dropout + wd + ctx + pct_strength + STACKED topo) | shipped (run 38601079) | `training { dropout, weight_decay, ... }` + synapse blocks | ~5 (guess) |
| 1 | **PCT proper** — forward-path predictive coding | **THIS COMMIT** | `training { pct_trunk: 0.5 }` | **~3.5** (cited ≥2× reduction) |
| 2 | Tonnetz toroidal attention mask | queued | `block_attention: "tonnetz"` flag on block templates | ~3 |
| 3 | BEMA branching optimizer + PPL-rise rollback | queued | `training { optimizer: "bema" }` | unchanged OOD; flatter loss curve |
| 4 | NEMORI predictive-forgetting episodic gate | queued | `module hippo { episodic_gate: "nemori", nemori_floor: ... }` | ~2.5 |
| 5 | Fisher-Rao retrieval metric | queued | `module hippo { retrieval_metric: "fisher_rao" }` | ~2 |
| 6 | μP scaling for the trunk | queued | `training { param_scaling: "mu_p" }` | unchanged @ 51M; matters at 1B+ |
| 7 | Curriculum + trunk-isolation hardening | queued | `data { curriculum: "easy_to_hard" }` + audit | small but compounds |

The **target sequence** is 1 → 4 → 5 → 2 by expected OOD impact per
hour of work (PCT first because the literature claim is the strongest
and the implementation is the most contained).

---

## Stage 1 — Predictive Coding Trunk (this commit)

### What it is

A *forward-path* PC loop: at every block boundary `i`, the next layer's
output `h_{i+1}` is fed through a learnable `TopDown` layer to predict
what `h_i` should be. The prediction error then *reshapes the trunk
residual at layer i*:

```
pred_i = TopDown_i(h_{i+1})
err_i  = h_i - pred_i
h_i   <- h_i - alpha * pct_trunk * err_i
```

This is different from the existing `pct_strength` knob which only
*weights an aux loss* (the PCH heads). PCT-proper changes the
**representation** the trunk converges to, not just adds a regularizer.

Citations from the OOD plan:
- "deeper layers a generative inverse of shallower" → top-down predicts shallower output ✓
- "filter surface statistics into a high-entropy noise channel" → `err_i` damped by α=0.5 ✓
- "≥2× lower OOD gap ratio at matched train PPL" → quantitative target

### DSL surface

```
training {
  pct_trunk: 0.50   # alpha-weighted forward predictive coding (0 = off)
}
```

### Where it lives

| File | Change |
|---|---|
| `neuroslm/dsl/training_config.py` | `pct_trunk: float = 0.0` field + parser |
| `architectures/rcc_bowtie/arch.neuro` | `pct_trunk: 0.50` in the `training` block |
| `neuroslm/dsl/nn_lang.py` | `DSLLanguageCortex(..., pct_trunk=...)` + `topdown_w` ParameterList + forward-path PCT pass |
| `neuroslm/train_dsl.py` | thread `cfg.pct_trunk` into the LM builder |
| `tests/dsl/test_pct_trunk.py` | 5 tests: zero-init residual identity, perturbation changes forward, parameter accounting, gradient flow |

### Safety properties

- **Zero-init `topdown_w`**: model starts identical to no-PCT baseline.
  PCT effect ramps up only as training learns useful predictions.
- **α = 0.5 damping**: PCT cannot dominate the forward signal even
  if `pct_trunk * topdown_w` blows up.
- **No new bit-identical-parity guarantee**: we are intentionally
  diverging from Brain on the PCT path. Tests verify the *baseline*
  (pct_trunk=0) still matches.

### How to A/B this stage

```
brian deploy --ood 3000               # uses the arch.neuro pct_trunk=0.5
```
vs disable temporarily:
```
# edit arch.neuro: pct_trunk: 0.00
brian deploy --ood 3000
```
Compare the two `ood_results_*.json` final entries in
`logs/vast/benchmarks/ood/`.

---

## Stage 2 — Tonnetz toroidal attention mask (queued)

Adds a periodic mask to the attention scores using a Tonnetz (musical-
torus) topology, constraining the attention's "reach" geometrically.
Constant spectral gap λ₁ — exponentially suppresses high-frequency
incoherent modes (the hallucination biomarker).

Entry points:
- New `nn_ops.causal_self_attention_tonnetz()` op
- Per-block flag: `attention: "tonnetz"` on `_STD_BLOCK_DSL` etc.
- Mask shape configurable: `attention_mask: { kind: "tonnetz", period: 12 }`

---

## Stage 3 — BEMA optimizer (queued)

Branching Ensemble Moving Average. Tracks `PPL_EMA`. When
`d(PPL)/dt > 0` for N consecutive steps, the optimizer's last K updates
are rolled back. Addresses the loss spikes we keep hitting at step
~2400 of every run.

Entry points:
- New `neuroslm/dsl/bema_optimizer.py`
- `training { optimizer: "bema", rollback_window: 50 }`
- Hooks into harness.train_step

---

## Stage 4 — NEMORI predictive-forgetting (queued)

Hippocampal episode admission gate: an episode is only stored if its
surprise exceeds `nemori_floor`. Reduces I(X;Z) (mutual info between
input and stored representation) which tightens the generalization
bound.

Entry points:
- DSL `module hippo { episodic_gate: "nemori", nemori_floor: 0.3 }`
- New `neuroslm/dsl/subsystems/nemori.py`
- Requires episodic-memory plumbing (currently only `hippo` is a
  population; no admission/recall mechanism yet)

---

## Stage 5 — Fisher-Rao retrieval (queued)

Replaces cosine similarity with Fisher-Rao metric for hippocampal
retrieval. Each dimension weighted by inverse variance — model ignores
noisy retrieval signals.

Entry points:
- Depends on Stage 4 (need a retrieval mechanism first)
- `module hippo { retrieval_metric: "fisher_rao" }`

---

## Stage 6 — μP scaling (queued)

Maximal Update Parameterization. Reparameterizes every Linear to keep
representation updates O(1) as width scales. Only matters at 1B+; at
our 51M scale it's a no-op.

Entry points:
- DSL `training { param_scaling: "mu_p" }`
- Affects `_alloc()` initialization + per-layer LR scaling
- Touches every block template

---

## Stage 7 — Curriculum + trunk-isolation hardening (queued)

Easy→hard data ordering during mid-training. Audit every sidecar path
to confirm `sem.detach()` is the only gradient route into the trunk.

Entry points:
- `data { curriculum: "easy_to_hard", crystallization_step: 2000 }`
- Static analysis test that walks the IR and asserts no
  non-detached path from sidecar → trunk
