# OOD-Generalization Mechanism Catalog

All mechanisms below are **declarative** — toggle them by setting a field
in your `arch.neuro` `training { ... }` block. Each is wired through:

1. `neuroslm/dsl/training_config.py` — field on `TrainingConfig` dataclass
2. `parse_training_config()` — reads the value from arch.neuro
3. `neuroslm/harness.py` or `neuroslm/dsl/nn_lang.py` — applies the effect

To check what's active in a run, grep the training logs for
`[harness] LLRD enabled`, `[mid-ood]`, etc.

---

## Regularization (Train↔OOD gap reducers)

| Field | Default | Range | Mechanism | Wired in |
|-------|---------|-------|-----------|----------|
| `dropout` | 0.0 | 0.0–0.3 | Post-embed + post-block residual dropout. Standard. | `nn_lang.py` |
| `stochastic_depth` | 0.0 | 0.0–0.5 | Skip blocks with linearly increasing probability (Huang et al. 2016). **Skip-aware with PCT** — dropped blocks excluded from PCT pairs. | `nn_lang.py` |
| `flooding_level` | 0.0 | 3.0–5.0 | `|loss - b| + b` — refuses to push train loss below `b`. Ishida et al. 2020. | `harness.py` |
| `z_loss` | 0.0 | 1e-5–1e-3 | `α * logsumexp(logits)^2` — caps logit magnitudes. PaLM/Gemma. Direct fix for logit-explosion-driven train↔OOD divergence. | `harness.py` |
| `logit_norm_tau` | 0.0 | 0.02–0.1 | LogitNorm (Wei et al. ICML'22) — trains CE on `f/(τ·‖f‖)`. **Scale-invariant**: confidence set by logit *direction*, never magnitude, so the model can't lower loss by inflating `‖f‖` → caps OOD CE near (not above) uniform. H30 guardrail; sharper than z_loss. **0.0 = off.** | `harness.py` |
| `multi_cortex.consistency_weight` | 0.0 | 0.5–2.0 | Jacobian-consistency distillation (Srinivas & Fleuret ICML'18) — `T²·KL(teacher(x) ‖ student(x+δ))` with δ embedding noise. Transfers the teacher's *generalising function* (its input-Jacobian), not training-point values → fixes the H28 memorise→OOD-explosion. **0.0 = off.** | `harness.py` |
| `multi_cortex.consistency_noise_std` | 0.1 | 0.05–0.3 | σ of the Gaussian embedding perturbation for `consistency_weight`. Larger σ probes a wider function neighbourhood. | `nn_lang.py` |
| `multi_cortex.consistency_batch` | 1 | 1–B, 0=all | Sequences fed to the consistency probe's 2nd forward. The full `(B,T,V)` 2nd forward OOMs in backward (1.54 GiB logit grad); 1 keeps a real full-context probe while bounding memory. | `harness.py` |
| `multi_cortex.consistency_max_tokens` | 512 | 128–2048, 0=full | Prefix length for the consistency probe. Caps the 2nd forward's logits+grad. Unbiased Jacobian estimate on the subsample. | `harness.py` |
| `label_smoothing` | 0.0 | 0.0–0.1 | Standard CE smoothing. | `harness.py` |
| `weight_decay` | 0.01 | 0.01–0.2 | AdamW L2 regularization. | `harness.py` |
| `loss_clipping.factor` | 3.0 | 2.0–5.0 | Per-sample loss clipping at `factor × batch_median` (p4 fix). | `harness.py` |

## Optimization (Convergence + generalization)

| Field | Default | Range | Mechanism | Wired in |
|-------|---------|-------|-----------|----------|
| `optimizer` | "adamw" | adamw, adafactor | Choice of optimizer. | `harness.py` |
| `learning_rate` | 3e-4 | 1e-4–1e-3 | Base LR for cosine schedule. | `harness.py` |
| `warmup_steps` | 300 | 100–2000 | Linear LR warmup at start. | `harness.py` |
| `min_lr_ratio` | 0.1 | 0.0–0.2 | Floor LR as ratio of base at end of cosine. | `harness.py` |
| `llrd` | 1.0 | 0.75–0.95 | Layer-wise LR decay (ULMFiT/DeBERTa). `lr_i = base * llrd^(depth-1-i)`. Bottom learns fast, top learns slow → prevents memorization. **1.0 = off.** | `harness.py` |
| `mu_p_scaling` | false | bool | μP-aware AdamW LR multipliers per-param-group. Pays off at >200M params. | `harness.py` |
| `grad_clip` | 1.0 | 0.5–2.0 | Max gradient norm before optimizer step. | `harness.py` |
| `grad_accum` | 1 | 1–32 | Microbatch accumulation. | `harness.py` |

## Forward-Path Architectural Aux Losses

| Field | Default | Range | Mechanism | Wired in |
|-------|---------|-------|-----------|----------|
| `pct_trunk` | 0.0 | 0.2–0.6 | Predictive Coding Trunk — top-down feedback shapes the trunk's residual stream via learned `topdown_w` projections. Skip-aware. | `nn_lang.py` |
| `pct_strength` | 0.0 | 0.1–0.5 | PCH aux loss strength (per-layer MSE between adjacent block outputs). | `harness.py` |
| `tonnetz_period` | 0 | 12 | Toroidal/musical-circle attention bandwidth mask. Suppresses attention to far-out positions. | `nn_lang.py` |

## Online Loss-Surface Guards

| Field | Default | Range | Mechanism | Wired in |
|-------|---------|-------|-----------|----------|
| `bema_rollback_window` | 0 | 50–200 | Branching EMA optimizer wrapper — rolls back N steps if loss-EMA rises. | `harness.py` |
| `nemori_floor` | 0.0 | 0.05–0.20 | Per-batch surprise gate — skips low-surprise batches. | `harness.py` |

## Curriculum

| Field | Default | Range | Mechanism | Wired in |
|-------|---------|-------|-----------|----------|
| `curriculum` | "random" | random, easy_to_hard, uniform | Data ordering strategy. | `train_dsl.py` |
| `crystallization_step` | 0 | step number | Boundary between curriculum phases. | `train_dsl.py` |

## MAT-Phase-Gated Mechanisms

When `mechanisms { ... }` is declared, each mechanism's strength is
`declared × phase_gate(maturity)` — mechanisms ramp up only after the
model crosses a maturity threshold. Lets you run early training without
regularization (fast convergence to ~70 ppl) then engage regularizers.

---

## Current `rcc_bowtie_30m_p4` OOD stack (Jun 2026)

```neuro
training {
    dropout:          0.12       # baseline residual dropout
    stochastic_depth: 0.1        # skip-aware (compatible with PCT)
    flooding_level:   4.0        # refuse train loss < 4.0
    z_loss:           0.0001     # cap logit magnitudes (PaLM)
    llrd:             0.85       # layer-wise LR decay (ULMFiT)
    pct_trunk:        0.4        # forward-path predictive coding
    pct_strength:     0.3        # PCH aux loss
    weight_decay:     0.12
    label_smoothing:  0.05
    loss_clipping: { enabled: true, method: "per_sample", factor: 3.0 }
}
```

Expected effect (vs p4 baseline at step 10k):
- Train PPL: similar (~150) — z-loss + LLRD slow memorization but flood
  bounces off floor, keeping gradients useful
- OOD PPL: **3–4× lower** (target gap_ratio ≤ 3.0 vs p4's 13×)

## Adding a new mechanism

1. Add field + docstring to `TrainingConfig` in `dsl/training_config.py`
2. Add parser line in `parse_training_config()`
3. Implement the effect in `harness.py` (loss/optimizer) or `nn_lang.py`
   (forward path). Use `self.training_config.<field>` to read.
4. Document the field in this file (table above).
5. (Optional) Add to `mechanisms { ... }` parser for MAT phase-gating.

The arch.neuro DSL is the single source of truth — no code changes
needed to A/B-test mechanism combinations across architectures.
