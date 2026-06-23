# allostasis — Synthetic HPA Axis

**Category:** training_dynamics  
**Implementation:** `neuroslm.harness.AllostasisController`  
**DSL spec:** [`mechanics/allostasis.neuro`](../../mechanics/allostasis.neuro)

## Overview

A synthetic Hypothalamic-Pituitary-Adrenal (HPA) axis with two timescales. A fast EMA (τ≈10 steps) integrates multi-modal stress (NE excess, GABA excess, loss spikes, grad spikes) into `load`. A slow EMA (τ≈50 steps) integrates `load` into `cort` (cortisol). Three downstream effectors reduce NE release, trophic growth (BDNF), and optimizer LR proportionally to `cort`. The 5× timescale ratio means transient spikes leave `cort≈0` while sustained 200-step stress drives `cort>0.25` where dampers bite.

## Equation

```
stress = w_NE·relu(z_NE−ne_baseline) + w_GABA·... + w_loss·... + w_grad·...
load   = (1−α_load)·load + α_load·stress        # τ ≈ 10 steps
cort   = (1−α_cort)·cort + α_cort·load          # τ ≈ 50 steps
ne_mult   = 1 − γ_NE·cort
lr_mult   = 1 − γ_LR·cort
```

## Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `load_ema_alpha` | `0.10` | Fast EMA α (τ ≈ 10 steps) |
| `cort_ema_alpha` | `0.02` | Slow EMA α (τ ≈ 50 steps) |
| `w_ne / w_gaba / w_loss / w_grad` | `0.30/0.20/0.30/0.20` | Stress channel weights (sum=1) |
| `gamma_ne / gamma_trophic / gamma_lr` | `0.7 / 1.0 / 0.5` | Effector gains |

## When to Use

When the model has a NT homeostat (NE, GABA, DA, 5HT) and you observe NE runaway over 40+ steps (NE rising monotonically → sharper routing → harder gradient → more NE). Motivated by run 38470012 (NE 0.20→0.93, gnorm 10→24 between steps 460-500).

## When NOT to Use

- No NT stack: ne_mult and troph_mult have no consumers
- Runs < 500 steps: cort integrator needs time to develop
- gamma_lr > 0.8 + sustained high cort: lr_eff may drop below effective learning threshold

## Properties

- **Acute vs chronic**: single bad batch → cort≈0; 200-step sustained stress → cort>0.25
- **No new parameters**: pure control system
- **Three independent axes**: any combination of NE/trophic/lr suppression

## Empirical Evidence

38-contract test suite (`tests/training/test_allostasis.py`): smoking-gun replay of run 38470012 steps 460-500. With allostasis: NE runaway suppressed.

## References

- Sterling & Eyer (1988) Allostasis: a new paradigm to explain arousal pathology
- Aston-Jones & Cohen (2005) Locus coeruleus model. Annu Rev Neurosci
- Internal: `tests/training/test_allostasis.py` (38 contracts)
