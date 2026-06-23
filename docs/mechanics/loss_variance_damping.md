# loss_variance_damping — BCM Sliding-Threshold LR Modulation

**Category:** training_dynamics  
**Implementation:** `neuroslm.emergent.gif7.LossVarianceDamper`  
**DSL spec:** [`mechanics/loss_variance_damping.neuro`](../../mechanics/loss_variance_damping.neuro)

## Overview

A metaplastic LR modulator implementing the Bienenstock-Cooper-Munro (1982) sliding threshold. The running loss standard deviation σ_L is compared to a reference σ_ref calibrated at post-warmup. When σ_L exceeds σ_ref (volatile training), lr_eff is reduced proportionally. When stable (σ_L ≈ σ_ref), lr_eff runs at full schedule. Part of GIF-7B.

**Root cause addressed**: train-PPL oscillating 3.7× (26→95) due to batch difficulty variance; gap ratio became noise-dominated.

## Equation

```
σ_L(t) = std(L_{t-k}, …, L_t)            # rolling window
lr_eff  = lr_sched · max(min_mult, min(1, σ_ref / σ_L))
```

## Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `window_k` | `50` | Rolling window size in steps |
| `sigma_ref_step` | `2000` | Step at which σ_ref is calibrated |
| `min_mult` | `0.1` | Floor on lr_eff/lr_sched |

## When to Use

When loss shows high step-to-step volatility post-warmup, especially with mixed-domain batches. Set sigma_ref_step to the end of warmup.

## When NOT to Use

- Multi-task runs with naturally high loss variance: would permanently clamp lr
- window_k < 10: σ_L estimate too noisy
- min_mult = 0: learning can stop completely

## Properties

- **BCM analogue**: σ_L high → threshold exceeded → lr drops
- **Self-calibrating**: σ_ref measured from the actual run
- **Self-correcting**: lr_eff recovers when loss stabilises

## Empirical Evidence

Internal (GIF-7B): loss oscillation 3.7× identified as root cause of gap-ratio noise-dominance. BCM, Cooper et al. (2004).

## References

- Bienenstock, Cooper, Munro (1982) Theory for the development of neuron selectivity. J Neurosci
- Internal: `lib/gif.neuro` (gif_loss_variance_damping equation)
