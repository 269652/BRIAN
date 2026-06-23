# divisive_grad_norm — Smooth Cortical Gain Control

**Category:** training_dynamics  
**Implementation:** `neuroslm.emergent.gif7.divisive_grad_normalize`  
**DSL spec:** [`mechanics/divisive_grad_norm.neuro`](../../mechanics/divisive_grad_norm.neuro)

## Overview

Replaces hard gradient clipping (`clip_grad_norm_`) with C∞-smooth divisive normalisation: `g' = g·c/√(c²+‖g‖²)`. The semi-saturation constant c is the gnorm at which the gradient is halved. Below c, behaviour is nearly linear (full gradient); above c, compressive (gradient magnitude ≈ c). No discontinuity means Adam's m/v estimates remain consistent. Biological analogue: Heeger (1992) V1 contrast gain control.

## Equation

```
g' = g · c / √(c² + ‖g‖²)
# ‖g‖ << c  →  g' ≈ g        (linear regime)
# ‖g‖ >> c  →  g' ≈ c·g/‖g‖  (compressive regime)
```

## Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `c` | `1.0` | Semi-saturation constant (gnorm at which gradient is halved) |

## When to Use

Replace `clip_grad_norm_` when gradient spikes cause loss divergence and you want a smooth alternative. Set c to the p90 gnorm at a stable training step.

## When NOT to Use

- c << gnorm_typical: effectively zeroes gradient at normal values
- sign-SGD variants: direction is preserved here (safe), but the compressive magnitude may conflict

## Properties

- **C∞ smooth**: no discontinuity
- **Monotone**: ‖g'‖ ≤ ‖g‖ always
- **Direction-preserving**: g' ∥ g
- **Adam-compatible**: no boundary discontinuity to miscalibrate m/v

## Empirical Evidence

Heeger (1992): divisive normalisation accounts for V1 gain control. GIF-7A: replacement of hard clip in neuroslm; isolated ablation pending.

## References

- Heeger, D. (1992) Normalisation of cell responses in cat striate cortex. Vis Neurosci 9: 181-197
- Carandini & Heeger (2012) Normalisation as a canonical neural computation. Nat Rev Neurosci
