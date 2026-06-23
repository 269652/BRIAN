# riemannian_motor_projection — Hyperbolic Tanh Projection

**Category:** training_dynamics  
**Implementation:** `neuroslm.harness.BrainHarness._riemannian_motor_project`  
**DSL spec:** [`mechanics/riemannian_motor_projection.neuro`](../../mechanics/riemannian_motor_projection.neuro)

## Overview

Projects the motor hidden state onto the Poincaré ball via a differentiable tanh mapping: `h_proj = ρ·tanh(‖h‖/ρ)·h/‖h‖`. This geometrically caps the magnitude of h_motor to ρ=1/√R (learnable radius) while being nearly-identity for small ‖h‖. Composes with VBB: the posterior encoder operates on h_proj, preventing numerically difficult large-activation inputs.

## Equation

```
ρ = 1/√(softplus(R_raw))          # learnable ball radius
h_proj = ρ · tanh(‖h‖/ρ) · h/‖h‖
‖h_proj‖ < ρ  always
```

## Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `motor_curvature` | `1.0` | Initial curvature R; ρ_init = 1/√motor_curvature |

## When to Use

When h_motor can grow large activations that destabilise the VBB posterior. motor_curvature=0.01 gives ρ=10 (loose); motor_curvature=1.0 gives ρ=1 (tight — typical after LayerNorm with ‖h‖≈22).

## When NOT to Use

- h_motor is already bounded (hard tanh, clamp)
- motor_curvature very large (ρ << 1): maps nearly all h_motor to ball boundary

## Properties

- **Strict bound**: ‖h_proj‖ < ρ always
- **Differentiable**: C∞ everywhere
- **Learnable radius**: ρ adapts during training
- **Identity near zero**: tanh(x) ≈ x for ‖h‖ << ρ

## References

- Nickel & Kiela (2017) Poincaré Embeddings. NeurIPS
- Ganea et al. (2018) Hyperbolic Neural Networks. NeurIPS
