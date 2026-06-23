# nfo — Neural Field Oscillator

**Category:** attention  
**Implementation:** `neuroslm.modules.neural_field_oscillator.NeuralFieldOscillator`  
**DSL spec:** [`mechanics/nfo.neuro`](../../mechanics/nfo.neuro)

## Overview

A residual block combining three mechanisms: (1) Kuramoto graph synchronisation of M complex oscillators lifted from the hidden state, (2) Swift-Hohenberg amplitude flow with cubic damping toward a learnable set-point A* (Lyapunov-stable), and (3) coherence-gated write-back where tokens with high local synchrony R_t get amplified. ReZero: write-back matrix W_out is zero-init, so step-0 output is bit-identical to baseline. Exposes Φ_κ (bipartition coherence lower bound) as a free diagnostic.

## Equation

```
z = A·e^{iφ}  lifted from h
# Kuramoto: Δφ_t = Σⱼ K_{ts}·sin(φⱼ − φ_t)  (causal attention coupling)
# Swift-Hohenberg: dA/dt = A·(μ − A²)          (dt=0.1; contraction bound H017: dt≤0.51)
R_t = ‖mean_m(z_{t,m}/|z_{t,m}|)‖              # local coherence
h_out = h + W_out_zero · (R_t/max R · Re(z))   # coherence-gated write
Φ_κ = −log(mean_m|mean_t(e^{iφ_{t,m}})|²)      # IIT Φ lower bound (H015)
```

## Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `n_osc` | `32` | M = oscillators per token |
| `n_steps` | `2` | Euler integration steps |
| `kappa_init` | `0.1` | Kuramoto coupling κ (learnable) |
| `alpha_init` | `0.01` | Gate gain α (learnable; ReZero-like start) |
| `mu_init` | `0.5` | Swift-Hohenberg set-point |
| `expose_phi_lower_bound` | `true` | Log Φ_κ to telemetry |

## When to Use

When you want explicit phase-synchrony dynamics across token positions (binding by synchrony). The coherence gate amplifies in-phase tokens and damps incoherent ones.

## When NOT to Use

- T < 4: insufficient tokens for Kuramoto dynamics
- n_steps > 4: cost grows linearly
- dt > 0.51: violates Swift-Hohenberg contraction bound (H017)

## Properties

- **ReZero contract**: W_out zero-init → baseline-identical at step 0 (H018)
- **Lyapunov-stable**: Swift-Hohenberg contractive (H017)
- **Φ diagnostic**: free lower bound on IIT Φ per step (H015)
- **Information-preserving gate**: H016

## Empirical Evidence

Four verified Lean proofs (H015-H018): Φ lower bound, information-preserving gate, contractive amplitude, zero-init identity. All zero sorry.

## References

- Kuramoto, Y. (1984) Chemical Oscillations, Waves, and Turbulence
- Cross & Hohenberg (1993) Pattern formation. Rev. Mod. Phys. 65
- Singer, W. (1999) Neuronal synchrony. Neuron 24, 49
