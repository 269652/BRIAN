# semantic_turbulence — Semantic Turbulence Engine

**Category:** attention  
**Implementation:** `neuroslm.emergent.semantic_turbulence.SemanticTurbulenceEngine`  
**DSL spec:** [`mechanics/semantic_turbulence.neuro`](../../mechanics/semantic_turbulence.neuro)

## Overview

Three physics-inspired mechanisms targeting 2-4× OOD PPL reduction: (1) Renormalisation Group (RG) cascade with Kolmogorov 5/3-law multi-scale attention; (2) Gross-Pitaevskii Equation (GPE) layer modelling the hidden state as a superfluid with a semantic coherence order parameter ρ; (3) NT Criticality Monitor driving the layer-to-layer branching ratio σ toward the critical point σ=1 (Beggs & Plenz 2003).

## Equations

**Module 1 (RG)**: `λ_g = 2^{-5g/6}` coupling; multi-scale coarse/fluctuation decomposition.  
**Module 2 (GPE)**: `ψ ← ψ − Δτ·(−∇²ψ/2 + g|ψ|²ψ)`, order parameter `ρ = |⟨ψ/|ψ|⟩|²`.  
**Module 3 (criticality)**: `σ = mean_ℓ(‖∂h_{ℓ+1}/∂h_ℓ‖_F)`, `L = weight·(σ−σ*)²`.

## Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `n_rg_groups` | `3` | G = number of RG scale groups |
| `kolmogorov_init` | `true` | Init λ_g ∝ 2^{-5g/6} |
| `gpe_steps` | `4` | GPE integration steps |
| `gpe_coupling_init` | `0.01` | Initial self-interaction g |
| `criticality_target` | `1.0` | σ* |
| `criticality_weight` | `0.01` | Weight on (σ−σ*)² |
| `criticality_da_reward` | `0.1` | DA increment at σ≈σ* |
| `rho_gate_enabled` | `true` | Gate GPE write-back by ρ |

## When to Use

Multi-scale attention enrichment + coherence diagnostics + criticality regulation. Requires NT harness for the full criticality feedback loop.

## When NOT to Use

- Seq < 2^{n_rg_groups} tokens (coarsest group empty)
- criticality_weight > 0.1 (dominates CE early)
- Without NT harness (DA/NE/GABA feedback silent)

## Properties

- **Three independent modules**: RG + GPE + criticality can be individually disabled
- **Kolmogorov init**: energy cascade scale-free prior
- **Superfluid analogy**: ρ=1 = semantically unambiguous, ρ=0 = polysemous

## Empirical Evidence

H-STE hypothesis: 2-4× OOD PPL reduction over same-parameter baseline. Full STE stack ablation ongoing.

## References

- Kolmogorov (1941) Local structure of turbulence. Proc. USSR Acad. Sci.
- Gross (1961), Pitaevskii (1961) BEC mean-field equation
- Beggs & Plenz (2003) Neuronal avalanches. J Neurosci
