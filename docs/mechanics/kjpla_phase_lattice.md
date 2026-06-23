# kjpla_phase_lattice — Kuramoto-Josephson Phase Lattice Attention

**Category:** attention  
**Implementation:** `neuroslm.mechanisms.kjpla.KJPLAttention`  
**Loss:** `neuroslm.mechanisms.kjpla.josephson_loss`  
**DSL spec:** [`mechanics/kjpla_phase_lattice.neuro`](../../mechanics/kjpla_phase_lattice.neuro)

## Overview

KJPLA is a phase-coherence attention mechanism combining Kuramoto synchronisation (within a layer) with Josephson-junction coupling (between layers). Content-phase carrier φ₀ is read from the hidden state; intra-layer Kuramoto dynamics synchronise nearby tokens; Josephson coupling pushes adjacent layers toward a shared phase. A phase-gated attention logit adds to the standard QK logit. The Josephson loss penalises decoherence; a phase-entropy floor prevents mode collapse. ReZero discipline: step-0 output is bit-identical to vanilla attention.

## Equation

```
φ₀ = angle(W_phase · h)                    # content carrier
φ̇ᵢ = Σⱼ Kᵢⱼ·sin(φ₀ⱼ − φ₀ᵢ)             # Kuramoto sync
logits += γ·cos(φ_sync_i − φ_sync_j)       # phase-gated attention bias
R_ℓ = |⟨e^{iφ}⟩| ∈ [0,1]                  # Josephson order parameter
L_J = josephson_strength·(1 − R_ℓ)²        # decoherence penalty
L_H = entropy_strength·max(0, eps_H − H(φ)) # phase entropy floor
```

## Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `josephson_strength` | `0.0` | Weight on decoherence penalty; 0 = diagnostic-only |
| `entropy_strength` | `0.0` | Weight on phase-entropy floor loss |
| `eps_H` | `0.5` | Minimum acceptable phase entropy |

## When to Use

Diagnostic mode first (all strengths = 0): observe R_ℓ and H(φ) distributions. Enable josephson_strength > 0 only once the baseline R_ℓ distribution shows systematic decoherence (R_ℓ < 0.3 sustained over >500 steps). Part of the THSD three-mechanism program (Phase 3).

## When NOT to Use

- josephson_strength > 0.1 without characterising baseline: phase loss dominates CE
- Very short contexts: insufficient tokens for Kuramoto dynamics to converge

## Properties

- **ReZero contract**: bit-identical to vanilla attention at step 0 (H018 analogue)
- **Diagnostic mode**: all losses at 0 with strengths = 0; R_ℓ and H(φ) logged
- **Josephson analogy**: R_ℓ = superconducting order parameter (BCS/Josephson 1962)

## Empirical Evidence

18-contract test suite: `tests/dsl/test_kjpla_dsl_parse.py` (all passing). Attention tests in `tests/dsl/test_kjpla_attention.py`.

## References

- Kuramoto, Y. (1984) Chemical Oscillations, Waves, and Turbulence
- Josephson, B. (1962) Possible new effects in superconductive tunnelling. Physics Letters
- Internal: `neuroslm/mechanisms/kjpla.py`, `tests/dsl/test_kjpla_dsl_parse.py`
