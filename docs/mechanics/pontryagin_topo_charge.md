# pontryagin_topo_charge — Skyrmion Topological-Charge Diagnostic

**Category:** physics  
**Implementation:** `neuroslm.mechanisms.topo_charge.PontryaginTopoCharge`  
**DSL spec:** [`mechanics/pontryagin_topo_charge.neuro`](../../mechanics/pontryagin_topo_charge.neuro)

## Overview

Phase 1 of the THSD research program. Projects attention-head outputs onto the unit sphere S² and computes the Berg-Lueschner discrete Pontryagin (skyrmion) topological charge Q_h using the numerically-stable van Oosterom-Strang atan2 formula. Also computes the Hopfion-lite inter-layer decorrelation diagnostic eps_ortho. At alpha=gamma=0, purely diagnostic (zero added to loss).

## Equation

```
n_h = F.normalize(h_head[..., :3])      # project to S²
# van Oosterom-Strang solid angle per triangle:
Ω = 2·atan2(n_a·(n_b×n_c), 1 + n_a·n_b + n_b·n_c + n_c·n_a)
Q_h = (1/4π)·Σ_{triangles} Ω          # topological charge
L_Q = α·Σ_h (Q_h − round(Q_h))²       # soft penalty toward ℤ
eps_ortho = Σ_{ℓ} mean(1 − n_{ℓ+1}·n_ℓ)  # inter-layer decorrelation
```

## Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `alpha` | `0.0` | Weight on L_Q (penalty toward integer Q); 0 = diagnostic |
| `gamma` | `0.0` | Weight on inter-layer decorrelation target |
| `Q_target` | `0.0` | Target for diversity penalty |
| `weight_init_std` | `0.02` | Std for any learnable projection weights |

## When to Use

Run diagnostic first: log Q_h and eps_ortho distributions over 1000+ steps. Healthy: Q_h ≈ 0. High |Q_h| ≥ 1 indicates topologically non-trivial attention maps. Enable alpha > 0 to push toward trivial topology (or toward Q=1 for structured maps).

## When NOT to Use

- d_head < 3: cannot project to S²
- T < 3: no triangles to sum
- alpha > 0.1: dominates CE

## Properties

- **Diagnostic at zero**: logs Q_h and eps_ortho; zero added to loss at alpha=gamma=0
- **atan2 stable**: avoids acos antipodal degeneracy
- **THSD Phase 1**: pairs with liouville_symplectic (Phase 2) and kjpla (Phase 3)

## Empirical Evidence

H23: diagnostic mode confirms Q_h ≈ O(ε) at init. Formal proof: `hypothesis/proofs/H023_pontryagin_homotopy.lean` (pending).

## References

- Berg & Lueschner (1981) Nuclear Physics B 190(3): 412-424
- van Oosterom & Strang (1983) IEEE Trans. Biomed. Eng. 30(2): 125-126
- Internal: `neuroslm/mechanisms/topo_charge.py`
