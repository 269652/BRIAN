# liouville_symplectic — Hamiltonian-Preserving Residual

**Category:** physics  
**Implementation:** `neuroslm.mechanisms.liouville_symplectic.LiouvilleSymplecticBlock`  
**DSL spec:** [`mechanics/liouville_symplectic.neuro`](../../mechanics/liouville_symplectic.neuro)

## Overview

Phase 2 of the THSD research program. Splits the hidden state into canonical coordinates (q, p) and runs one Stoermer-Verlet leapfrog step on a learned Hamiltonian H(q,p). The Noether residual (H_final − H_initial)² measures energy non-conservation; at noether_strength=0 this is purely diagnostic. The leapfrog construction guarantees det(Jacobian)=1 (Liouville's theorem — volume-preserving dynamics).

## Equation

```
[q, p] = split(x)
KE(p) = ½‖M^{-1/2}p‖²,  V(q) = ½‖Aq‖² (or SwiGLU)
# Stoermer-Verlet leapfrog:
p_{½} = p − (τ/2)·∇_q H(q, p)
q₁    = q + τ·M^{-1}·p_{½}
p₁    = p_{½} − (τ/2)·∇_q H(q₁, p_{½})
L_Noether = (H(q₁,p₁) − H(q,p))²
```

## Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `noether_strength` | `0.0` | Weight on L_Noether; 0 = diagnostic-only |
| `dtau_init` | `0.1` | Leapfrog stride τ (learnable) |
| `potential_kind` | `"quadratic"` | `"quadratic"` or `"swiglu"` |
| `w_rank` | `4` | Rank of LowRankPairwise W(q) potential |

## When to Use

Run in diagnostic mode first (noether_strength=0): log H_diff to verify it's O(τ²). Enable penalty only after baseline is characterised. Part of THSD Phase 2 — pairs with pontryagin_topo_charge (Phase 1) and kjpla_phase_lattice (Phase 3).

## When NOT to Use

- L > 16: leapfrog accumulates error across many layers
- noether_strength > 0.1: Noether loss dominates CE

## Properties

- **Symplectic by construction**: leapfrog det(J)=1 guaranteed
- **O(τ²) bound**: Hairer-Wanner theorem; L_Noether ≈ 0 in exact arithmetic
- **Type-enforced**: QOnlyPotential ABC prevents p from leaking into V(q)

## Empirical Evidence

H25: diagnostic mode confirms H_diff ≈ O(τ²). Formal proof: `hypothesis/proofs/H025_liouville_det1.lean` (pending).

## References

- Hairer, Lubich, Wanner (2006) Geometric Numerical Integration. Springer
- Greydanus, Dzamba, Yosinski (2019) Hamiltonian Neural Networks. NeurIPS
- Internal: `neuroslm/mechanisms/liouville_symplectic.py`
