# fisher_rao_retrieval — Precision-Weighted Similarity

**Category:** training_dynamics  
**Implementation:** `neuroslm.thsd.engine.CellularSheaf`  
**DSL spec:** [`mechanics/fisher_rao_retrieval.neuro`](../../mechanics/fisher_rao_retrieval.neuro)

## Overview

Replaces Euclidean or cosine similarity with the Fisher-Rao information metric — a context-dependent inner product weighted by a learned precision matrix Σ⁻¹(h). High-precision dimensions (strong signal) contribute more; low-precision dimensions (noisy) are down-weighted. Provides the metric structure for THSD sheaf stalks.

## Equation

```
Σ⁻¹(h) = diag(softplus(W_prec · h))     # diagonal precision (context-dependent)
⟨h_a, h_b⟩_F = h_aᵀ · Σ⁻¹(h) · h_b
sim_FR = ⟨h_a, h_b⟩_F / (√⟨h_a,h_a⟩_F · √⟨h_b,h_b⟩_F)
```

## Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `precision_rank` | `-1` | -1=diagonal, 0=scalar, k=rank-k |
| `precision_init` | `1.0` | Initial precision (1.0 = Euclidean at init) |

## When to Use

In the THSD framework where each brain-region stalk requires a Riemannian metric. In retrieval contexts where representation dimensions have heterogeneous signal-to-noise ratios.

## When NOT to Use

- Very early training (< 1000 steps): precision matrix is unstable
- Diagonal rank only handles axis-aligned anisotropy; full matrix needed for correlated dimensions

## Properties

- **Geometry-aware**: accounts for manifold curvature
- **Context-dependent**: different contexts → different metrics
- **Euclidean at init**: precision_init=1.0 → standard inner product

## References

- Rao, C. R. (1945) Information and accuracy attainable. Bull. Calcutta Math. Soc.
- Amari, S. (1985) Differential-Geometrical Methods in Statistics. Springer
- Internal: `neuroslm/thsd/engine.py`
