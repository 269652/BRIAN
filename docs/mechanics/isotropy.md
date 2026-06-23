# isotropy — Isotropy Whitening

**Category:** regularizer  
**Implementation:** `neuroslm.regularizers.IsotropyLoss`  
**DSL spec:** [`mechanics/isotropy.neuro`](../../mechanics/isotropy.neuro)

## Overview

Isotropy whitening penalises rank collapse in the hidden-state distribution. It accumulates a buffer of hidden states, computes the Gram matrix G = HᵀH/N, and penalises its deviation from the identity. When G ≈ I, all directions in representation space carry equal variance — the representation is isotropic and cannot have degenerate rank-collapsed directions.

## Equation

```
H_buf ∈ ℝ^{N×d}                       # accumulated hidden states
G = H_bufᵀ · H_buf / N               # Gram matrix
L_iso = ‖G − I‖_F² / d²              # Frobenius deviation from identity
L_total += weight · L_iso
```

## Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `weight` | `0.05` | Scale on L_iso (was 0.005; increased 10× in Jun 2026 after erank 53→7 collapse) |
| `buffer` | `4096` | Buffer size for Gram matrix estimation |
| `distance` | `"frobenius"` | Divergence measure; "frobenius" is the standard |

## When to Use

Whenever you observe effective rank (erank) of hidden states collapsing early in training (step ~200-500). Isotropy fires immediately (isotropy_activation_step=0) rather than waiting for warmup, because rank collapse begins before the warmup window closes.

**Critical**: `weight=0.005` was empirically insufficient (contributed ~0.0003 vs LM loss ~5.5). Use `weight≥0.05`.

## When NOT to Use

- weight > 0.1: isotropy can fight the LM objective (encourages representation diversity over task-relevant structure)
- Very small buffers (< 256): Gram matrix estimate is noisy

## Properties

- **Early activation**: fires at step 0 (rank collapse starts ~step 200)
- **Gram-matrix based**: directly targets the spectral distribution
- **GIF-3 scheduled**: can be combined with GIF maturity schedule

## Empirical Evidence

Mu & Viswanath (2018): isotropic word vectors are better OOD representatives. Internal: erank collapsed 53→7 by step 300 without isotropy; with `weight=0.05`, collapse slowed.

## References

- Mu, J. & Viswanath, P. (2018) All-But-The-Top: Simple Postprocessing for Word Representations. ICLR
- Internal: `neuroslm/regularizers.py::IsotropyLoss`
