# freq_balance — Frequency-Balanced Cross-Entropy

**Category:** regularizer  
**Implementation:** `neuroslm.regularizers.FreqBalanceLoss`  
**DSL spec:** [`mechanics/freq_balance.neuro`](../../mechanics/freq_balance.neuro)

## Overview

Reweights per-token CE loss by the ratio of OOD-distribution unigram frequency to training-distribution unigram frequency (raised to exponent β). Tokens that are common in the OOD target but rare in training receive higher weight, biasing the gradient toward OOD-relevant tokens.

## Equation

```
ratio[v] = (freq_ood[v] / freq_train[v])^β
w[v]     = clip(ratio[v], w_min, w_max) / mean(clip(ratio, w_min, w_max))
L_freq   = mean(w[target_ids] · CE_per_token)
```

## Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `beta` | `0.5` | Exponent; 0=uniform, 0.5=sqrt-smooth, 1.0=exact inverse freq |
| `w_min` | `0.2` | Minimum weight floor |
| `w_max` | `5.0` | Maximum weight ceiling |

## When to Use

When you have pre-computed unigram frequency tables for both training and OOD distributions and they differ significantly. Requires a one-time corpus scan.

## When NOT to Use

- No pre-computed frequency tables
- OOD distribution unknown at training time
- Very small vocabulary (< 1000): frequency estimates are noisy

## Properties

- **Mean-normalised**: average loss scale unchanged
- **No extra parameters**: pure loss reweighting
- **Beta=0.5**: Mikolov (2013) subsampling analogue

## Empirical Evidence

Mikolov et al. (2013): subsampling by inverse frequency improves word embedding quality. Internal: not yet ablated (requires corpus scan first).

## References

- Mikolov et al. (2013) Distributed Representations of Words and Phrases. NIPS
- Internal: `neuroslm/regularizers.py::FreqBalanceLoss`
