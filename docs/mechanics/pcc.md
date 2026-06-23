# pcc — Predictive Contrastive Coding

**Category:** regularizer  
**Implementation:** `neuroslm.regularizers.PCCLoss`  
**DSL spec:** [`mechanics/pcc.neuro`](../../mechanics/pcc.neuro)

## Overview

PCC applies an InfoNCE contrastive loss over future token representations. For each position t, the positive sample is the hidden state at t+k steps ahead; negatives are hidden states from other documents in a cross-doc buffer. This forces the encoder to produce representations that are predictive across time — a form of self-supervised OOD regularisation.

## Equation

```
h_pos = h_{t+k}                      # future state (positive)
h_neg = buffer[random sample]        # cross-document negatives
L_InfoNCE = −log[ exp(h_t·h_pos/τ) / (exp(h_t·h_pos/τ) + Σ_neg exp(h_t·h_neg/τ)) ]
```

## Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `k` | `4` | Future step for positive sample |
| `n_negatives` | `64` | Number of cross-doc negative samples |
| `tau` | `0.1` | Temperature for InfoNCE |
| `layers` | `[]` | Which layers to apply PCC at (empty = all) |
| `weight` | `0.1` | Scale on L_InfoNCE |

## When to Use

When representations are not temporally coherent — i.e. the model learns token-level features but not sentence/paragraph-level structure. PCC adds a temporal prediction objective that forces multi-step coherence.

## When NOT to Use

- Very small batches (< 16): insufficient negatives for reliable contrastive signal
- k > seq_len/4: positive samples fall outside the context window
- tau < 0.05: contrastive loss becomes too hard (uniform gradient over negatives)

## Properties

- **Cross-doc negatives**: avoids same-document false negatives
- **Temperature controlled**: tau smoothes the contrast
- **Layer-specific**: can apply at any subset of layers

## Empirical Evidence

van den Oord et al. (2018) CPC: contrastive predictive coding improves speech representations; Ozbulak et al. (2023) extend to language.

## References

- van den Oord, A. et al. (2018) Representation Learning with Contrastive Predictive Coding. arXiv:1807.03748
