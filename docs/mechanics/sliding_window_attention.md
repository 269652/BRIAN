# sliding_window_attention — Summary

**Category:** attention  
**DSL spec:** [`mechanics/sliding_window_attention.neuro`](../../mechanics/sliding_window_attention.neuro)

## Overview

Restricts each query to a fixed-size local window of recent keys, giving O(T · w) attention cost and a bounded KV cache. Stacking L layers yields an effective receptive field of ≈ L · window_size (CNN-like growth).

## Equation

Allowed keys for query i: `j ∈ [max(0, i − w + 1), i]`.  
`mask_{i,j} = 0` if `(i − w) < j ≤ i`, else `−∞`.  
Standard scaled-dot-product attention on the unmasked window.

## Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `window_size` | 4096 | Past tokens (inclusive of self) each query may attend to; caps both compute and KV cache |

## When to Use / When NOT to Use

**Use when:** sequences are long and most useful context is local; bounded constant-size KV cache for streaming inference is needed; model has sufficient depth for L · window_size to cover required dependencies.

**Avoid when:** critical long-range dependencies exceed L · window_size; very shallow models with a small effective receptive field; exact global attention is required.

## References

- Beltagy, Peters, Cohan (2020) Longformer: The Long-Document Transformer. arXiv 2004.05150
- Jiang, Sablayrolles, Mensch et al. (2023) Mistral 7B. arXiv 2310.06825
