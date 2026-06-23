# alibi — Summary

**Category:** position  
**DSL spec:** [`mechanics/alibi.neuro`](../../mechanics/alibi.neuro)

## Overview

Adds a head-specific linear penalty proportional to query-key distance directly to attention logits. Slopes form a geometric sequence per head so different heads cover different distance scales — no learned parameters required.

## Equation

`scores_{i,j} = (q_i · k_j) / √d_k − m_h · (i − j)`  
where `m_h = 2^(−8h / n_heads)` (per-head slope). Softmax is applied as usual.

## Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `n_heads` | 12 | Number of attention heads; sets the geometric slope schedule |
| `max_slope` | 0.5 | Anchor for the steepest head's slope (m_1); smaller = softer global penalty |

## When to Use / When NOT to Use

**Use when:** training on short sequences but evaluating on longer ones; zero-parameter position scheme is needed; recency/locality bias is desirable.

**Avoid when:** precise long-range absolute/relative indexing is needed; bidirectional encoders; mixing with RoPE on the same head dims.

## References

- Press, Smith, Lewis (2021) Train Short, Test Long: Attention with Linear Biases Enables Input Length Extrapolation. arXiv 2108.12409
- Su, Lu, Pan, Wen, Liu (2021) RoFormer (RoPE comparison baseline). arXiv 2104.09864
