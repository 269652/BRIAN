# mqa — Summary

**Category:** attention  
**DSL spec:** [`mechanics/mqa.neuro`](../../mechanics/mqa.neuro)

## Overview

All query heads attend to one shared key/value head, collapsing the KV cache to a single head for maximum decode speed. The `kv_heads = 1` limit of GQA; in practice GQA (kv_heads ≈ 4–8) usually dominates on the quality/speed frontier.

## Equation

`Q ∈ (B, n_heads, T, d_k)`, `K, V ∈ (B, 1, T, d_k)`.  
Broadcast: `K̂ = K.expand(B, n_heads, T, d_k)`.  
Standard scaled-dot-product attention per query head on the broadcast K̂, V̂.  
KV cache is `n_heads×` smaller than MHA.

## Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `n_heads` | 32 | Number of query heads (all share the single KV head) |

## When to Use / When NOT to Use

**Use when:** decode throughput / KV-cache memory is the dominant constraint and a quality trade-off is accepted; extreme batch-size serving where per-sequence KV cache must be tiny; as the kv_heads=1 endpoint of a GQA sweep.

**Avoid when:** quality-critical training from scratch (GQA almost always wins); small head counts (n_heads ≤ 4); models showing training instability under a single shared KV head.

## References

- Shazeer (2019) Fast Transformer Decoding: One Write-Head is All You Need. arXiv 1911.02150
- Ainslie, Lee-Thorp, de Jong, Zemlyanskiy, Lebrón, Sanghai (2023) GQA. arXiv 2305.13245
