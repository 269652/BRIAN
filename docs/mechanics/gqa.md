# gqa — Summary

**Category:** attention  
**DSL spec:** [`mechanics/gqa.neuro`](../../mechanics/gqa.neuro)

## Overview

Interpolates between MHA and MQA: `kv_heads` key/value heads are shared across `n_heads` queries, shrinking the KV cache by `kv_heads/n_heads`. The modern default for large decoder LMs (LLaMA-2/3, Mistral, Qwen).

## Equation

`group_size = n_heads / kv_heads`  
`K̂ = repeat_interleave(K, group_size, dim=heads)` (broadcast each KV head to its group of queries)  
`A_h = softmax((Q_h K̂_hᵀ) / √d_k)`, `out_h = A_h V̂_h`

## Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `n_heads` | 32 | Number of query heads |
| `kv_heads` | 4 | Number of shared key/value heads; must divide n_heads |

## When to Use / When NOT to Use

**Use when:** deploying a decoder LM where KV-cache memory and decode bandwidth are the bottleneck; most of MHA's quality is needed at a fraction of the cache cost.

**Avoid when:** tiny models where n_heads is already small (≤4); maximum per-head KV diversity is needed; kv_heads does not divide n_heads.

## References

- Ainslie, Lee-Thorp, de Jong, Zemlyanskiy, Lebrón, Sanghai (2023) GQA: Training Generalized Multi-Query Transformer Models from Multi-Head Checkpoints. arXiv 2305.13245
- Shazeer (2019) Fast Transformer Decoding: One Write-Head is All You Need. arXiv 1911.02150
