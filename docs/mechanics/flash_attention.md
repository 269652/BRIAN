# flash_attention — Summary

**Category:** attention  
**DSL spec:** [`mechanics/flash_attention.neuro`](../../mechanics/flash_attention.neuro)

## Overview

Exact attention computed in SRAM tiles with online softmax — never materializes the N×N score matrix. Bit-for-bit equivalent to standard softmax attention with O(N) memory instead of O(N²).

## Equation

Standard `O = softmax(Q Kᵀ / √d) V` computed via tiled blocks. For each query block i over key/value blocks j, running max `m` and sum `l` are maintained online. The N×N matrix is never written to HBM; backward pass recomputes it from stored softmax statistics.

## Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `block_size_q` | 128 | Rows of Q processed per tile (B_q). Larger = more SRAM, fewer passes |
| `block_size_kv` | 64 | Rows of K/V processed per tile (B_kv). Tuned to fit SRAM with B_q |
| `causal` | true | Apply causal mask; fully-masked KV blocks are skipped entirely |

## When to Use / When NOT to Use

**Use when:** sequence length is long (T ≥ 1024); training is HBM-bandwidth bound; exact attention is needed with lower memory; targeting modern GPUs (A100, H100).

**Avoid when:** CPU-only or memory-flat hardware; very short sequences (T ≤ 128); interpretability probes needing the explicit N×N matrix; naïve pure-PyTorch eager reimplementation.

## References

- Dao, Fu, Ermon, Rudra, Ré (2022) FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness. arXiv 2205.14135
- Dao (2023) FlashAttention-2: Faster Attention with Better Parallelism and Work Partitioning. arXiv 2307.08691
- Milakov, Gimelshein (2018) Online normalizer calculation for softmax. arXiv 1805.02867
