# mla — Summary

**Category:** attention  
**DSL spec:** [`mechanics/mla.neuro`](../../mechanics/mla.neuro)

## Overview

Caches a small joint latent for K and V (low-rank), with a separate decoupled RoPE branch, slashing KV-cache size while keeping MHA-grade quality. The attention design used in DeepSeek-V2/V3.

## Equation

`c^{KV}_t = W^{DKV} · h_t` (low-rank latent, the only KV cache entry).  
Keys = `[W^{UK} · c^{KV}_t ; RoPE(W^{KR} · h_t)]`; Values = `W^{UV} · c^{KV}_t`.  
Cache footprint per token = `kv_lora_rank + rope_head_dim` (vs `2 · n_h · d_h` for MHA).

## Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `kv_lora_rank` | 512 | Dimension of the shared compressed KV latent (dominant term in cache size) |
| `q_lora_rank` | 1536 | Rank of the query down-projection (reduces query-path params; not cached) |
| `rope_head_dim` | 64 | Dimension of the decoupled RoPE branch carried per token |

## When to Use / When NOT to Use

**Use when:** both small KV cache AND MHA-grade quality are needed, beyond what GQA gives; long-context decoder serving where cache memory dominates; large models where the matrix-absorption trick recovers MQA-like decode speed.

**Avoid when:** small models where extra projection machinery outweighs the cache win; encoder/bidirectional settings; implementations that cannot afford up-projection / absorption bookkeeping.

## References

- DeepSeek-AI (2024) DeepSeek-V2: A Strong, Economical, and Efficient Mixture-of-Experts Language Model. arXiv 2405.04434
- DeepSeek-AI (2024) DeepSeek-V3 Technical Report. arXiv 2412.19437
