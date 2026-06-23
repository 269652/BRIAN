# episodic_memory — kNN Episodic Memory Cache

**Category:** attention  
**Implementation:** `neuroslm.modules.differentiable_memory.EpisodicMemory`  
**DSL spec:** [`mechanics/episodic_memory.neuro`](../../mechanics/episodic_memory.neuro)

## Overview

An external key-value episodic memory bank (Memorizing Transformers / RETRO style). Hidden states are written to a circular buffer; at each step, k-nearest neighbours are retrieved by cosine similarity and blended into the residual via a learnable zero-init scalar α. The ReZero discipline (alpha_init=0) means step-0 output is bit-identical to baseline.

## Equation

```
# Write
K_mem[tail] = q_t   (circular FIFO buffer of 'slots' entries)

# Read
scores  = q_t · K_mem.T  (cosine similarity)
top_k   = argtopk(scores, k)
attn    = softmax(scores[top_k] / √d_key)
retrieved = attn @ V_mem[top_k]

# Write-back (ReZero)
h_out = h + α_mem · retrieved   (α_mem initialised to alpha_init=0)
```

## Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `slots` | `4096` | Maximum entries in the memory bank |
| `k` | `32` | Nearest neighbours to retrieve |
| `alpha_init` | `0.0` | Initial blend scalar (ReZero start) |
| `write_gate` | `"all"` | `"all"` or `"surprise"` |
| `write_quantile` | `0.8` | Quantile threshold for surprise gate |

## When to Use

When you want explicit episodic memory of past contexts. alpha_init=0 ensures the memory starts silent; the gradient lifts α off zero as retrieval helps. write_gate='all' is the safe Phase-5 baseline.

## When NOT to Use

- Very short context runs (insufficient history)
- alpha_init > 0 without warmup (retrieval noise corrupts early training)
- slots > 65536 (use FAISS or PQ for very large banks)

## Properties

- **ReZero contract**: alpha_init=0 → baseline-identical at step 0
- **FIFO buffer**: oldest memories evicted as new ones arrive
- **Differentiable**: retrieval weights softmax-differentiable
- **Surprise gating**: writes only high-surprise moments

## References

- Wu et al. (2022) Memorizing Transformers. ICLR
- Borgeaud et al. (2022) RETRO. ICML
