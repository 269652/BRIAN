# tonnetz_attention — Toroidal Harmonic Position Masking

**Category:** attention  
**Implementation:** `neuroslm.dsl.nn_ops.causal_self_attention_tonnetz`  
**DSL spec:** [`mechanics/tonnetz_attention.neuro`](../../mechanics/tonnetz_attention.neuro)

## Overview

Tonnetz Attention adds a toroidal harmonic bias to the attention mask. Positions that are harmonically related (separated by multiples of a musical period P) receive a bonus; positions outside a local window are masked. The mechanism models the cyclic structure of rhythm and harmonic repetition — useful for music, poetry, and any domain with periodic long-range dependencies.

## Equation

```
circ_dist(i, j) = min(|i−j| mod P, P − |i−j| mod P)
tonnetz_mask[i,j] = exp(−circ_dist(i,j)² / (2·bandwidth²)) − 1{|i−j| > local_window}
logits = causal_logits + tonnetz_mask
```

## Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `tonnetz_period` | `12` | Period P of the harmonic cycle (12 = Western chromatic scale) |
| `local_window` | `12` | Max local-causal window (positions beyond this are masked in non-harmonic positions) |

## When to Use

When the input domain has periodic structure (music, poetry with regular meter, code with syntactic repetition). The period P should match the domain's natural cycle (12 for chromatic harmony, 4 for common time rhythm, n for n-gram repetition cycles).

## When NOT to Use

- Prose text without cyclic structure: random softmax penalty
- P much larger than typical sequence length: no harmonic positions ever coincide
- When combined with RoPE torus (rope_torus): potential double-counting of periodic structure

## Properties

- **Circular distance**: computes shortest arc on the period torus
- **Gaussian envelope**: smooth falloff from harmonic positions
- **Causal preserved**: local_window mask enforces causality

## Empirical Evidence

Tonnetz space (Euler 1739; Balzano 1980) encodes music-theoretic harmonic relationships. Neural Tonnetz models (Hadjeres & Pachet 2017) exploit this for music generation.

## References

- Balzano, G. (1980) The group-theoretic description of 12-fold and microtonal pitch systems. CMJ
- Internal: `neuroslm/dsl/nn_ops.py`
