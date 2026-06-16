---
code_refs:
  - neuroslm/emergent/gif.py (compute_head_diversity_loss, GIFController.head_diversity_weight)
  - neuroslm/dsl/nn_ops.py (_DIVERSITY_STASH)
  - neuroslm/harness.py (GIF-5 stash activation + loss wiring)
  - lib/gif.neuro (gif_head_diversity_loss, gif_head_diversity_weight)
created_at: "2026-06-16T22:00:00Z"
id: H009
proof_path: null
proof_status: missing
references:
  - docs/OOD_MECHANISMS.md
  - Li et al. (2018) — Multi-Head Attention with Disagreement Regularization
  - Voita et al. (2019) — Analyzing Multi-Head Self-Attention
status: stated
tags: [gif, ood, attention, diversity, cosine-similarity, encoder]
test_refs:
  - tests/dsl/test_gif_head_diversity.py
theorem_name: Brian.GIFHeadDiversityReducesRedundancy
title: "GIF-5: Attention head diversity loss reduces encoder-side redundancy"
updated_at: "2026-06-17T00:00:00Z"
---

## H009 — GIF-5: Attention head diversity loss reduces encoder-side redundancy

### Statement

Let $Q^{(\ell)} \in \mathbb{R}^{B \times H \times T \times d_h}$ be the
post-RoPE query tensor at layer $\ell$, and let $q_i = \text{flatten}(Q^{(\ell)}_{:,i,:,:})$.
Define the head diversity loss:

$$L_\text{div} = \frac{1}{L} \sum_{\ell=1}^{L} \frac{2}{H(H-1)} \sum_{i < j} \left[\max\!\left(0,\; \cos(q_i, q_j)\right)\right]^2$$

with gap-reactive weight:

$$w_\text{div} = w_0 \cdot \ln\!\left(1 + \frac{G(t)}{G^*}\right)$$

**Claim:** Adding $w_\text{div} \cdot L_\text{div}$ to the training loss
reduces the mean pairwise cosine similarity between attention heads by
≥30% at step 5000, and OOD PPL improves by ≥10% vs GIF-1/2/3/4 alone.

### Mechanism

GIF-4 hits a ceiling because it attacks only the **output side**
(distribution sharpness). Meanwhile, attention heads converge to
identical FineWeb-Edu co-occurrence detectors — **encoder-side
redundancy**. When all heads attend to the same features, the
representation capacity is wasted on domain-specific patterns
that don't generalise.

The diversity loss penalises positively-correlated heads (cos > 0)
while leaving anti-correlated heads alone (they're already diverse).
Squaring concentrates gradient on near-identical heads.

**Implementation note:** Q tensors are stashed (with grad) during the
forward pass via `nn_ops._DIVERSITY_STASH`. When `None` (default),
no stashing occurs — zero overhead. Activated by the harness only
when `attn_div_weight > 0` and `head_diversity_weight > 0`.

### Ablation protocol

| Variant | GIF-4 | GIF-5 | w₀ | Expected |
|---------|-------|-------|----|----------|
| GIF-4 only | ✓ | ✗ | 0 | Gap rebounds to ~2.8 |
| GIF-5 low | ✓ | ✓ | 0.005 | Mild diversity pressure |
| GIF-5 mid | ✓ | ✓ | 0.01 | Gap ~2.3 (deployed) |
| GIF-5 high | ✓ | ✓ | 0.05 | Strong diversity, possible train PPL hit |
| GIF-5 no gap-react | ✓ | static=0.01 | — | Active from step 0, may hurt early training |

### Key commits

- `f9baf0b` — GIF-5: attention head diversity loss
- `7523d59` — dsl: add GIF-5 equations to lib/gif.neuro

### Config

```neuro
gif: {
  attn_div_weight: 0.01
}
```

### Empirical evidence

- Instance 41238415 (commit `7523d59`): GIF-5 active but log lines
  truncated at 515 chars, so `div=` telemetry not visible. Gap trajectory:
  - Step 3000: gap=1.77, OOD=215
  - Step 5000: gap=2.36, OOD=189
  - Step 6500: gap=2.53, OOD=177
  Gap still rising (train PPL falling faster than OOD PPL), suggesting
  GIF-5's weight (~0.009) is too small relative to LM loss (~4.2).
  This motivated GIF-6 (structural fix instead of aux loss).
