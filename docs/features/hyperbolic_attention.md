# Hyperbolic (Poincaré-Disc) Multi-Head Attention

**Feature:** `@brian/features/hyperbolic_attention`
**Implementation:** `neuroslm/modules/hyperbolic_attention.py`
**Contract tests:** `tests/test_hyperbolic_attention.py` (18 invariants)
**Default state:** `active: false` (clean baseline)

## What it does

Multi-head attention whose query/key vectors live on the **Poincaré
ball** $\mathbb{D}^n_c$ — the open ball of radius $1/\sqrt{c}$ equipped
with the conformal metric

$$
g^c_x = \lambda_c(x)^2\, g^E,
\qquad
\lambda_c(x) = \frac{2}{1 - c\|x\|^2}.
$$

After linear projection, $Q$ and $K$ are mapped into the ball via the
exponential map at the origin

$$
\exp_0^c(v) = \tanh(\sqrt{c}\,\|v\|)\, \frac{v}{\sqrt{c}\,\|v\|}.
$$

Attention logits are **negative hyperbolic distances** scaled like a
standard transformer:

$$
\text{logit}_{ij} = -\, \frac{d_c(Q_i, K_j)}{\sqrt{d_{\text{head}}}}
\qquad
d_c(x,y) = \tfrac{2}{\sqrt{c}}\,
           \text{artanh}\!\bigl(\sqrt{c}\,\|{-x \oplus_c y}\|\bigr).
$$

where $\oplus_c$ is Möbius vector addition (Ungar 2005). The softmax is
taken over keys; values $V$ stay Euclidean and are combined by the
attention weights in the standard way (Ganea, Bécigneul, Hofmann 2018).

## Why it might help

The Poincaré ball is the *canonical embedding* for tree-structured data
because its volume grows exponentially with radius — perfectly matched
to the number of nodes at depth $r$ in a tree of branching factor
$> 1$. Concretely:

- Code (AST, call graphs, scope chains)
- Natural-language constituency / dependency trees
- File-system / taxonomy hierarchies
- Compositional reasoning (function composition graphs)

Standard dot-product attention assigns similarity based on Euclidean
inner products; hyperbolic attention assigns similarity based on
geodesic distance. The hypothesis is that for hierarchical context the
hyperbolic metric concentrates attention on the right ancestors.

## When it likely won't help

- Pure repetitive sequence data with no hierarchical structure (e.g.
  WikiText short windows). Expect ≈ baseline performance.
- Heads where the data is already "flat" — hyperbolic geometry collapses
  to Euclidean near the origin, so a model can choose to ignore the
  curvature, but training stability may take a hit while it learns to do
  so.

## Activation

In an `arch.neuro`:

```neuro
import { hyperbolic_attention } from "@brian/features/hyperbolic_attention"

# Override the default to turn it on:
feature hyperbolic_attention {
    equation: hyperbolic_attention_eq,
    active:   true,
    impl:     "neuroslm.modules.hyperbolic_attention.HyperbolicMultiHeadAttention",
    params: {
        d_model: 512,
        n_heads: 8,
        c:       1.0
    },
    endpoints: {
        edge: { kind: "edge", inputs: [x_pre], output: y }
    }
}

# Then wire it into a synapse:
synapse cortex_a -> cortex_b {
    feature: "hyperbolic_attention.edge",
    weight: 1.0
}
```

## Mathematical contracts (pinned by tests)

All 18 contracts in `tests/test_hyperbolic_attention.py` must stay
green. The most important ones (per CLAUDE.md §14):

- **`test_projection_lands_strictly_inside_unit_ball`** — for any
  finite Euclidean input, the projected vector has norm $< 1/\sqrt{c}$.
- **`test_left_cancellation_recovers_zero`** —
  $x \oplus_c (-x) = 0$.
- **`test_logmap0_inverts_expmap0`** — round-trips at machine precision.
- **`test_distance_from_origin_matches_closed_form`** — at $c=1$,
  $d(0,y) = 2\,\text{artanh}(\|y\|)$ to 1e-5.
- **`test_attention_weights_concentrate_on_nearby_keys`** — given a
  query and two keys (one close, one far in hyperbolic distance), the
  near key receives **strictly more** attention weight than the far
  key. This is the only test that asserts the mechanism *does what it
  says* — every other test asserts numerical / structural invariants.
- **`test_gradient_flows_through_attention`** — every learnable
  parameter receives a non-zero, finite gradient.
- **`test_no_nans_on_pathological_inputs`** — zeros, boundary points,
  and $10^6$-magnitude inputs all yield finite outputs.

## Numerical-stability choices

- `BALL_EPS = 1e-5` — projection keeps every point at norm
  $\le 1/\sqrt{c} - \varepsilon$ so `artanh` is well-defined.
- `_safe_norm` adds `1e-15` inside the sqrt — avoids the undefined
  gradient of $\|\cdot\|$ at zero.
- `_artanh` clamps argument to `[-1 + 1e-7, 1 - 1e-7]`.

These constants are conservative enough that the contracts above hold
on float32 CPU; relaxing them is a research decision (faster but riskier
near the boundary), not a refactor.

## Cost

Per attention call, the additional cost over a vanilla MHA is:

| Op                          | Extra FLOPs (per Q,K pair)               |
| --------------------------- | ---------------------------------------- |
| `expmap0(Q)`, `expmap0(K)`  | $O(d_{\text{head}})$                     |
| Möbius add for distance      | $O(d_{\text{head}})$                     |
| `artanh` of norm             | $O(1)$ per pair                          |

In practice this is a 1.4–1.8× wall-clock multiplier vs FlashAttention,
because the Möbius operations are not fused. Not a blocker for research
runs; would need a custom kernel for production-scale training.

## Open questions

- **Per-head learnable curvature** — currently `c` is a fixed feature
  param. Some variants of the paper learn $c_h$ per head. Would need
  the test contract to handle the gradient through `c`.
- **KV cache** — straightforward (Q/K are still Euclidean
  pre-projection); not implemented because `HyperbolicMultiHeadAttention`
  recomputes the full attention each call.
- **Causal masking** — the `attn_mask` arg already supports masking;
  no special handling needed because masking happens at the logit
  level before softmax.

## References

- M. Nickel, D. Kiela. *Poincaré Embeddings for Learning Hierarchical
  Representations.* NeurIPS 2017.
- O. Ganea, G. Bécigneul, T. Hofmann. *Hyperbolic Neural Networks.*
  NeurIPS 2018.
- A. Ungar. *Analytic Hyperbolic Geometry: Mathematical Foundations and
  Applications.* World Scientific, 2005.
- F. Sala, C. De Sa, A. Gu, C. Ré. *Representation Tradeoffs for
  Hyperbolic Embeddings.* ICML 2018.
