# Surprise-Gated Mixture-of-Experts

**Mechanism #5.** Standard MoE (Shazeer et al. 2017) extended with
dynamic per-token top-$k$ driven by a "surprise" signal.

## Hypothesis

Standard MoE wastes compute on trivial tokens (everything goes through
the same $k$ experts). Making $k$ data-dependent on per-token surprise:

- matches the free-energy principle (Friston 2010): surprise drives
  perception / inference / compute allocation;
- gives the model a soft early-exit / extra-think knob without
  architectural surgery;
- composes naturally with predictive coding (mechanism #4): the PC
  residual norm IS the surprise input.

Compute economics resemble Mixture-of-Depths (Raposo et al. 2024) but
with the routing decided by surprise instead of a learned router gate.

## Math

Per-token gating: $g(x) = \text{softmax}(W_g x)$ over $E$ experts.

Per-token surprise $s(x) \in \mathbb{R}_{\ge 0}$:
- supplied externally (e.g. PC residual norm), or
- computed internally as normalised gating entropy:
  $s = H(g(x)) / \log E$ (low entropy = confident = low $s$).

Dynamic top-$k$:

$$k(x) = \text{round}\Bigl(k_\text{min} + (k_\text{max} - k_\text{min}) \cdot \sigma\bigl(\text{steepness} \cdot (s(x) - \text{midpoint})\bigr)\Bigr)$$

Per-token output:

$$y(x) = \sum_{e \in \text{top}_k(g(x))} \frac{g_e(x)}{\sum_{e' \in \text{top}_k} g_{e'}(x)} \cdot \text{Expert}_e(x)$$

Auxiliary load-balance loss (Switch-style):

$$L_\text{aux} = E \cdot \sum_e f_e \cdot P_e$$

where $f_e$ = fraction of tokens routed to expert $e$, $P_e$ = average
gating probability over $e$. Minimised at uniform routing.

## Wiring

```neuro
import { surprise_gated_moe } from "@brian/features/surprise_gated_moe"

feature surprise_gated_moe {
    equation: surprise_gated_moe_eq,
    active:   true,
    impl:     "neuroslm.modules.surprise_gated_moe.SurpriseGatedMoE",
    params: {
        d_model:             256,
        n_experts:           8,
        d_hidden:            1024,
        k_min:               1,
        k_max:               4,
        midpoint:            1.0,
        steepness:           4.0,
        load_balance_weight: 0.01
    },
    endpoints: { edge: { kind: "edge", inputs: [x_pre], output: y } }
}

synapse association -> pfc { feature: "surprise_gated_moe.edge", weight: 1.0 }
```

The aux loss is exposed via `circuit.feature_surprise_gated_moe.last_aux_loss` for the trainer to add to the main objective; mean $k$ via `last_mean_k` for diagnostics.

## Ablation protocol

1. Baseline: `active: false`.
2. Fixed-$k$ MoE: `k_min == k_max`, e.g. both = 2.
3. Surprise-gated: `k_min: 1, k_max: 4`, internal surprise.
4. Surprise-gated + PC: surprise = PC residual norm from mechanism #4.
5. Compare loss curves + mean-$k$ trajectories.

## References

- Shazeer, N., et al. — "Outrageously Large Neural Networks: The
  Sparsely-Gated Mixture-of-Experts Layer", *ICLR* 2017.
- Fedus, W., Zoph, B., Shazeer, N. — "Switch Transformers", *JMLR* 23
  (2022).
- Raposo, D., et al. — "Mixture-of-Depths: Dynamically allocating
  compute in transformer-based language models", *NeurIPS* 2024.
- Friston, K. — "The free-energy principle: a unified brain theory?",
  *Nature Rev. Neurosci.* 11(2), 2010.

## Implementation

- Module: `neuroslm/modules/surprise_gated_moe.py`
- Contracts: `tests/test_surprise_gated_moe.py` (20 contracts)
- Feature spec: `architectures/lib/features/surprise_gated_moe.neuro`
