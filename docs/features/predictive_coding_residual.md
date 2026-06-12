# Predictive Coding Residuals

**Mechanism #4.** Rao-Ballard (1999) style local residual computation.

## Hypothesis

The classical "activation propagates forward" view of deep networks
gets inverted in hierarchical predictive coding (PC): each layer
*predicts* the layer below; only the prediction error $\varepsilon$
moves upward.

Cortical predictions:
- explain away what's already known → upward signal is sparser;
- match the surprise-driven update rule predicted by free-energy
  formulations (Friston 2005);
- compose with surprise-gated MoE (mechanism #5) — the PC residual
  norm is exactly the "surprise" signal that routes compute.

## Math

At layer $\ell$ with state $x_\ell$ and a predictor $g_\ell$ from above:

$$\begin{aligned}
\hat{x}_\ell  &= g_\ell(x_{\ell+1})        \quad\text{(top-down prediction)}\\
\varepsilon_\ell &= x_\ell - \hat{x}_\ell  \quad\text{(residual / prediction error)}\\
x'_\ell &= x_\ell - \alpha \cdot \varepsilon_\ell \quad\text{(gradient step on } \tfrac{1}{2}\varepsilon^2\text{)}
\end{aligned}$$

Step size $\alpha \in (0, 1]$; iterating drives $\varepsilon \to 0$
(convergence pinned by `test_iterative_mode_decreases_residual`).

The `PredictiveCodingResidual` module wraps this as an edge endpoint
in *self-predicting* form: $g$ is a learned linear map, $x_\text{above}
= x_\text{below} = x$, so each layer learns the identity plus a
residual sparsifier — a well-studied recurrent generative model.

## Wiring

```neuro
import { predictive_coding_residual } from "@brian/features/predictive_coding_residual"

feature predictive_coding_residual {
    equation: predictive_coding_residual_eq,
    active:   true,
    impl:     "neuroslm.modules.predictive_coding_residual.PredictiveCodingResidual",
    params: {
        d_model:      256,
        step_size:    0.1,
        mode:         "iterative",
        n_iterations: 3
    },
    endpoints: { edge: { kind: "edge", inputs: [x_pre], output: y } }
}

synapse pfc -> association { feature: "predictive_coding_residual.edge", weight: 1.0 }
```

The residual norm is exposed via `circuit.feature_predictive_coding_residual.last_residual_norm` for downstream loss heads (e.g. as the surprise input to a surprise-gated MoE).

## Ablation protocol

1. Baseline: `active: false`.
2. Single-step PC: `active: true`, `mode: "single"`.
3. Iterative PC: `active: true`, `mode: "iterative"`, `n_iterations: 3`.
4. Compare validation loss + residual-norm trajectories.

## References

- Rao, R.P.N., Ballard, D.H. — "Predictive coding in the visual cortex:
  a functional interpretation of some extra-classical receptive-field
  effects", *Nature Neurosci.* 2(1), 1999.
- Friston, K. — "A theory of cortical responses", *Phil. Trans. R. Soc.
  B* 360(1456), 2005.
- Whittington, J., Bogacz, R. — "An approximation of the error back-
  propagation algorithm in a predictive coding network with local
  Hebbian synaptic plasticity", *Neural Comp.* 29(5), 2017.

## Implementation

- Module: `neuroslm/modules/predictive_coding_residual.py`
- Contracts: `tests/test_predictive_coding_residual.py` (14 contracts)
- Feature spec: `architectures/lib/features/predictive_coding_residual.neuro`
