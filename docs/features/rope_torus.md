# RoPE-on-a-Torus

**Mechanism #2.** Multi-frequency rotary positional encoding on the
product manifold T^n = (S^1)^n.

## Hypothesis

Sequence position is naturally cyclic at multiple scales (token,
sentence, paragraph, document). Classical RoPE (Su et al. 2021,
*RoFormer*) bakes a single exponential-decay frequency schedule into
the rotation; the torus formulation:

- gives every 2D embedding slice its own period drawn from a
  configurable schedule (geometric / linear / harmonic);
- wraps each slice's angle modulo its period (explicit torus topology),
  instead of relying on the geometric schedule's exponential decay tail;
- optionally exposes the period vector as `nn.Parameter` so the
  optimiser can shape the schedule per-data.

The classical RoPE relative-position property is preserved per slice:

$$\langle \text{RoPE}(q, m), \text{RoPE}(k, n)\rangle_j = \langle q, R_{n-m, j} k\rangle_j$$

inside slice $j$'s period.

## Math

For slice $j$ with period $P_j$ (in tokens) and position $p$:

$$\theta_{p,j} = 2\pi \cdot \frac{p \bmod P_j}{P_j} \qquad
R_{p,j} = \begin{pmatrix} \cos\theta_{p,j} & -\sin\theta_{p,j} \\
\sin\theta_{p,j} & \cos\theta_{p,j} \end{pmatrix}$$

Each pair $(x_{2j}, x_{2j+1})$ is rotated by $R_{p,j}$. The whole
operation is norm-preserving (O(2) on each slice).

Period schedules:

| Schedule | Formula | Used for |
|----|----|----|
| `geometric` | $P_j = 2\pi \cdot \text{base}^{2j/d}$ | classical RoPE (default) |
| `linear` | $P_j = (j+1)\cdot\text{base}/n_{pairs}$ | ablation |
| `harmonic` | $P_j = \text{base}/(j+1)$ | dense short periods |

## Wiring

```neuro
import { rope_torus } from "@brian/features/rope_torus"

# Activate with overrides:
feature rope_torus {
    equation: rope_torus_eq,
    active:   true,
    impl:     "neuroslm.modules.rope_torus.RoPETorus",
    params: {
        d_model:           256,
        max_seq_len:       4096,
        schedule:          "geometric",
        learnable_periods: true
    },
    endpoints: { edge: { kind: "edge", inputs: [x_pre], output: y } }
}

synapse sensory -> thalamus { feature: "rope_torus.edge", weight: 1.0 }
```

## Ablation protocol

1. Train with `active: false` for baseline.
2. Train with `active: true`, `learnable_periods: false`, default schedule.
3. Train with `active: true`, `learnable_periods: true`.
4. Compare loss curves on long-context benchmarks (PG19, ArXiv).

## References

- Su, J., Lu, Y., Pan, S., Murtadha, A., Wen, B., Liu, Y. — "RoFormer:
  Enhanced Transformer with Rotary Position Embedding", *Neurocomputing*
  568 (2024).
- Bronstein, M., et al. — *Geometric Deep Learning* (2021), §5.4.

## Implementation

- Module: `neuroslm/modules/rope_torus.py`
- Contracts: `tests/test_rope_torus.py` (19 contracts)
- Feature spec: `architectures/lib/features/rope_torus.neuro`
