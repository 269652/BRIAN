# HRR Memory (Holographic Reduced Representations)

**Mechanism #3.** Content-addressable memory via circular convolution
binding (Plate 1995, *Holographic Reduced Representations*).

## Hypothesis

For long-range associative recall, a memory whose **storage** cost is
$O(d)$ per binding and whose **read** cost is $O(d \log d)$ per token
beats softmax attention's $O(T^2 d)$ when the working-set size $N \ll d$.

The binding operation (circular convolution) is associative and
commutative; the unbinding operation (involution-correlation) returns
the value bound to a key with bounded noise that scales as
$\sqrt{N/d}$ (Plate Thm 3.1) — graceful degradation under load.

## Math

Bind: $a \circledast b = \text{IFFT}(\text{FFT}(a) \odot \text{FFT}(b))$

Inverse: $a^{-1} \approx \text{involution}(a)$ — the spectral conjugate.

Unbind: $\text{unbind}(M, k) = M \circledast k^{-1}$.

For memory $M = \sum_i k_i \circledast v_i$:

$$\text{unbind}(M, k_j) = v_j + \underbrace{\sum_{i \ne j} (k_i \circledast k_j^{-1}) \circledast v_i}_{\text{noise: } O(\sqrt{N/d})}$$

The `HRRMemory` module exposes this as a self-attention-shaped layer:

1. Project $x \to (k, v, q)$ via three linear maps.
2. Build per-batch memory $M_b = \sum_t k_{b,t} \circledast v_{b,t}$.
3. Retrieve $r_{b,t} = M_b \circledast q_{b,t}^{-1}$.
4. Project back: $y = W_\text{out} r$.

## Wiring

```neuro
import { hrr_memory } from "@brian/features/hrr_memory"

feature hrr_memory {
    equation: hrr_memory_eq,
    active:   true,
    impl:     "neuroslm.modules.hrr_memory.HRRMemory",
    params: {
        d_model:        256,
        d_memory:       1024,   # larger = less noise under superposition
        normalize_keys: true,
        bias:           true
    },
    endpoints: { edge: { kind: "edge", inputs: [x_pre], output: y } }
}

synapse hippo -> association { feature: "hrr_memory.edge", weight: 1.0 }
```

## Ablation protocol

1. Baseline: `active: false`.
2. Activate with `d_memory = d_model` (no overhead, lower fidelity).
3. Activate with `d_memory = 4 · d_model` (higher fidelity).
4. Compare long-range retrieval benchmarks (LRA-style: copy, ListOps).

## References

- Plate, T.A. — *Holographic Reduced Representations: Convolution
  Algebra for Compositional Distributed Representations* (CSLI 2003).
- Kanerva, P. — "Hyperdimensional Computing: An Introduction",
  *Cognitive Computation* 1 (2009).
- Schlegel, K., et al. — "A comparison of vector symbolic
  architectures", *Artificial Intelligence Review* 55 (2022).

## Implementation

- Module: `neuroslm/modules/hrr_memory.py`
- Contracts: `tests/test_hrr_memory.py` (16 contracts)
- Feature spec: `architectures/lib/features/hrr_memory.neuro`
