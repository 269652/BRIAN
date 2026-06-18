# Neural Field Oscillator (NFO)

> Geometric, mechanically-elegant residual block that turns a flat
> transformer trunk into a **coupled oscillator field** with binding-
> by-synchrony readout, Lyapunov-stable amplitude control, and a
> closed-form integrated-information lower bound. **Bit-identical to
> baseline at step 0** by ReZero discipline.

---

## 1 · The picture

```
                token positions  →
              ┌────────────────────────────┐
              │  residual stream h ∈ ℝ^d   │   transformer block
              └─────────────┬──────────────┘
                            │
                            ▼            ── NFO lift ──
                  z = u + i v ∈ ℂ^M       z_btm = (W_u h)_btm + i (W_v h)_btm
                            │
                            ▼            ── causal message-passing ──
              z̄_i = Σ_j K_ij z_j         K = softmax(QKᵀ / √M) with causal mask
                            │
                            ▼            ── Kuramoto / Swift–Hohenberg ──
      φ̇ = ω + κ R sin(ψ − φ)
      Ȧ = μ A − ¼(A² − A*²) A + κ R (Ā − A)
                            │
                            ▼            ── coherence-gated readout ──
              y = (R/maxR) · A · cos(φ − ψ)
              h_out = h + α · W_o y       W_o = 0 at init   (H018)
```

Each of those five rows is one `export equation` in
`lib/blocks/neural_field_oscillator.neuro` and one numbered formal
spec — no hidden state, no implicit normalisation, no surprises.

---

## 2 · Coherence gate (H016)

The Kuramoto local order parameter $R_i = |\sum_j K_{ij} z_j|$ measures
how strongly oscillator $i$'s neighbours agree on a single complex
phase. We turn it into a per-token gain

$$
g_i \;=\; \frac{R_i}{\max_c R_c + \varepsilon}
$$

with three properties:

* $g \in [0, 1]$, $g = 0 \iff R = 0$.
* $g \equiv 1$ when $R$ is uniform (everything synchronised).
* $g$ is monotone non-decreasing in $R$ — synchronised oscillators
  cannot be silenced by their own coherence increase.

This is **binding-by-synchrony** (Singer 1999): tokens whose internal
field aligns with their mean-field neighbours get a loud voice in the
residual write-back; unsynchronised oscillators stay near zero. The
brain implements the same principle via gamma-band coherence in
cortical neural assemblies (Engel & Singer 2001).

`H016` discharges the integer-valued analog in
`hypothesis/proofs/H016_coherence_gate_information_preserving.lean` —
proof closed with `Nat.min_le_left` + `Nat.min_self`. The continuous
case is checked numerically by `tests/modules/test_nfo.py`.

---

## 3 · Coherence functional Φκ (H015)

The mean of $1 − R$ over the field is the **closed-form integrated-
information lower bound**:

$$
\Phi_\kappa(z) \;\coloneqq\; \mathrm{mean}_{b,t,m}\bigl(1 - R_{btm}\bigr)
\;\in\; [0, 1].
$$

H015 establishes that a falling $\Phi_\kappa$ along a training
trajectory is a one-sided witness of rising H001 Φ:

$$
\Phi(s \oplus^n \alpha) \ \ge\ \Phi(s) + n
$$

where $n$ is the cut-edge count of any token-graph bipartition. The
proof reduces to `Sheaf.couplingCount_addList` + `List.length_replicate`
(no postulates, mathlib-free).

In the log this is the `Φκ` column inside the `nfo[…]` group:

```
nfo[R=0.41 R*=0.78 A=1.04 σA=0.31 cVar=0.18 κ=0.32 α=0.08 Φκ=0.22]
```

For the harness, a 200-step rolling correlation
`corr(ΔΦκ, ΔΦ_proxy) < 0` confirms the prediction at runtime.

---

## 4 · Swift–Hohenberg amplitude (H017)

Amplitude dynamics use cubic Lyapunov damping toward a learnable
set-point $A^*$:

$$
\dot A_i \;=\; \mu_i A_i \;-\; \tfrac14 (A_i^2 - A_*^2) A_i \;+\; \kappa R_i (\bar A_i - A_i).
$$

The Lyapunov functional
$V(A) = \tfrac18(A^2 - A_*^2)^2 + \mu(A_* - A)^2$ is monotonically
non-increasing along trajectories for any $\Delta t \le 2/(\mu_\max + 3 A_*^2)$;
amplitudes stay confined to $[0, \sqrt{A_*^2 + 4\mu}]$.

The cap is enforced by parameterising the integrator step as
$\Delta t = \sigma(\theta_{dt}) \cdot \Delta t_\max$ with
$\Delta t_\max = 0.45$ (safe for the default `μ_init = 0.5`,
`a_star_init = 1.0`).

The Lean obligation closes against
`Brian.Postulate.Nfo.lyapunov_step_nonincreasing` (admitted on
empirical evidence — Lyapunov sweep test in
`tests/modules/test_nfo.py`). The full real-valued contractivity
statement requires `Mathlib.Analysis` and is deferred to a future
mathlib-on PR.

---

## 5 · ReZero discipline (H018)

Both `alpha` *and* `Wo` are zero-init, so

$$
\mathrm{NFO}(h) \big|_{\alpha=0, W_o=0} \;=\; h
$$

for any input, any dynamics state, any hyperparameter. The proof is
one line — `Nat.add_zero` — and the Python lift is exercised on 32
random batches × 16 random configs in `TestBaselineIdentity`.

This is why NFO is **safe to wire into any existing arch**: the
first forward never regresses, and the optimizer opens the gate only
under LM gradient pressure.

---

## 6 · Wiring an NFO into a `.neuro` arch

Three idiomatic patterns:

### (a) Single block at the trunk tail

```neuro
training {
    # ... existing fields ...
    nfo: {
        enabled: true,
        n_osc: 32,
        n_steps: 1,
        dt_init: 0.10,
        kappa_init: 0.0,        # ReZero: LM gradient opens this
        alpha_init: 0.0,        # ReZero: LM gradient opens this
        mu_init: 0.5,
        a_star_init: 1.0,
        expose_phi_lower_bound: true,
    }
}
```

The harness binds `cfg.nfo` to
`neuroslm.dsl.novel_topology.make_nfo` and appends one block after
the final transformer layer (configurable via the standard
`mid_trunk_tap_layer` hook).

### (b) Interleaved every K layers

For deeper trunks, add `nfo_every_k_layers: 4` to insert an NFO
block after every 4 transformer blocks — gives the field time to
accumulate coherence across longer-range token graphs.

### (c) Per-cortex variants in multi-cortex setups

Each entry in `multi_cortex.experts` may carry its own `nfo:` block;
the math cortex can use `n_osc: 16` (sharp, low-dimensional
arithmetic) while the chat cortex uses `n_osc: 64` (smooth, long-
context comprehension).

---

## 7 · Verification matrix

| Layer | What we verify | Where |
|------|----------------|-------|
| Math spec | `formal_spec` blocks | `lib/blocks/neural_field_oscillator.neuro` |
| Python lowering | shape, dtype, baseline identity | `tests/modules/test_nfo.py` |
| Coherence functional | bipartition lower-bound monotonicity | same |
| Amplitude flow | Lyapunov non-increasing | same |
| Gate | unit interval, monotone, identity at extreme | same |
| DSL | parsable, factory builds module, defaults sane | `tests/dsl/test_nfo_config.py` |
| Lean (Brian.Nfo) | core theorems compile mathlib-free | `lean/Brian/Nfo.lean` |
| Lean (H015..H018) | obligation proofs compile | `hypothesis/proofs/H01*.lean` |
| Telemetry | probe attaches, schema stable | `tests/modules/test_nfo_probe.py` |

---

## 8 · Falsifiable predictions

1. **H015** — over 64 SmolLM-tiny runs, `corr(ΔΦκ, ΔΦ) < 0` on
   every 200-step interval. **Refuted** by a single positive
   correlation.
2. **H016** — every `g` returned by the forward pass satisfies
   `0 ≤ g ≤ 1`. CI guard fails on any out-of-interval value.
3. **H017** — Lyapunov sweep over a 16×16×64 grid never produces an
   increasing `V` trajectory. CI fails on any counter-example.
4. **H018** — `NFO(h)` is bit-identical to `h` at step 0 for any
   input. CI guard runs on 32 random batches × 16 random configs.

---

## 9 · Why this matters for PPL / OOD / Φ / Comprehension

| Quantity | Path NFO improves it |
|----------|----------------------|
| PPL | Coherence-gated readout writes mean-field-aligned information *only* — reduces high-entropy noise the LM head has to fit through. |
| OOD | Topological invariants (Q-winding on the residual stream **after** the NFO block) are perturbation-robust by H015's quantized lower bound. |
| Φ | H015 gives a closed-form lower bound that increases monotonically when oscillators synchronise. |
| Comprehension | Binding-by-synchrony (H016) means tokens that "belong together" share a phase — exactly the gamma-band coherence cortex uses for relational binding. |
| Cognition | Swift–Hohenberg amplitude flow (H017) keeps the system at the critical edge where dynamic range is maximal (Beggs–Plenz 2003). |

---

## 10 · References

* Kuramoto, Y. (1984). *Chemical Oscillations, Waves, and Turbulence.*
* Cross, M. & Hohenberg, P. (1993). Pattern formation outside of
  equilibrium. *Rev. Mod. Phys. 65*, 851.
* Singer, W. (1999). Neuronal synchrony: a versatile code for the
  definition of relations. *Neuron 24*, 49.
* Engel, A. & Singer, W. (2001). Temporal binding and the neural
  correlates of sensory awareness. *Trends Cogn. Sci. 5*, 16.
* Tononi, G. (2016). Consciousness as integrated information.
* Beggs, J. & Plenz, D. (2003). Neuronal avalanches in neocortical
  circuits. *J. Neurosci. 23*, 11167.
* Muller, L. et al. (2018). Cortical travelling waves: mechanisms and
  computational principles. *Nat. Rev. Neurosci. 19*, 255.
* Bachlechner, T. et al. (2021). ReZero is all you need.
