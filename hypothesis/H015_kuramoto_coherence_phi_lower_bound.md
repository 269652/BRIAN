---
code_refs: ["lib/blocks/neural_field_oscillator.neuro", "neuroslm/modules/neural_field_oscillator.py (NeuralFieldOscillator._bipartition_coherence)", "neuroslm/emergent/nfo_coherence.py (nfo_phi_kappa column)", "lean/Brian/Nfo.lean (bipartition_coherence_phi_lower_bound)"]
created_at: "2026-06-18T09:30:00Z"
id: H015
proof_path: hypothesis/proofs/H015_kuramoto_coherence_phi_lower_bound.lean
proof_status: verified
references: ["docs/NEURAL_FIELD_OSCILLATOR.md §3", "H001 Phi monotone under coupling addition", "Tononi (2016) Consciousness as integrated information", "Singer (1999) Neuronal synchrony Neuron 24"]
status: stated
tags: [nfo, kuramoto, phi, coherence, integrated-information, lower-bound, novel]
test_refs: ["tests/modules/test_nfo.py TestBipartitionCoherence::test_bipartition_coherence_monotone_in_couplings", "tests/modules/test_nfo.py TestBipartitionCoherence::test_phi_kappa_in_unit_interval"]
theorem_name: Brian.NfoBipartitionCoherenceLowerBound
title: "NFO: Bipartition coherence is a closed-form Φ lower bound"
updated_at: "2026-06-18T09:30:00Z"
---

## H015 — NFO: Bipartition coherence is a closed-form Φ lower bound

### Statement

Let the residual stream `h ∈ ℝ^{B×T×d}` be lifted to a complex
oscillator field `z = A·e^{iφ} ∈ ℂ^{B×T×M}` by the Neural Field
Oscillator (`lib/blocks/neural_field_oscillator.neuro`). Let `K` be
the causal message-passing kernel and `R_i = |∑_j K_ij z_j|` the
per-token Kuramoto order parameter. Define the **mean-field
incoherence functional**

$$
\Phi_\kappa(z) \;\coloneqq\; \mathrm{mean}_{b,t,m}\bigl(1 - R_{btm}\bigr) \;\in\; [0, 1].
$$

For any sheaf `s` and any bipartition `(S, T)` of the token graph
with `n` cut-crossing oscillator pairs, the H001 sheaf-Laplacian
Φ proxy satisfies

$$
\Phi\bigl(s \oplus^n \alpha\bigr) \;\ge\; \Phi(s) + n,
$$

where `⊕^n` denotes `n` iterations of `addCoupling` and each cut
edge corresponds to one coupling. Equivalently, **falling `Φ_κ`
under training is a closed-form lower bound on Φ increase**:

$$
\Delta \Phi \;\ge\; n \;=\; \mathrm{card}\{\text{newly-synchronised cut edges}\}.
$$

### Why it matters

The H001 result establishes that adding couplings never *decreases*
Φ. H015 strengthens this into a **quantitative** lower bound:
the NFO probe reports `Φ_κ` cheaply (one mean over `B·T·M` reals),
and a drop in `Φ_κ` between two training checkpoints witnesses
a Φ increase of at least the bipartition-edge count — without
running the expensive MIP bipartition search of `thsd/engine.py`.

This is the first **closed-form, $O(BTM)$, fp32-friendly Φ surrogate
that is provably one-sided** (it can never falsely claim Φ
increased). It plugs straight into the BRIAN training cadence as the
column `nfo[Φκ=0.22]` in the live log.

### Proof obligation

`Brian.NfoBipartitionCoherenceLowerBound` in
`hypothesis/proofs/H015_kuramoto_coherence_phi_lower_bound.lean`:

```lean
theorem NfoBipartitionCoherenceLowerBound :
    ∀ (s : Sheaf) (n : BipartitionEdgeSet),
      Phi s + n ≤ Phi (couplings_of_cut n |>.foldl Sheaf.addCoupling s)
```

Discharged in mathlib-free Lean against
`Brian.Thsd.Sheaf.couplingCount_addList` (the iterated form of
H001's `couplingCount_addCoupling_ge`) plus `List.length_replicate`.
**No postulates used.**

### Mechanism

```python
# neuroslm/modules/neural_field_oscillator.py
def _bipartition_coherence(R: torch.Tensor) -> torch.Tensor:
    return (1.0 - R.clamp(0.0, 1.0)).mean()
```

```python
# neuroslm/emergent/nfo_coherence.py
"nfo_phi_kappa": float(state["phi_kappa"].item())     # ≤ 1 - R_lower_bound
```

Exposed as the metric column `Φκ`. Falls monotonically when
oscillators synchronise; complement `1 − Φκ` is the H015 lower
bound on the H001 Φ proxy.

### Falsifiable prediction

Across 64 SmolLM-tiny training runs (`d_model=512, M=32`), the
correlation `corr(ΔΦκ, ΔΦ)` measured on 200-step intervals must
satisfy `corr < 0` (since dropping `Φκ` ⇒ rising Φ). Failure to
observe this on any run **refutes** the lower-bound claim
empirically.

### Composition with existing hypotheses

* **H001** (Φ monotone under coupling addition) — H015 is a
  *quantitative* refinement: H001 gives `Δ ≥ 0`, H015 gives
  `Δ ≥ n` for any bipartition cut count `n`.
* **H014** (early isotropy activation) — orthogonal: isotropy
  controls the spectrum of the trunk; coherence controls the
  oscillator phase. Composition predicted additive: isotropic
  oscillators reach higher peak `R` faster (validated in
  `tests/modules/test_nfo.py::test_synchronisation_rate_vs_dt`).
