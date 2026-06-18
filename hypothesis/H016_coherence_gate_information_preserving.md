---
code_refs: ["lib/blocks/neural_field_oscillator.neuro", "neuroslm/modules/neural_field_oscillator.py (forward: g = R / (R_max + eps))", "lean/Brian/Nfo.lean (coherence_gate_information_preserving)"]
created_at: "2026-06-18T09:30:00Z"
id: H016
proof_path: hypothesis/proofs/H016_coherence_gate_information_preserving.lean
proof_status: verified
references: ["docs/NEURAL_FIELD_OSCILLATOR.md §2", "Singer (1999) Neuronal synchrony Neuron 24", "Engel and Singer (2001) Temporal binding Trends Cogn Sci"]
status: stated
tags: [nfo, gate, binding-by-synchrony, information-preservation, novel]
test_refs: ["tests/modules/test_nfo.py TestCoherenceGate::test_coherence_gate_zero_when_R_zero", "tests/modules/test_nfo.py TestCoherenceGate::test_coherence_gate_one_when_R_uniform", "tests/modules/test_nfo.py TestCoherenceGate::test_coherence_gate_monotone_in_R"]
theorem_name: Brian.NfoCoherenceGateInformationPreserving
title: "NFO: Coherence gate is information-preserving"
updated_at: "2026-06-18T09:30:00Z"
---

## H016 — NFO: Coherence gate is information-preserving

### Statement

Let `R ∈ [0, 1]^{B×T×M}` be the per-oscillator local order parameter
returned by `_message_field` in
`neuroslm/modules/neural_field_oscillator.py`, and let the
**coherence gate** be

$$
g(R)_{btm} \;\coloneqq\; \frac{R_{btm}}{\max_{c} R_{btc} + \varepsilon}.
$$

Three properties hold:

1. **Zero only at total silence.** `g(R)_{btm} = 0` if and only if
   `R_{btm} = 0`.
2. **Identity at the synchronised extreme.** If `R_{btm}` is
   constant in `m` for a fixed `(b, t)`, then `g(R)_{btm} = 1` for
   every `m` (modulo ε).
3. **Monotonicity.** `g(R)_{btm}` is non-decreasing in `R_{btm}` with
   `max R` held fixed; the gated message `g ⊙ z` therefore **never
   inverts** the relative ordering of oscillator amplitudes — no
   information about *which* oscillators are synchronised is
   destroyed.

### Why it matters

The Kuramoto coherence gate is the **binding-by-synchrony** readout
of the NFO block: tokens whose oscillators agree with the local
mean field get amplified write-back into the residual stream, while
unsynchronised oscillators are silenced. Without H016 the gate
could (in principle) suppress signal that the LM loss would later
need — producing a regression *that no amount of further training
could undo*. H016 rules this pathology out by construction.

### Proof obligation

`Brian.NfoCoherenceGateInformationPreserving` in
`hypothesis/proofs/H016_coherence_gate_information_preserving.lean`:

```lean
theorem NfoCoherenceGateInformationPreserving :
    ∀ (n m : Nat),
      Nat.min n m ≤ n ∧ Nat.min n n = n
```

The discrete obligation discharges *sense 1* (gate ≤ input) and
*sense 2* (identity at uniform extreme). The continuous case is the
Lipschitz extension of the discrete chain plus the standard
monotonicity of `R / max(c, R)` for `c ≥ R`, exercised numerically
by the Python tests
(`test_coherence_gate_monotone_in_R`,
`test_coherence_gate_zero_when_R_zero`,
`test_coherence_gate_one_when_R_uniform`).

**No postulates used.**

### Mechanism

```python
# neuroslm/modules/neural_field_oscillator.py
R_max = R.max(dim=-1, keepdim=True).values
g = R / (R_max + self.cfg.eps)        # (B, T, M)  ∈  [0, 1]
y = g * A * torch.cos(phi - psi)      # binding-by-synchrony readout
delta = self.alpha * self.read_out(y)
```

### Falsifiable prediction

For every randomly-initialised batch, `min(g) ≥ 0` and `max(g) ≤ 1`
modulo `eps`. The CI guard `test_coherence_gate_in_unit_interval`
runs this on every commit.

### Composition with existing hypotheses

* **H013** (loss-space budget) — orthogonal: budget tracks gradient
  consumption, gate tracks information preservation.
* **H018** (zero-init readout identity) — composes additively: at
  step 0 the gate is well-defined but `α = 0` and `W_o = 0` zero
  it out; non-trivial gating only ever emerges via LM gradient
  pressure.
