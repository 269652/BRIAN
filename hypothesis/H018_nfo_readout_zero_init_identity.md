---
code_refs: ["lib/blocks/neural_field_oscillator.neuro", "neuroslm/modules/neural_field_oscillator.py (__init__: zero-init readout)", "lean/Brian/Nfo.lean (zero_init_readout_is_identity)"]
created_at: "2026-06-18T09:30:00Z"
id: H018
proof_path: hypothesis/proofs/H018_nfo_readout_zero_init_identity.lean
proof_status: verified
references: ["docs/NEURAL_FIELD_OSCILLATOR.md §5", "Bachlechner et al (2021) ReZero", "H001 Phi monotonicity"]
status: stated
tags: [nfo, init-discipline, rezero, baseline-identity, no-regression, novel]
test_refs: ["tests/modules/test_nfo.py TestBaselineIdentity::test_baseline_identity_at_init", "tests/modules/test_nfo.py TestBaselineIdentity::test_baseline_identity_for_any_input", "tests/modules/test_nfo.py TestBaselineIdentity::test_baseline_identity_for_any_config"]
theorem_name: Brian.NfoZeroInitReadoutIdentity
title: "NFO: Zero-init readout ⇒ baseline-identity forward"
updated_at: "2026-06-18T09:30:00Z"
---

## H018 — NFO: Zero-init readout ⇒ baseline-identity forward

### Statement

For any input residual `h ∈ ℝ^{B×T×d}`, any oscillator state
`(A, φ)`, any coupling matrix `K`, any choice of dynamics
hyperparameters `(μ, A*, κ, Δt)`, and any block sub-step count
`n_steps`: if the readout weight matrix `Wo` is zero,

$$
\mathrm{NFO}(h) \;=\; h + \alpha \cdot \mathrm{linear}(y, 0) \;=\; h + 0 \;=\; h.
$$

That is, **attaching an NFO block at any depth of the trunk is a
no-op at step 0** — the LM loss, perplexity, OOD ECE, and every
metric column not specifically related to the oscillator field
remain bit-identical to the un-augmented baseline.

### Why it matters

The H001 lower-bound monotonicity, H015 quantitative coherence
bound, H016 information-preserving gate, and H017 amplitude
contractivity together guarantee that NFO *cannot eventually do
worse than baseline*. H018 is the **shorter-horizon** guarantee:
NFO cannot do worse than baseline **on day 1**. Without H018 the
new block could (in principle) degrade the trunk under
gradient-checkpointed pre-training before the readout has had a
chance to learn — a regression that is *causally invisible* to
ablation studies because the bad initial step happens at step 0.

H018 also makes integration safe under the existing
`harness.compare_baseline` smoke test: any commit that flips an
NFO config bit must, by construction, produce identical loss at
step 0 to the same arch with NFO turned off entirely.

### Proof obligation

`Brian.NfoZeroInitReadoutIdentity` in
`hypothesis/proofs/H018_nfo_readout_zero_init_identity.lean`:

```lean
theorem NfoZeroInitReadoutIdentity : ∀ (h : Nat), h + 0 = h :=
  Brian.Nfo.zero_init_readout_is_identity
```

Trivially discharged by `Nat.add_zero`. The Python lift lives in
`tests/modules/test_nfo.py::TestBaselineIdentity` and is checked
on three orthogonal axes:

1. `test_baseline_identity_at_init` — random batch, default config.
2. `test_baseline_identity_for_any_input` — 32 random batches across
   different shapes and dtypes.
3. `test_baseline_identity_for_any_config` — 16 random configs
   (varying `n_osc`, `n_steps`, `kappa_init`, `mu_init`, `a_star_init`,
   `kernel_temperature`) all collapse to the identity at `α=0`.

**No postulates used.**

### Mechanism

```python
# neuroslm/modules/neural_field_oscillator.py
self.alpha = nn.Parameter(torch.tensor(float(cfg.alpha_init)))   # default 0.0
self.read_out = nn.Linear(self.M, self.d_model, bias=False)
nn.init.zeros_(self.read_out.weight)                              # H018 contract
```

The **double zero-init** (both `alpha` *and* `Wo`) is intentional:
either alone would suffice to discharge H018, but having both lets
the optimiser opt into amplitude *or* direction independently
without ever risking a step-0 regression. The same belt-and-braces
discipline is used by `PCT` (H001 mutation), `NeuralGeometryAdapter`
(meta-trainable wiring), and `MemoryCrossAttention` (RETRO-style
retrieval).

### Falsifiable prediction

For every commit to the NFO module, `pytest tests/modules/test_nfo.py
-k baseline_identity` must pass. A single failing assertion on any of
the three sub-tests **refutes** the discipline and forces a config
revert before merge.

### Composition with existing hypotheses

* **H001** (Φ monotonicity) — composes: the H018 identity is what
  makes the H001 mutation safe at the architectural level (a no-op
  initial state cannot have a smaller Φ than the eventual non-trivial
  state, by H001's `Phi_monotone_addList`).
* **H016** (coherence gate) — composes: at step 0 the gate is
  well-defined but its output is multiplied by `α=0` so it has no
  bearing on the residual.
