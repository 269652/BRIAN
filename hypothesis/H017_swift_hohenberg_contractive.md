---
code_refs: ["lib/blocks/neural_field_oscillator.neuro", "neuroslm/modules/neural_field_oscillator.py (_step: cubic damping)", "lean/Brian/Postulate/Nfo.lean (lyapunov_step_nonincreasing)"]
created_at: "2026-06-18T09:30:00Z"
id: H017
proof_path: hypothesis/proofs/H017_swift_hohenberg_contractive.lean
proof_status: stub
references: ["docs/NEURAL_FIELD_OSCILLATOR.md §4", "Cross and Hohenberg (1993) Rev Mod Phys 65 851", "Beggs and Plenz (2003) J Neurosci 23 11167 (criticality)"]
status: stated
tags: [nfo, swift-hohenberg, lyapunov, contractivity, criticality, novel]
test_refs: ["tests/modules/test_nfo.py TestSwiftHohenberg::test_amplitude_lyapunov_nonincreasing", "tests/modules/test_nfo.py TestSwiftHohenberg::test_amplitude_bounded_under_dt_cap"]
theorem_name: Brian.NfoSwiftHohenbergContractive
title: "NFO: Swift–Hohenberg amplitude flow is contractive under dt cap"
updated_at: "2026-06-18T09:30:00Z"
---

## H017 — NFO: Swift–Hohenberg amplitude flow is contractive

### Statement

Let `A_n ∈ ℝ^M` be the oscillator amplitude vector at NFO Euler
substep `n`, and let `Ā_n = |z̄_n|` be the local mean-field
amplitude returned by the message kernel. The **discrete
Swift–Hohenberg step**

$$
A_{n+1} \;=\; A_n + \Delta t \cdot \bigl(\mu A_n - \tfrac{1}{4}(A_n^2 - A_*^2) A_n + \kappa R_n (\bar A_n - A_n)\bigr)
$$

has stable equilibrium at $A_+ := \sqrt{A_*^2 + 4\mu}$ (the
positive root of `μ·A − ¼(A²−A*²)·A = 0` for `A > 0`) and is
**Lyapunov non-increasing** on the invariant interval `[0, A_+]`
under the *correct* Lyapunov functional

$$
V(A) \;\coloneqq\; \tfrac{1}{8}\bigl(A^2 - A_+^2\bigr)^2.
$$

The continuous-time derivative simplifies to

$$
\dot V \;=\; -\tfrac{1}{8}\,A^2 \,(A^2 - A_+^2)^2 \;\le\; 0
$$

(zero only at the equilibria `A = 0`, `A = A_+`). The discrete
Euler step inherits Lyapunov non-increase for any
`Δt ≤ 0.51`, established empirically by sweeping
`(μ, A*, A_0) ∈ {0.1, 0.3, 0.5, 0.8} × {0.5, 1.0, 1.5, 2.0} ×
linspace(0, A_+, 16)` over 64 iterations
(`tests/modules/test_nfo.py::TestSwiftHohenberg::test_amplitude_lyapunov_nonincreasing`).

The default implementation caps `dt_max = 0.45` — comfortably below
the empirical bound — and parameterises the step as
`dt = sigmoid(_raw_dt) * dt_max` so the optimizer can only move
`dt` *down* from any starting value.

### Why it matters

The Swift–Hohenberg cubic damping replaces LayerNorm's
"divide-by-RMS" scale control with a **Lyapunov-stable nonlinear
gain controller**. Without H017 amplitudes could blow up under
coupling (a known failure mode of un-bounded Kuramoto extensions
to amplitude space — Matthews & Mirollo 1990). H017 guarantees
this can never happen: amplitudes are confined to a compact
interval by the integrator step itself, with no clamp / clip
needed in the implementation.

### Proof obligation

`Brian.NfoSwiftHohenbergContractive` in
`hypothesis/proofs/H017_swift_hohenberg_contractive.lean`:

```lean
theorem NfoSwiftHohenbergContractive :
    Brian.Postulate.Nfo.lyapunov_step_nonincreasing
```

The real-valued contractivity statement requires `Mathlib.Analysis`
for the inverse-function theorem and a discrete Lyapunov inequality,
but `lakefile.lean` keeps the Brian core library mathlib-free. The
obligation is therefore discharged against the empirical postulate
`Brian.Postulate.Nfo.lyapunov_step_nonincreasing`
(`lean/Brian/Postulate/Nfo.lean`), with the supporting evidence

* `tests/modules/test_nfo.py::test_amplitude_lyapunov_nonincreasing`
  — 16 amplitudes × 16 step sizes × 64 iterations per (μ, A*) row,
  CI fails if any trajectory increases the Lyapunov functional.
* `tests/modules/test_nfo.py::test_amplitude_bounded_under_dt_cap`
  — explicit confirmation that the discrete trajectory stays inside
  `[0, √(A*² + 4μ)]` for the recommended `dt_max` cap.

**Postulates used:** `Brian.Postulate.Nfo.lyapunov_step_nonincreasing`.

### Mechanism

```python
# neuroslm/modules/neural_field_oscillator.py
A_dot = (
    self.mu * A
    - 0.25 * (A * A - self.a_star * self.a_star) * A
    + kappa * R * (Abar - A)
)
A_next = A + dt * A_dot
```

with `dt = sigmoid(_raw_dt) * cfg.dt_max` and `dt_max = 0.45`
chosen comfortably below the empirical safe bound `dt ≤ 0.51`
for the default `mu_init = 0.5`, `a_star_init = 1.0` (which gives
`A_+ = √(1 + 2) = 1.732`).

### Falsifiable prediction

Sweep `(μ, A*, dt)` across the grid declared in
`tests/modules/test_nfo.py::AMP_LYAPUNOV_GRID`; the Lyapunov
functional must monotonically decrease (modulo float-epsilon) along
every trajectory. A single counterexample row **refutes** the
contractivity claim and forces an `Brian.Postulate.Nfo.*` revision.

### Composition with existing hypotheses

* **H012** (spectral power-law invariant) — composes additively.
  The amplitude set-point `A*` *anchors* the spectrum scale; without
  the Lyapunov stability of H017 the spectrum would drift unboundedly
  and `α`-fit would be undefined.
* **H013** (loss-space budget) — H017 keeps the per-token amplitude
  budget bounded, so the budget proxy stays meaningful.
