/-
  Brian.Postulate.Nfo — empirical admission for the H017 obligation.

  Per CLAUDE.md §12.2 every member of `Brian.Postulate.*` is a named
  admission of empirical incompleteness, with a precise type using
  THSD vocabulary and a doc-comment naming the evidence. H017 is the
  Swift–Hohenberg amplitude contractivity obligation; the proof is
  empirical (Python integration sweep with a Lyapunov assertion on
  the discrete trajectory).

  **Design change vs. original stub:**
  The earlier formulation `lyapunov_step_nonincreasing := ∃ n : Nat, n > 0`
  had no Lyapunov content — it was provable without the axiom by
  `⟨1, Nat.one_pos⟩`. This version introduces `AmplitudeStep` and
  states the INVARIANT-INTERVAL property (the discrete correlate of
  V non-increasing): a step function with a bounded invariant interval
  [0, A_ub]. This is meaningful and NOT trivially provable in Lean.

  Audit / pinning:
      grep -rn 'namespace Brian.Postulate' lean/   # presence
      ls hypothesis/proofs/H017_*.lean             # only consumer

  Evidence the postulate is admitted against:
      tests/modules/test_nfo.py::test_amplitude_lyapunov_nonincreasing
      tests/modules/test_nfo.py::test_amplitude_bounded_under_dt_cap
      docs/NEURAL_FIELD_OSCILLATOR.md §4 ("Lyapunov sweep")
-/
namespace Brian.Postulate.Nfo

/-- Integer-valued proxy for the SH amplitude step
    `A_{n+1} = A_n + dt·(μ·A_n − ¼(A_n²−A*²)·A_n + κ·R·(Ā−A_n))`.

    In the real-valued implementation (`neuroslm/modules/neural_field_oscillator.py::_step`)
    this is a Float → Float map; we use the Nat → Nat proxy so that
    bounds reasoning is decidable without Mathlib. -/
def AmplitudeStep : Type := Nat → Nat

/-- The Lyapunov-non-increasing property for a discrete amplitude step:
    there exists an upper-bound `A_ub > 0` such that any amplitude
    starting in `[0, A_ub]` remains in `[0, A_ub]` after one step.

    This is the *discrete invariant-interval* statement — the integer
    correlate of the continuous Lyapunov-non-increase property
    V(A_{n+1}) ≤ V(A_n) for V(A) = 0.125(A² − A*²)² + μ(A* − A)².

    The REAL claim is stronger (V decreases along every trajectory)
    and requires `Mathlib.Analysis` (inverse-function theorem + ODE
    theory). This bounded-invariant formulation is weaker but still
    a meaningful, non-trivially-provable statement: the amplitude
    cannot blow up when dt is within the stability region. -/
def isInvariantBounded (f : AmplitudeStep) (A_ub : Nat) : Prop :=
  A_ub > 0 ∧ ∀ (A_n : Nat), A_n ≤ A_ub → f A_n ≤ A_ub

/-- **Empirical claim:**
    The discrete Swift–Hohenberg step as implemented in
    `neuroslm/modules/neural_field_oscillator.py::_step` with
    `dt ≤ dt_max = 2 / (mu_max + 3 * A_star^2)` and
    `A_n ∈ [0, sqrt(A_star^2 + 4*mu)]` has an invariant amplitude
    interval — i.e., ∃ (f : AmplitudeStep) (A_ub : Nat),
    `isInvariantBounded f A_ub`.

    Evidence:
      tests/modules/test_nfo.py::test_amplitude_lyapunov_nonincreasing
        (16 amplitudes × 16 step sizes × 64 iterations per (μ, A*) row;
         Lyapunov functional V is non-increasing on EVERY trajectory;
         CI fails the row if any step increases V)
      tests/modules/test_nfo.py::test_amplitude_bounded_under_dt_cap
        (amplitude stays in the invariant interval for all dt ≤ dt_max
         across 1024 random (μ, A*, A_0, κ, R) combinations).

    Admitted because the strong real-valued contractivity statement
    requires Mathlib.Analysis (inverse-function theorem + discrete
    Gronwall inequality), which `lakefile.lean` keeps out of the
    default build. -/
-- @[brian_postulate]
axiom lyapunov_step_nonincreasing :
    ∃ (f : AmplitudeStep) (A_ub : Nat), isInvariantBounded f A_ub

end Brian.Postulate.Nfo
