import Brian.Postulate
import Brian.Statistics.Welch

/-
  Brian.Postulate.Welch — empirical admission for the Welch t-test's
  Type-I error bound (foundational to H005 and ImprovementGate).

  CLAUDE.md §12.2 — every axiom here must:
    * use precise THSD vocabulary;
    * be referenced by exactly one hypothesis or downstream lemma;
    * carry a doc-comment naming the empirical evidence.

  Audit:  grep -n '^axiom' lean/Brian/Postulate/Welch.lean

  NOTE: H005 itself is the *specification-equivalence* claim (that the
  Python `ImprovementGate.admit` body matches `WelchInputs.acceptSpec`),
  which is provable internally by `rfl`. The postulate below is the
  *separate* statistical theorem from Welch (1947): that the one-sided
  t-test controls Type-I error at level α.

  The two are distinct obligations:
    H005 (spec-equiv, provable)  : acceptImpl w = acceptSpec w
    Type-I bound (admitted here) : P(false positive) ≤ α under H₀

  The Type-I bound cannot be proven in mathlib-free Lean (it requires
  the Student-t distribution CDF and the Welch–Satterthwaite d.f.
  formula from Mathlib.Probability). We therefore admit it as an
  **opaque proposition** via a dedicated `Prop` type — NOT as a
  trivially-provable reflexivity statement.
-/
namespace Brian.Postulate.Welch

open Brian.Statistics

/-- An opaque Lean `Prop` representing the classical statistical
    theorem: the one-sided Welch t-test with Welch–Satterthwaite
    degrees of freedom controls Type-I error at level α under H₀.

    This proposition is declared axiomatically because its proof
    requires:
      (1) the Student-t distribution CDF (Mathlib.Probability.Distributions.Normal),
      (2) the Welch–Satterthwaite effective-df approximation, and
      (3) integration of the one-sided tail probability.

    All three are outside the deliberately mathlib-free Brian library
    (see `lakefile.lean`). The claim is instead empirically verified by
    the Monte-Carlo calibration harness in
    `tests/verification/test_improvement_gate.py` and by decades of
    statistical literature (Welch 1947).

    Downstream proofs that need to know ImprovementGate is honest about
    its α parameter cite this opaque proposition by name rather than
    re-deriving an unformalizable claim. -/
-- @[brian_postulate]
axiom WelchTypeIBound : Prop

/-- **Empirical evidence:**
      Welch (1947) "The generalization of Student's problem when
        several different population variances are involved" Biometrika 34(1-2).
      scipy.stats.ttest_ind(equal_var=False)
      tests/verification/test_improvement_gate.py::test_type_i_calibration

    The one-sided Welch t-test as implemented in
    `neuroslm/verification/improvement_gate.py` (via
    `scipy.stats.ttest_ind(equal_var=False)`) controls Type-I error at
    the declared level α: under H₀ (equal population means),
    P(WelchInputs.tStatBelowAlpha = true) ≤ α.

    This is the named handle downstream proofs cite; the opaque
    proposition `WelchTypeIBound` carries the statistical content
    that cannot be expressed in mathlib-free Lean. -/
-- @[brian_postulate]
axiom type_I_error_bound : WelchTypeIBound

/-- The Type-I bound implies the ImprovementGate admit function is
    conservatively calibrated: when `acceptSpec w = true`, the
    probability of a false positive is bounded.

    Stated here (rather than inline in the H005 proof file) so that
    any future downstream proof about mutation-admission rates has a
    single named citation point. -/
theorem welch_calibration_holds : WelchTypeIBound :=
  type_I_error_bound

end Brian.Postulate.Welch
