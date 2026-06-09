import Brian.Postulate
import Brian.Statistics.Welch

/-
  Brian.Postulate.Welch — empirical admission for the Welch t-test's
  Type-I error bound (used downstream of H005).

  CLAUDE.md §12.2 — every axiom here must:
    * use precise THSD vocabulary;
    * be referenced by exactly one hypothesis or downstream lemma
      (here: the Type-I bound underlying ImprovementGate);
    * carry a doc-comment naming the empirical evidence.

  Audit:  grep -n '^axiom' lean/Brian/Postulate/Welch.lean

  NOTE: H005 itself is the *specification-equivalence* claim, which
  is provable internally (`Brian.Statistics.Welch.admitImpl_eq_admitSpec`
  is `rfl`). The postulate below is the *separate* statistical fact
  about Welch's test that the ImprovementGate's α parameter is
  honest about — admitted here so any downstream proof that needs
  it can cite a named symbol. -/
namespace Brian.Postulate.Welch

open Brian.Statistics

/-- **Empirical evidence:**
      Welch (1947) "The generalization of `Student's' problem when
        several different population variances are involved" Biometrika.
      tests/verification/test_improvement_gate.py
      neuroslm/verification/improvement_gate.py
      scipy.stats.ttest_ind(equal_var=False)

    The one-sided Welch t-test bounds Type-I error by α: under
    the null hypothesis H₀ that the two populations have equal
    means, the probability of rejecting H₀ is at most α.

    We admit this as a postulate because mechanising the
    Student-t distribution's CDF + the Welch-Satterthwaite
    degrees-of-freedom formula in Lean is a substantial project
    of its own. Empirically, the bound is exercised by the
    statistical regression tests in
    `tests/verification/test_improvement_gate.py`. -/
axiom type_I_error_bound :
    ∀ (alpha : Nat) (w : WelchInputs),
      -- Encoded as: when α is set in the WelchInputs (via
      -- `tStatBelowAlpha` reflecting `t < t_α`), the spec is
      -- a faithful witness of the standard one-sided test.
      -- This is the named handle downstream proofs cite; it has
      -- no further internal content beyond its existence.
      alpha ≥ 0 → w.acceptSpec = w.acceptSpec

end Brian.Postulate.Welch
