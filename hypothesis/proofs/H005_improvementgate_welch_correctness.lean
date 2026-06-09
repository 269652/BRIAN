/-
  Brian -- Hypothesis H005 proof.

  Title:       ImprovementGate Welch correctness
  Theorem:     Brian.ImprovementGateWelch
  Obligation:  ImprovementGate.accept returns true iff
                 (t-stat below alpha) and (effect at or above min)
                 and (direction-of-effect holds).

  THSD vocabulary (CLAUDE.md section 12.1):
    Brian.Statistics.WelchInputs             the three booleans
    Brian.Statistics.WelchInputs.acceptSpec  mathematical spec
    Brian.Statistics.WelchInputs.acceptImpl  Lean mirror of the
                                             Python implementation
    Brian.Statistics.WelchInputs.acceptImpl_eq_acceptSpec

  Spec:        docs/formal_framework.md section 10.2 (H005 row)
               docs/formal_framework.md section 9 (ImprovementGate)
  Code refs:   neuroslm/verification/improvement_gate.py
  Tests:       tests/verification/test_improvement_gate.py

  Proof strategy:
    H005 is a SPECIFICATION-EQUIVALENCE claim. The deeper Type-I
    error bound from Welch (1947) is handled separately in
    Brian.Postulate.Welch.type_I_error_bound. The equivalence H005
    actually asserts is provable by rfl because acceptImpl and
    acceptSpec are definitionally equal.

  Postulates used: NONE in the proof body.
-/
import Brian.Core

open Brian.Statistics

namespace Brian

/-- H005: ImprovementGate Welch correctness. -/
theorem ImprovementGateWelch :
    ∀ (w : WelchInputs),
      WelchInputs.acceptImpl w = WelchInputs.acceptSpec w :=
  WelchInputs.acceptImpl_eq_acceptSpec

/-- Logical form of the spec: acceptance iff each conjunct. -/
theorem ImprovementGateWelch_iff :
    ∀ (w : WelchInputs),
      WelchInputs.acceptImpl w = true ↔
        (w.tStatBelowAlpha = true ∧ w.effectAboveMin = true
          ∧ w.directionHolds = true) := by
  intro w
  rw [WelchInputs.acceptImpl_eq_acceptSpec]
  exact WelchInputs.acceptSpec_eq_true_iff w

end Brian
