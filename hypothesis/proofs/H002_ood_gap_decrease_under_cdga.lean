/-
  Brian -- Hypothesis H002 proof.

  Title:       OOD gap decrease under CDGA
  Theorem:     Brian.OodGapDecrease
  Obligation:  oodGap (apply theta lam) <= oodGap theta  for lam >= 0.

  THSD vocabulary (CLAUDE.md section 12.1):
    Brian.Cdga.CdgaRegularizer  bundled contractivity guarantee
    Brian.Cdga.CdgaRegularizer.ood_gap_decrease  the contraction theorem

  Spec:        docs/formal_framework.md section 10.2 (H002 row)
               docs/CDGA.md (P1: gap-non-increase property)
  Code refs:   neuroslm/regularizers.py::cdga_loss
  Tests:       tests/test_cdga_smoke.py

  Proof strategy:
    Contraction is BUNDLED into the CdgaRegularizer structure as
    the gap_monotone field. The theorem unpacks that field.

    The empirical content (the concrete cdga_loss implementation
    really IS such a regularizer) lives in the postulate
    Brian.Postulate.Cdga.cdga_regularizer_exists.

  Postulates used:
    Brian.Postulate.Cdga.cdga_regularizer_exists
      (at instantiation time only, not in the proof body)
-/
import Brian.Core

open Brian.Cdga

namespace Brian

/-- H002: OOD gap decrease under CDGA. -/
theorem OodGapDecrease :
    ∀ {Params : Type} (R : CdgaRegularizer Params)
      (theta : Params) (lam : Nat),
      R.oodGap (R.apply theta lam) ≤ R.oodGap theta :=
  fun R theta lam => R.ood_gap_decrease theta lam

/-- Trivial lam = 0 witness: no postulate needed. -/
theorem OodGapDecrease_trivial (g : Nat → Nat) (theta : Nat) :
    let R := CdgaRegularizer.trivial Nat g
    R.oodGap (R.apply theta 0) ≤ R.oodGap theta :=
  (CdgaRegularizer.trivial Nat g).ood_gap_decrease theta 0

end Brian
