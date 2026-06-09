import Brian.Core

/-
  Brian library smoke tests — exercises the THSD primitives in
  isolation so a regression in `Brian/Thsd/*.lean` shows up here
  before it breaks the hypothesis proofs.

  Run with:  lake build BrianTest
-/

namespace Brian.Test

open Brian.Thsd
open Brian.Cdga
open Brian.Verification
open Brian.Statistics

/-! ## Sheaf + Coupling -/

private def sampleSheaf : Sheaf :=
  { base := { dimMax := 1, numVertices := 3, numEdges := 2 }
  , stalkDim := 8
  , couplingCount := 5
  }

private def alpha : Coupling := { weight := 1 }

example : sampleSheaf.couplingCount = 5 := rfl
example : (sampleSheaf ⊕ alpha).couplingCount = 6 := rfl

example : sampleSheaf.couplingCount ≤ (sampleSheaf ⊕ alpha).couplingCount :=
  Sheaf.couplingCount_addCoupling_ge sampleSheaf alpha

example : sampleSheaf.couplingCount < (sampleSheaf ⊕ alpha).couplingCount :=
  Sheaf.couplingCount_addCoupling_lt sampleSheaf alpha

/-! ## Phi monotonicity (H001 ingredient) -/

example : Phi sampleSheaf ≤ Phi (sampleSheaf ⊕ alpha) :=
  Phi_monotone_addCoupling sampleSheaf alpha

example : Phi sampleSheaf < Phi (sampleSheaf ⊕ alpha) :=
  Phi_strict_addCoupling sampleSheaf alpha

example :
    Phi sampleSheaf ≤
      Phi ([alpha, alpha, alpha].foldl Sheaf.addCoupling sampleSheaf) :=
  Phi_monotone_addList sampleSheaf [alpha, alpha, alpha]

/-! ## CDGA contractivity (H002 ingredient) — trivial regularizer -/

private def gap (n : Nat) : Nat := n
private def trivReg : CdgaRegularizer Nat := CdgaRegularizer.trivial Nat gap

example : trivReg.oodGap (trivReg.apply 7 3) ≤ trivReg.oodGap 7 :=
  trivReg.ood_gap_decrease 7 3

/-! ## Triple-Guard soundness (H004) -/

private def guard : TripleGuardSpec Nat :=
  { phiPasses := fun n => n > 0
  , h1Passes  := fun _ => True
  , lambdaPasses := fun n => n > 0
  }

example : guard.accept 5 ↔ (5 > 0) ∧ True ∧ (5 > 0) :=
  TripleGuardSpec.accept_iff_all_pass guard 5

example : guard.accept 5 := by
  -- `guard.phiPasses 5` definitionally unfolds (via struct projection
  -- + beta) to `5 > 0 = 0 < 5`. `by decide` can't synthesize the
  -- Decidable instance through the projection, but `by exact` lets the
  -- elaborator drive the unification and pick up the constructor
  -- proof `Nat.succ_pos 4 : 0 < 5`.
  exact ⟨Nat.succ_pos 4, True.intro, Nat.succ_pos 4⟩

/-! ## Welch spec equivalence (H005) -/

private def w : WelchInputs :=
  { tStatBelowAlpha := true
  , effectAboveMin := true
  , directionHolds := true
  }

example : WelchInputs.acceptImpl w = WelchInputs.acceptSpec w :=
  WelchInputs.acceptImpl_eq_acceptSpec w

example : WelchInputs.acceptSpec w = true := rfl

end Brian.Test
