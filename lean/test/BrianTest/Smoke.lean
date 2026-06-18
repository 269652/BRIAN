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
open Brian.NN

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

/-! ## Brian.NN — LinearLayer smoke tests (H018 ingredient) -/

private def readoutLayer : LinearLayer := LinearLayer.zeroInit 64 64

-- A zero-init layer is zero-contributing.
example : readoutLayer.isZeroContrib = true :=
  LinearLayer.zeroInit_layer_output_is_zero 64 64

-- Zero-contributing with explicit `isZeroInit` witness.
example : readoutLayer.isZeroInit = true := rfl

-- zeroInit_output_is_zero works for any well-formed zero layer.
example : readoutLayer.isZeroContrib = true :=
  LinearLayer.zeroInit_output_is_zero readoutLayer rfl

/-! ## Brian.NN — ResidualUpdate smoke tests (H018 ingredient) -/

private def nfoUpdate : ResidualUpdate :=
  ResidualUpdate.nfoZeroInit 64 64 true

-- NFO zero-init update is always identity.
example : nfoUpdate.isIdentity = true :=
  ResidualUpdate.nfoZeroInit_is_identity 64 64 true

-- Also works when alpha is zero.
example : (ResidualUpdate.nfoZeroInit 64 64 false).isIdentity = true :=
  ResidualUpdate.nfoZeroInit_is_identity 64 64 false

/-! ## Brian.NN — CoherenceGate smoke tests (H016 ingredient) -/

private def cg : CoherenceGate :=
  { R := 3, R_max := 5, hMax := by norm_num }

-- Gate ≤ R_max.
example : cg.apply ≤ cg.R_max :=
  CoherenceGate.apply_le_R_max cg

-- Identity at uniform coherence (R = R_max).
private def cgUniform : CoherenceGate :=
  { R := 7, R_max := 7, hMax := Nat.le_refl 7 }

example : cgUniform.apply = cgUniform.R_max :=
  CoherenceGate.apply_identity_at_max cgUniform rfl

-- Zero R ⇒ zero gate.
private def cgZero : CoherenceGate :=
  { R := 0, R_max := 5, hMax := Nat.zero_le 5 }

example : cgZero.apply = 0 :=
  CoherenceGate.apply_zero_when_R_zero cgZero rfl

-- Monotone in R.
private def cg1 : CoherenceGate := { R := 2, R_max := 8, hMax := by norm_num }
private def cg2 : CoherenceGate := { R := 5, R_max := 8, hMax := by norm_num }

example : cg1.apply ≤ cg2.apply :=
  CoherenceGate.apply_monotone cg1 cg2 rfl (by norm_num)

end Brian.Test
