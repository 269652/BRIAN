/-
  Brian -- Hypothesis H004 proof.

  Title:       Triple-Guard soundness
  Theorem:     Brian.TripleGuardSound
  Obligation:  TripleGuard accepts iff all three sub-guards
               (Phi, H1, lambda) hold.

  THSD vocabulary (CLAUDE.md section 12.1):
    Brian.Verification.TripleGuardSpec         three sub-guards
    Brian.Verification.TripleGuardSpec.accept  conjunction
    Brian.Verification.TripleGuardSpec.accept_iff_all_pass

  Spec:        docs/formal_framework.md section 10.2 (H004 row)
               docs/formal_framework.md section 6.4 (Triple-Guard linter)
  Code refs:   neuroslm/verification/triple_guard.py
  Tests:       tests/training/test_rcc_bowtie_triple_guard.py

  Proof strategy:
    TripleGuardSpec.accept is DEFINED as the conjunction of the
    three sub-guards. Soundness is therefore Iff.rfl --
    the gate's specification IS its definition.

  Postulates used: NONE.
-/
import Brian.Core

open Brian.Verification

namespace Brian

/-- H004: Triple-Guard soundness. -/
theorem TripleGuardSound :
    ∀ {Params : Type} (G : TripleGuardSpec Params) (theta : Params),
      G.accept theta ↔ G.phiPasses theta ∧ G.h1Passes theta
                        ∧ G.lambdaPasses theta :=
  fun G theta => TripleGuardSpec.accept_iff_all_pass G theta

/-- Forward direction packaged as three projections. -/
theorem TripleGuardSound_proj
    {Params : Type} (G : TripleGuardSpec Params) (theta : Params)
    (h : G.accept theta) :
    G.phiPasses theta ∧ G.h1Passes theta ∧ G.lambdaPasses theta :=
  (TripleGuardSpec.accept_iff_all_pass G theta).mp h

/-- Backward direction: three passes implies acceptance. -/
theorem TripleGuardSound_intro
    {Params : Type} (G : TripleGuardSpec Params) (theta : Params)
    (hPhi : G.phiPasses theta) (hH1 : G.h1Passes theta)
    (hLam : G.lambdaPasses theta) :
    G.accept theta :=
  TripleGuardSpec.all_pass_imp_accept G theta hPhi hH1 hLam

end Brian
