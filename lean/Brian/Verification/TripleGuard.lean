import Brian.Thsd.Sheaf
import Brian.Thsd.Phi
import Brian.Thsd.Coboundary

/-
  Brian.Verification.TripleGuard — three-conjunct admission gate.

  Mirrors `neuroslm/verification/triple_guard.py`. See
  `docs/formal_framework.md` §6.4 ("Triple-Guard linter") and
  §10.2 (H004).

  The Triple Guard accepts a mutation iff *all three* sub-guards pass:

      accept(θ) ⟺ Φ(θ) > Φ_min  ∧  H¹(K, F) = 0  ∧  λ₁(L) > λ_min

  H004 is the **soundness** claim — that the Python predicate is
  *exactly* the conjunction of its three sub-predicates. Because we
  *define* `accept` as that conjunction, soundness is `Iff.rfl`.

  The method is named `accept` rather than `admit` because `admit`
  is a Lean-3 proof-evading tactic that the CLAUDE.md §12 lint
  bans — naming a regular method `admit` would force the lint to
  weave around context which is brittle.
-/
namespace Brian.Verification

open Brian.Thsd

/-- The three pieces of evidence the triple guard requires. -/
structure TripleGuardSpec (Params : Type) where
  /-- The Φ-guard predicate `Φ(θ) > Φ_min`. -/
  phiPasses : Params → Prop
  /-- The cohomology guard `H¹(K, F) = 0`. -/
  h1Passes : Params → Prop
  /-- The Fiedler guard `λ₁(L) > λ_min`. -/
  lambdaPasses : Params → Prop

namespace TripleGuardSpec

variable {Params : Type}

/-- The acceptance predicate: conjunction of the three sub-guards. -/
def accept (G : TripleGuardSpec Params) (θ : Params) : Prop :=
  G.phiPasses θ ∧ G.h1Passes θ ∧ G.lambdaPasses θ

/-- **H004 (Triple-Guard soundness).** The acceptance predicate
    holds iff every sub-guard passes. Provable by definitional
    unfolding — soundness IS the definition. -/
theorem accept_iff_all_pass (G : TripleGuardSpec Params) (θ : Params) :
    G.accept θ ↔ G.phiPasses θ ∧ G.h1Passes θ ∧ G.lambdaPasses θ :=
  Iff.rfl

/-- Forward direction: acceptance implies each sub-guard. -/
theorem accept_phi (G : TripleGuardSpec Params) (θ : Params)
    (h : G.accept θ) : G.phiPasses θ := h.1

theorem accept_h1 (G : TripleGuardSpec Params) (θ : Params)
    (h : G.accept θ) : G.h1Passes θ := h.2.1

theorem accept_lambda (G : TripleGuardSpec Params) (θ : Params)
    (h : G.accept θ) : G.lambdaPasses θ := h.2.2

/-- Backward direction: the three sub-guards together suffice for
    acceptance. -/
theorem all_pass_imp_accept (G : TripleGuardSpec Params) (θ : Params)
    (hPhi : G.phiPasses θ) (hH1 : G.h1Passes θ) (hLam : G.lambdaPasses θ) :
    G.accept θ := ⟨hPhi, hH1, hLam⟩

end TripleGuardSpec

/-- The canonical sheaf-based triple guard: Φ > 0, H¹ vanishes, and
    the coupling count exceeds a Fiedler-floor proxy. This is the
    instance the H001 + H004 chain relies on. -/
def sheafTripleGuard (phiMin : Nat) (lamMin : Nat) :
    TripleGuardSpec Sheaf :=
  { phiPasses   := fun s => phiMin < Phi s
  , h1Passes    := fun s => H1Vanishes s
  , lambdaPasses := fun s => LambdaPositive s lamMin
  }

end Brian.Verification
