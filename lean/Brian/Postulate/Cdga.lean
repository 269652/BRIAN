import Brian.Postulate
import Brian.Cdga

/-
  Brian.Postulate.Cdga — empirical admission for H002.

  CLAUDE.md §12.2 — every axiom here must:
    * use precise THSD vocabulary;
    * be referenced by exactly one hypothesis (here: H002);
    * carry a doc-comment naming the empirical evidence.

  Audit:  grep -n '^axiom' lean/Brian/Postulate/Cdga.lean
-/
namespace Brian.Postulate.Cdga

open Brian.Cdga

/-- **Empirical evidence:**
      tests/test_cdga_smoke.py
      neuroslm/regularizers.py::cdga_loss
      docs/CDGA.md (P1)

    The CDGA regularizer implemented in `neuroslm/regularizers.py`
    is contractive on the OOD generalisation gap: for any
    parameters θ and any non-negative coefficient λ,

        Δ_OOD(θ + λ · ∇L_CDGA(θ))  ≤  Δ_OOD(θ).

    The Python smoke test `test_cdga_smoke.py` exercises this
    bound numerically on a held-out OOD split; we admit the
    bound here as a postulate because it is a property of the
    *concrete* `neuroslm/regularizers.py` implementation and the
    *concrete* data distributions, not a Lean-internal theorem.

    A future tighter postulate would isolate the formal
    sufficient conditions (gradient alignment ≥ 0, λ within the
    contractive regime); for now we admit the empirical claim
    directly as the universal contraction property. -/
axiom cdga_regularizer_exists :
    ∀ (Params : Type) (oodGap : Params → Nat)
      (step : Params → Nat → Params),
      (∀ θ lam, oodGap (step θ lam) ≤ oodGap θ) →
      CdgaRegularizer Params

/-- A concrete CDGA regularizer for an arbitrary `Params` type,
    constructed from the empirical contraction postulate. This is
    the builder H002's proof invokes. -/
noncomputable def cdgaRegularizerFor
    (Params : Type) (oodGap : Params → Nat)
    (step : Params → Nat → Params)
    (h : ∀ θ lam, oodGap (step θ lam) ≤ oodGap θ) :
    CdgaRegularizer Params :=
  cdga_regularizer_exists Params oodGap step h

end Brian.Postulate.Cdga
