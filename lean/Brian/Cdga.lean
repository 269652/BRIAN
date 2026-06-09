import Brian.Postulate

/-
  Brian.Cdga — Cross-Distribution Gradient Alignment regularizer.

  Mirrors `neuroslm/regularizers.py::cdga_loss` and `docs/CDGA.md`.
  See `docs/formal_framework.md` §10.2 (H002).

  CDGA is a regularizer of the form
      L_total(θ) = L_base(θ) + λ · L_CDGA(θ)
  with λ ≥ 0. The H002 claim is that turning on the CDGA term
  cannot *widen* the OOD generalisation gap:
      Δ_OOD(θ + λ · ∇L_CDGA) ≤ Δ_OOD(θ),     λ ≥ 0.

  In Lean we capture this contract by *bundling* the monotonicity
  guarantee into the regularizer's type. Any term of type
  `CdgaRegularizer Params` carries, as a structure field, the
  proof that its gradient step satisfies the contractive property.
  Constructing such a term is the obligation — that obligation is
  discharged in `Brian.Postulate.Cdga.cdga_regularizer_for` for the
  concrete `neuroslm/regularizers.py` implementation. -/
namespace Brian.Cdga

/-- A CDGA regularizer parameterised by an opaque `Params` type
    standing in for the model parameters.

    The regularizer carries:
      * the gradient-step operator `apply θ λ` modelling
        `θ + λ · ∇L_CDGA(θ)`;
      * the OOD gap functional `oodGap : Params → Nat` (we use
        `Nat` to keep ordering reasoning decidable; in reality
        this is a real-valued loss difference);
      * the contractive guarantee `gap_monotone` — the proof
        that for any λ ≥ 0, applying the step cannot widen the
        gap. -/
structure CdgaRegularizer (Params : Type) where
  /-- One gradient step of the CDGA term.
      Models `θ ↦ θ + λ · ∇L_CDGA(θ)`. -/
  apply : Params → Nat → Params
  /-- OOD generalisation gap functional Δ_OOD : Params → ℕ. -/
  oodGap : Params → Nat
  /-- **H002 obligation, bundled.** For any λ, applying the CDGA
      gradient step cannot widen the OOD gap. -/
  gap_monotone : ∀ (θ : Params) (lam : Nat),
                   oodGap (apply θ lam) ≤ oodGap θ

namespace CdgaRegularizer

variable {Params : Type}

/-- The H002 statement, restated as a function of an arbitrary
    `CdgaRegularizer`. The proof is a direct unpacking of the
    bundled `gap_monotone` field. -/
theorem ood_gap_decrease (R : CdgaRegularizer Params)
    (θ : Params) (lam : Nat) :
    R.oodGap (R.apply θ lam) ≤ R.oodGap θ :=
  R.gap_monotone θ lam

/-- λ = 0 case: a trivial CDGA step (no regularization) is the
    identity on parameters, hence trivially contractive. This is
    a constructive witness that the type `CdgaRegularizer` is
    inhabited without invoking any postulate. -/
def trivial (P : Type) (g : P → Nat) : CdgaRegularizer P :=
  { apply        := fun θ _ => θ
  , oodGap       := g
  , gap_monotone := fun _ _ => Nat.le_refl _
  }

end CdgaRegularizer

end Brian.Cdga
