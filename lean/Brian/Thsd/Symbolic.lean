import Brian.Thsd.Simplex
import Brian.Postulate

/-
  Brian.Thsd.Symbolic — Symbolic Expression Units (discovery operator 1/4).

  Mirrors `neuroslm/thsd/engine.py::SymbolicSimplex` and the
  symbolic substrate in `neuroslm/thsd/symbolic.py`. See
  `docs/formal_framework.md` §3 ("Symbolic Expression Units") and
  §10.2 (H003).

  A symbolic unit emits an `Expr` chosen by a Gumbel-Softmax over a
  fixed dictionary of operators. As the annealing temperature τ → 0
  the distribution collapses to a one-hot, so the emitted Expr
  becomes a discrete program: the "sparsity collapse" we want
  formalised.

  In Lean we model the unit's *post-annealing* output directly: an
  `AnnealedUnit` is a `SymbolicUnit` paired with a critical
  temperature and a bundled proof that once `τ ≤ critical`, the
  emitted Expr is *not* the trivial `Expr.identity`. The bundled
  proof is the formal counterpart of the annealing-correctness
  postulate from `Brian.Postulate.Symbolic`.
-/
namespace Brian.Thsd.Symbolic

/-- A symbolic expression: either the trivial identity operator
    (the pre-collapse "diffuse" state) or a non-trivial operator
    (post-collapse). Real `SymbolicSimplex.symbolic_expression`
    returns a string; here we capture only the structural
    distinction needed for H003. -/
inductive Expr where
  /-- The trivial identity: f(x) = x. Pre-collapse default. -/
  | identity : Expr
  /-- A non-trivial operator (any specific symbol post-collapse). -/
  | nontrivial : String → Expr
  deriving DecidableEq, Inhabited

namespace Expr

/-- Predicate: this expression is the trivial identity. -/
def isIdentity : Expr → Bool
  | .identity => true
  | .nontrivial _ => false

/-- An expression is non-identity iff `isIdentity` is false. -/
theorem ne_identity_iff (e : Expr) : e ≠ .identity ↔ e.isIdentity = false := by
  cases e <;> simp [isIdentity]

end Expr

/-- A single symbolic unit before annealing: it carries its current
    emitted expression and a temperature. -/
structure SymbolicUnit where
  current : Expr
  /-- Annealing temperature τ > 0. We use Nat-encoded inverse:
      `tempInv = n` means τ = 1/n (so high `tempInv` = low τ). -/
  tempInv : Nat
  deriving Inhabited

/-- A unit whose annealing schedule has progressed past the
    collapse threshold. The `collapsed` field is the *bundled
    proof* that, at this stage, the emitted Expr is no longer the
    trivial identity.

    Constructing an `AnnealedUnit` therefore *requires* discharging
    the collapse property. The empirical content (that the
    Gumbel-Softmax distribution actually does collapse as τ → 0)
    lives in `Brian.Postulate.Symbolic.gumbel_softmax_collapses`,
    which is the canonical builder for this structure. -/
structure AnnealedUnit where
  unit : SymbolicUnit
  /-- Critical inverse-temperature: when `unit.tempInv ≥ criticalInv`
      the unit is considered post-collapse. -/
  criticalInv : Nat
  /-- Bundled annealing guarantee: once we are past critical, the
      emitted Expr is not the trivial identity. -/
  collapsed : unit.tempInv ≥ criticalInv → unit.current ≠ Expr.identity

namespace AnnealedUnit

/-- A directly-constructed non-trivial unit at maximum cooling.
    Used to exhibit constructively that the type is inhabited. -/
def ofNontrivial (sym : String) : AnnealedUnit :=
  { unit       := { current := .nontrivial sym, tempInv := 1 }
  , criticalInv := 1
  , collapsed := by
      intro _h
      intro hEq
      cases hEq
  }

end AnnealedUnit

end Brian.Thsd.Symbolic
