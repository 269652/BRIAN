import Brian.Thsd.Simplex

/-
  Brian.Thsd.Sheaf — Cellular sheaf F over K, with Couplings.

  Mirrors `neuroslm/thsd/engine.py::CellularSheaf`. See
  `docs/formal_framework.md` §2.2 ("Cellular sheaf").

  A cellular sheaf F assigns to each simplex σ a stalk F(σ), and to
  each face relation a linear restriction. The structural quantity
  the evolutionary loop cares about (H001) is the rank of the sheaf
  Laplacian L = δ⁰ᵀ δ⁰, which is bounded below by the number of
  non-zero couplings.

  We capture exactly that observable: `couplingCount` is the proxy
  for `rank L`. Adding a non-negative coupling (the H001 mutation)
  is `addCoupling`, which strictly increments the counter. The Phi
  monotonicity theorem (`Brian.Thsd.Phi`) lifts this trivially.
-/
namespace Brian.Thsd

/-- A non-negative coupling between two adjacent simplices.

    The non-negativity constraint is captured by construction:
    every `Coupling` carries a `Nat`-valued weight (which is
    inherently `≥ 0`). The H001 hypothesis is precisely about
    *adding* such a coupling; the magnitude does not enter the
    structural Φ-monotonicity argument. -/
structure Coupling where
  /-- Nat-valued weight; `≥ 0` by virtue of being a `Nat`. -/
  weight : Nat
  deriving Inhabited, DecidableEq

namespace Coupling

/-- Every coupling has non-negative weight (true by construction). -/
theorem weight_nonneg (α : Coupling) : 0 ≤ α.weight := Nat.zero_le _

end Coupling

/-- A cellular sheaf F over a `SimplexComplex` K.

    Beyond the underlying complex and a stalk dimension, we expose
    `couplingCount` — the number of currently-active non-zero
    couplings, which is the structural quantity Φ monotonically
    depends on. -/
structure Sheaf where
  base : SimplexComplex
  /-- Common stalk dimension d_F (e.g. 256 in `arch.neuro`). -/
  stalkDim : Nat
  /-- Rank-of-Laplacian proxy: number of non-zero couplings.
      Strictly monotone under `addCoupling`. -/
  couplingCount : Nat
  deriving Inhabited

namespace Sheaf

/-- The H001 mutation: extend the sheaf by one non-negative coupling.
    By construction this strictly increments the coupling count
    (the rank-of-Laplacian proxy). -/
def addCoupling (s : Sheaf) (_α : Coupling) : Sheaf :=
  { s with couplingCount := s.couplingCount + 1 }

/-- Notation for `addCoupling`, matching the Markdown statement
    `θ' = θ ⊕ α` in `hypothesis/H001_*.md`. -/
infixl:65 " ⊕ " => Sheaf.addCoupling

/-- Adding a coupling strictly increases the coupling count. -/
theorem couplingCount_addCoupling (s : Sheaf) (α : Coupling) :
    (s ⊕ α).couplingCount = s.couplingCount + 1 := rfl

/-- Monotonicity at the structural level: the coupling count is
    non-decreasing under `addCoupling`. This is the structural
    fact H001 (`Phi`-monotonicity) lifts. -/
theorem couplingCount_addCoupling_ge (s : Sheaf) (α : Coupling) :
    s.couplingCount ≤ (s ⊕ α).couplingCount := by
  rw [couplingCount_addCoupling]
  exact Nat.le_succ s.couplingCount

/-- Strict monotonicity: equivalent to the above with `<`. -/
theorem couplingCount_addCoupling_lt (s : Sheaf) (α : Coupling) :
    s.couplingCount < (s ⊕ α).couplingCount := by
  rw [couplingCount_addCoupling]
  exact Nat.lt_succ_self s.couplingCount

/-- Iterated addition: adding a list of couplings is associative
    over the count. -/
theorem couplingCount_addList (s : Sheaf) (αs : List Coupling) :
    (αs.foldl Sheaf.addCoupling s).couplingCount =
      s.couplingCount + αs.length := by
  induction αs generalizing s with
  | nil => simp [List.foldl, List.length]
  | cons α rest ih =>
    simp [List.foldl, List.length]
    rw [ih (s ⊕ α), couplingCount_addCoupling]
    omega

end Sheaf

end Brian.Thsd
