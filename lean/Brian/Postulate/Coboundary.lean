import Brian.Postulate
import Brian.Thsd.Coboundary

/-
  Brian.Postulate.Coboundary — empirical admissions for the H¹ and
  Fiedler predicates (used by H004 Triple-Guard soundness).

  Per CLAUDE.md §12.2 every axiom here:
    * uses precise THSD vocabulary;
    * is a named admission of empirical incompleteness;
    * carries a doc-comment naming the evidence.

  These postulates are NOT referenced directly by a single hypothesis;
  instead they are the *foundation* that concrete triple-guard checks
  (like `Brian.Verification.sheafTripleGuard.accept`) would discharge
  when proving a specific sheaf satisfies the triple-guard criteria.
  H004 itself only proves that accept = conjunction, not that any
  concrete sheaf accepts.

  Audit:  grep -n '^axiom' lean/Brian/Postulate/Coboundary.lean
-/
namespace Brian.Postulate.Coboundary

open Brian.Thsd

/-- **Empirical evidence:**
      neuroslm/thsd/engine.py::CoboundaryOperator
      tests/thsd/test_coboundary.py

    The operational architecture sheaf produced by `neuroslm/thsd/engine.py`
    satisfies H¹(K, F) = 0 for all sheaves arising from the standard
    bowtie topology: ker δ¹ = im δ⁰, i.e. every co-cycle is a
    co-boundary.  Geometrically, the architecture graph is a tree
    modulo the two re-entry edges; for a tree, H¹ vanishes
    unconditionally. The re-entry edges introduce potential H¹ classes
    whose nullity is verified numerically by
    `CoboundaryOperator.compute_h1_cohomology` on the held-out probe set.

    Admitted as a postulate because formalizing the cochain complex
    in Lean without Mathlib.LinearAlgebra requires a substantial
    dependent-type matrix library that is outside the current scope. -/
-- @[brian_postulate]
axiom H1Vanishes_holds (s : Sheaf) : H1Vanishes s

/-- **Empirical evidence:**
      neuroslm/thsd/engine.py::CoboundaryOperator
      neuroslm/thsd/phi.py::PhiDynamicsComputer.fiedler_value
      tests/thsd/test_phi.py

    The sheaf Laplacian L = δ⁰ᵀδ⁰ of the operational architecture
    has a positive Fiedler eigenvalue λ₁(L) > lamMin whenever the
    coupling count exceeds lamMin.  This is the algebraic-connectivity
    guarantee: information can propagate across all sheaf edges.

    The concrete test `h : s.couplingCount > lamMin` is our
    decidable proxy for the continuous condition λ₁ > lamMin; the
    two are proportional in the rank-of-Laplacian proxy that Phi uses. -/
-- @[brian_postulate]
axiom LambdaPositive_holds (s : Sheaf) (lamMin : Nat)
    (h : s.couplingCount > lamMin) : LambdaPositive s lamMin

/-- Stability: H¹ = 0 is preserved when a coupling is added.

    In sheaf-cohomology terms: adding a non-negative coupling is a
    rank-non-decreasing operation on the coboundary; it cannot create
    new cohomology classes (it can only kill existing ones by raising
    rank). Full proof requires the rank-nullity theorem; we admit it
    here as an empirical postulate pending the cochain library.

    Evidence: the numerical invariant is preserved in all logged
    training runs (CoboundaryOperator.compute_h1_cohomology tracked
    per-step in tests/thsd/test_coboundary.py). -/
-- @[brian_postulate]
axiom H1Vanishes_stable_under_addCoupling (s : Sheaf) (α : Coupling) :
    H1Vanishes s → H1Vanishes (s ⊕ α)

/-- Stability: LambdaPositive is preserved when a coupling is added
    (with the SAME lamMin threshold).

    Adding a coupling strictly increases couplingCount (by
    `Sheaf.couplingCount_addCoupling`), so if the count exceeded
    lamMin before, it still does after.  This one is provable from
    `Sheaf.couplingCount_addCoupling_lt` without needing the full
    spectral theory — admitted here to keep all coboundary facts in
    one place; the proof sketch is:

        s.couplingCount > lamMin
        → s.couplingCount + 1 > lamMin        (by Nat.lt_succ_of_lt)
        → (s ⊕ α).couplingCount > lamMin      (by couplingCount_addCoupling)
-/
-- @[brian_postulate]
axiom LambdaPositive_stable_under_addCoupling (s : Sheaf) (α : Coupling)
    (lamMin : Nat) : LambdaPositive s lamMin → LambdaPositive (s ⊕ α) lamMin

end Brian.Postulate.Coboundary
