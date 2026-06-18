import Brian.Thsd.Sheaf

/-
  Brian.Thsd.Coboundary — δ⁰, δ¹ and the H¹ / Fiedler predicates.

  Mirrors `neuroslm/thsd/engine.py::CoboundaryOperator`. See
  `docs/formal_framework.md` §6.4 ("Triple-Guard linter").

  In the full theory δ^k : C^k(K, F) → C^{k+1}(K, F) is a linear
  cochain map and the cohomology group H¹(K, F) := ker δ¹ / im δ⁰
  measures algebraic inconsistency: a non-trivial H¹ class is an
  "obstruction to gluing local sections into a global one"
  (formal_framework.md §6.3).

  For the H004 triple-guard soundness theorem we need *predicate-level*
  abstractions `H1Vanishes K` and `LambdaPositive K lamMin`. These are
  declared as **opaque axioms** (not `True`) so that:

    1. Any proof using them must go through
       `Brian.Postulate.Coboundary.H1Vanishes_holds` /
       `Brian.Postulate.Coboundary.LambdaPositive_holds` — a named
       admission of the empirical content.
    2. The pattern `def H1Vanishes _ : Prop := True` is banned by
       CLAUDE.md §12.2 ("no `: True` obligations"); the opaque axiom
       form is the correct substitute.

  The concrete cochain machinery is in `neuroslm/thsd/engine.py`;
  the H004 proof is a tautology over the conjunction — what matters
  is that the three predicates are *distinct and non-trivial*.
-/
namespace Brian.Thsd

/-- Predicate: "the first cohomology of (K, F) vanishes."

    Concretely: ker δ¹ = im δ⁰ for the sheaf cochain complex
    C⁰ →^{δ⁰} C¹ →^{δ¹} C².  A non-trivial H¹ class is an
    obstruction to gluing local sections into a global one
    (docs/formal_framework.md §6.3).

    Declared as an opaque axiom — not `True` — so consumers must
    discharge it via `Brian.Postulate.Coboundary.H1Vanishes_holds`.
    The full cochain machinery will eventually make this a real theorem
    derived from `CoboundaryOperator`; for now the predicate exists as
    a named obligation in the THSD vocabulary. -/
-- @[brian_postulate]
axiom H1Vanishes : Sheaf → Prop

/-- Predicate: "the Fiedler eigenvalue λ₁(L) exceeds a threshold."

    `λ₁(L) > lamMin` where L = δ⁰ᵀδ⁰ is the sheaf Laplacian. A
    positive Fiedler value guarantees algebraic connectivity: every
    section can propagate across the sheaf graph.

    Declared as an opaque axiom — not `True` — so consumers must
    discharge it via
    `Brian.Postulate.Coboundary.LambdaPositive_holds`. -/
-- @[brian_postulate]
axiom LambdaPositive : Sheaf → Nat → Prop

end Brian.Thsd
