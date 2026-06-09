/-
  Brian -- Hypothesis H003 proof.

  Title:       Symbolic sparsity collapse
  Theorem:     Brian.SymbolicSparsity
  Obligation:  Annealed symbolic unit emits non-identity Expr
               once its inverse temperature is past the threshold.

  THSD vocabulary (CLAUDE.md section 12.1):
    Brian.Thsd.Symbolic.Expr          symbolic expression ADT
    Brian.Thsd.Symbolic.AnnealedUnit  post-collapse unit with the
                                      sparsity guarantee BUNDLED

  Spec:        docs/formal_framework.md section 10.2 (H003 row)
               docs/formal_framework.md section 3 (Symbolic Expr Units)
  Code refs:   neuroslm/thsd/engine.py::SymbolicSimplex
               neuroslm/thsd/symbolic.py
  Tests:       tests/thsd/test_symbolic.py
               tests/thsd/test_symbolic_simplex.py

  Proof strategy:
    The sparsity-collapse property is BUNDLED into the
    AnnealedUnit.collapsed field. The theorem unpacks that field.

    The empirical content (Gumbel-Softmax really does collapse as
    tau -> 0) lives in the postulate
    Brian.Postulate.Symbolic.gumbel_softmax_collapses_for.

  Postulates used:
    Brian.Postulate.Symbolic.gumbel_softmax_collapses_for
      (at instantiation time only, not in the proof body)
-/
import Brian.Core

open Brian.Thsd.Symbolic

namespace Brian

/-- H003: Symbolic sparsity collapse. -/
theorem SymbolicSparsity :
    ∀ (u : AnnealedUnit),
      u.unit.tempInv ≥ u.criticalInv → u.unit.current ≠ Expr.identity :=
  fun u h => u.collapsed h

/-- Constructive witness: AnnealedUnit is inhabited. -/
theorem SymbolicSparsity_inhabited :
    (AnnealedUnit.ofNontrivial "tanh").unit.current ≠ Expr.identity :=
  (AnnealedUnit.ofNontrivial "tanh").collapsed (Nat.le_refl 1)

end Brian
