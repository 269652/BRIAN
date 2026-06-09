/-
  Brian -- Hypothesis H001 proof.

  Title:       Phi monotone under coupling addition
  Theorem:     Brian.PhiMonotone
  Obligation:  for every sheaf s and coupling a, Phi s <= Phi (s + a)

  THSD vocabulary (CLAUDE.md section 12.1):
    Brian.Thsd.Sheaf             cellular sheaf F over a SimplexComplex K
    Brian.Thsd.Sheaf.addCoupling the H001 mutation s + a
    Brian.Thsd.Phi               IIT 4.0 Phi proxy (rank-of-Laplacian)

  Spec:        docs/formal_framework.md section 10.2 (H001 row)
               docs/architecture.md section 5.5
  Code refs:   neuroslm/thsd/phi.py
               neuroslm/thsd/engine.py::PhiDynamicsComputer
               neuroslm/verification/triple_guard.py
  Tests:       tests/thsd/test_phi.py
               tests/training/test_rcc_bowtie_triple_guard.py

  Proof strategy:
    Phi is defined as s.couplingCount (the rank-of-Laplacian
    proxy). The H001 mutation addCoupling strictly increments
    that counter by definitional equality. Hence the result
    follows from Sheaf.couplingCount_addCoupling_ge, lifted to
    Phi via Phi_monotone_addCoupling in Brian.Thsd.Phi.

  Postulates used: NONE.
-/
import Brian.Core

open Brian.Thsd

namespace Brian

/-- H001: Phi monotonicity under coupling addition. -/
theorem PhiMonotone :
    ∀ (s : Sheaf) (a : Coupling), Phi s ≤ Phi (s ⊕ a) :=
  Phi_monotone_addCoupling

/-- A single additive mutation strictly increases Phi. -/
theorem PhiMonotone_strict :
    ∀ (s : Sheaf) (a : Coupling), Phi s < Phi (s ⊕ a) :=
  Phi_strict_addCoupling

/-- Iterated lift: any list of additive mutations is Phi-non-decreasing. -/
theorem PhiMonotone_list :
    ∀ (s : Sheaf) (as : List Coupling),
      Phi s ≤ Phi (as.foldl Sheaf.addCoupling s) :=
  Phi_monotone_addList

end Brian
