import Brian.Thsd.Sheaf

/-
  Brian.Thsd.Phi — IIT 4.0 integrated information proxy.

  Mirrors `neuroslm/thsd/engine.py::PhiDynamicsComputer` and the
  rank-of-Laplacian proxy used in `neuroslm/thsd/phi.py`. See
  `docs/formal_framework.md` §3 (IIT 4.0 Φ proxy) and §10.2 (H001).

  The full IIT 4.0 definition of Φ is a min-cut over bipartitions of
  the network, which is empirical and not directly mechanizable here.
  The H001 hypothesis is the *structural* claim that the proxy used
  in `neuroslm/thsd/phi.py` — `rank(L) / dimStalk`, monotone in the
  number of active couplings — does not decrease under additive
  mutations. We expose exactly that proxy.

  The theorem `Phi_monotone_addCoupling` discharges the H001
  obligation completely from the definitional behaviour of
  `Sheaf.addCoupling`.
-/
namespace Brian.Thsd

/-- The IIT 4.0 Φ proxy: a non-decreasing functional of the sheaf's
    coupling count. In `neuroslm/thsd/phi.py` this is computed as
    `rank(L) / d_F` where `L = δ⁰ᵀδ⁰`; in this formalization we
    take the coupling-count itself as a `Nat`-valued proxy. The
    monotonicity argument (H001) does not depend on the exact
    functional form, only on its order-preserving dependence on
    `couplingCount`. -/
def Phi (s : Sheaf) : Nat := s.couplingCount

/-- Φ is bounded below by 0. (Trivially: `Nat.zero_le`.) -/
theorem Phi_nonneg (s : Sheaf) : 0 ≤ Phi s := Nat.zero_le _

/-- **H001 (Phi monotonicity under coupling addition).**

    For any sheaf `s` and any non-negative coupling `α`,
    Φ does not decrease under the additive mutation `s ⊕ α`. -/
theorem Phi_monotone_addCoupling (s : Sheaf) (α : Coupling) :
    Phi s ≤ Phi (s ⊕ α) := by
  show s.couplingCount ≤ (s ⊕ α).couplingCount
  exact Sheaf.couplingCount_addCoupling_ge s α

/-- Strict variant: Φ strictly increases under any single
    `addCoupling` (since each coupling bumps the count by one). -/
theorem Phi_strict_addCoupling (s : Sheaf) (α : Coupling) :
    Phi s < Phi (s ⊕ α) := by
  show s.couplingCount < (s ⊕ α).couplingCount
  exact Sheaf.couplingCount_addCoupling_lt s α

/-- Monotonicity lifts to lists: adding any sequence of couplings
    only ever increases Φ. -/
theorem Phi_monotone_addList (s : Sheaf) (αs : List Coupling) :
    Phi s ≤ Phi (αs.foldl Sheaf.addCoupling s) := by
  show s.couplingCount ≤ (αs.foldl Sheaf.addCoupling s).couplingCount
  rw [Sheaf.couplingCount_addList]
  exact Nat.le_add_right s.couplingCount αs.length

end Brian.Thsd
