/-
  Brian -- Hypothesis H015 proof.

  Title:       NFO bipartition coherence is a closed-form Φ lower bound
  Theorem:     Brian.NfoBipartitionCoherenceLowerBound
  Obligation:  For every sheaf s and every bipartition with n cut edges,
                   Φ(s) + n  ≤  Φ(s ⊕^n α)
               where ⊕^n α is shorthand for "add n couplings". The
               coherence functional Φ_κ = mean(1 − R) of the Neural
               Field Oscillator (lib/blocks/neural_field_oscillator.neuro)
               is a monotone-decreasing image of the cut-edge count, so
               drops in Φ_κ are a closed-form witness of Φ increase.

  THSD vocabulary (CLAUDE.md section 12.1):
    Brian.Thsd.Sheaf             cellular sheaf F over a SimplexComplex K
    Brian.Thsd.Sheaf.addCoupling H001 mutation s + α
    Brian.Thsd.Sheaf.couplingCount_addList iterated form (H001 corollary)
    Brian.Thsd.Phi               IIT 4.0 Φ proxy (= couplingCount)
    Brian.Nfo.couplings_of_cut   cut-edge count → list of couplings
    Brian.Nfo.bipartition_coherence_phi_lower_bound (the obligation)

  Spec:        lib/blocks/neural_field_oscillator.neuro
                  (`formal_spec nfo_bipartition_coherence_lower_bounds_phi`)
               docs/NEURAL_FIELD_OSCILLATOR.md §3 (Coherence functional)
  Code refs:   neuroslm/modules/neural_field_oscillator.py
                  (`_bipartition_coherence`)
               neuroslm/emergent/nfo_coherence.py
                  (`NFOCoherenceProbe.step`, `nfo_phi_kappa` column)
  Tests:       tests/modules/test_nfo.py
                  ::test_bipartition_coherence_monotone_in_couplings

  Proof strategy:
    The Φ proxy in neuroslm/thsd/phi.py is `rank(L) / d_F`, monotone
    in the coupling count by H001. The NFO bipartition coherence
    functional `Φ_κ(R) = mean(1 − R)` is a real-valued image of the
    integer "unsynchronised cut edges" count — and adding a coupling
    is exactly the operation that synchronises one cut edge (so the
    functional decreases). The integer-valued lower bound

        Φ(s ⊕^n α) ≥ Φ(s) + n

    is the cut-count statement; it is provable in mathlib-free Lean
    via Sheaf.couplingCount_addList and List.length_replicate. The
    real-valued statement follows by Lipschitz extension from the
    integer chain.

  Postulates used: NONE.
-/
import Brian.Core

open Brian.Thsd Brian.Nfo

namespace Brian

/-- H015: NFO bipartition coherence is a closed-form Φ lower bound. -/
theorem NfoBipartitionCoherenceLowerBound :
    ∀ (s : Sheaf) (n : BipartitionEdgeSet),
      Phi s + n ≤ Phi (couplings_of_cut n |>.foldl Sheaf.addCoupling s) :=
  Brian.Nfo.bipartition_coherence_phi_lower_bound

/-- The `n = 0` boundary: empty bipartition cannot decrease Φ. -/
theorem NfoBipartitionCoherenceLowerBound_zero :
    ∀ (s : Sheaf),
      Phi s ≤ Phi (couplings_of_cut 0 |>.foldl Sheaf.addCoupling s) :=
  Brian.Nfo.bipartition_coherence_phi_lower_bound_zero

end Brian
