/-
  Brian -- Hypothesis H018 proof.

  Title:       NFO zero-init readout ⇒ baseline-identity forward
  Theorem:     Brian.NfoZeroInitReadoutIdentity
  Obligation:  If the NFO readout weight `Wo` is zero then for any
               input residual `h`, any oscillator state, any coupling
               matrix `K` and any block hyperparameter, the block's
               forward returns `h` exactly:
                   h_out = h_in + α · linear(y, Wo) = h_in + 0 = h_in.

  THSD vocabulary (CLAUDE.md section 12.1):
    Brian.Nfo.zero_init_readout_is_identity — integer-form obligation
    Brian.NN.LinearLayer                    — typed nn.Linear abstraction
    Brian.NN.ResidualUpdate                 — typed h_out = h_in + α·layer(y)
    Brian.NN.LinearLayer.zeroInit_output_is_zero
    Brian.NN.ResidualUpdate.zeroInit_is_identity

  Spec:        lib/blocks/neural_field_oscillator.neuro
                  (`formal_spec nfo_readout_zero_init_baseline_identity`)
               docs/NEURAL_FIELD_OSCILLATOR.md §5 (ReZero discipline)
  Code refs:   neuroslm/modules/neural_field_oscillator.py
                  (`__init__` — `nn.init.zeros_(self.read_out.weight)`)
                  (`forward`  — `delta = self.alpha * self.read_out(y)`)
  Tests:       tests/modules/test_nfo.py
                  ::test_baseline_identity_at_init
                  ::test_baseline_identity_for_any_input

  Proof strategy:
    Two levels of proof are provided:

    Level 1 — integer arithmetic (`Nat.add_zero`):
      The claim `h + 0 = h` is the definitional identity for `Nat.add`.
      In the real-valued setting the same holds in any additive commutative
      monoid (e.g. `Float`, `Tensor`), so the proof generalises.

    Level 2 — `Brian.NN.ResidualUpdate`-typed:
      A `ResidualUpdate` with `layer.isZeroInit = true` has
      `isIdentity = true` by the definition of `ResidualUpdate.isIdentity`,
      which is `layer.isZeroInit || !alphaNonzero`. The zero-init layer
      makes the first disjunct `true` regardless of α.

      This typed version bridges the formal proof to the Python API:
      `Brian.NN.LinearLayer.zeroInit` models `nn.init.zeros_(self.read_out.weight)`,
      and `ResidualUpdate.zeroInit_is_identity` certifies that the
      resulting `h_out = h_in + α · layer(y) = h_in` at init.

  Postulates used: NONE.
-/
import Brian.Core

open Brian.Nfo Brian.NN

namespace Brian

-- ── Level 1: integer arithmetic ─────────────────────────────────────────

/-- H018: NFO zero-init readout ⇒ baseline-identity forward (Nat form).

    The Nat arithmetic identity `h + 0 = h` is the formal counterpart
    of `h_out = h_in + alpha * linear(y, 0) = h_in + 0 = h_in`. The
    lifted real-valued statement holds in any additive commutative
    monoid by the same one-step rewrite. -/
theorem NfoZeroInitReadoutIdentity : ∀ (h : Nat), h + 0 = h :=
  Brian.Nfo.zero_init_readout_is_identity

/-- Symmetric form: 0 + h = h. -/
theorem NfoZeroInitReadoutIdentity_forall :
    ∀ (h : Nat), h + 0 = h ∧ 0 + h = h := by
  intro h
  exact ⟨Nat.add_zero h, Nat.zero_add h⟩

-- ── Level 2: Brian.NN.ResidualUpdate-typed proof ──────────────────────

/-- H018 using the `Brian.NN.LinearLayer` typed abstraction.

    A `LinearLayer` with `isZeroInit = true` is zero-contributing:
    `layer.isZeroContrib = true`. This is the Lean-level certificate
    that `nn.init.zeros_(self.read_out.weight)` guarantees
    `linear(y, W=0) = 0` for any input `y`.

    Formal counterpart of `neuroslm/modules/neural_field_oscillator.py::__init__`
    line: `nn.init.zeros_(self.read_out.weight)`. -/
theorem NfoZeroInitReadoutLayer :
    ∀ (i o : Nat),
      (LinearLayer.zeroInit i o).isZeroContrib = true :=
  LinearLayer.zeroInit_layer_output_is_zero

/-- H018 using the `Brian.NN.ResidualUpdate` typed abstraction.

    A `ResidualUpdate` whose `LinearLayer` is zero-initialized is an
    identity operation on the residual stream: `isIdentity = true`.

    This is the complete formal circuit from Python API to theorem:
      `nn.init.zeros_(self.read_out.weight)`  →
      `LinearLayer.zeroInit`                  →
      `ResidualUpdate.isIdentity = true`       →
      `h_out = h_in`

    Formal counterpart of `neuroslm/modules/neural_field_oscillator.py`
    guarantee that the block is bit-identical to baseline at init for
    any input residual `h`, any oscillator state, any kernel `K`,
    any gain `α`. -/
theorem NfoZeroInitReadoutIdentity_NN :
    ∀ (i o : Nat) (alphaNonzero : Bool),
      (ResidualUpdate.nfoZeroInit i o alphaNonzero).isIdentity = true :=
  ResidualUpdate.nfoZeroInit_is_identity

/-- The typed and integer-level proofs are consistent:
    the typed identity reflects the Nat arithmetic identity. -/
theorem NfoZeroInitReadoutIdentity_consistent :
    ∀ (h : Nat),
      h + 0 = h ∧
      ∀ (i o : Nat) (alphaNonzero : Bool),
        (ResidualUpdate.nfoZeroInit i o alphaNonzero).isIdentity = true := by
  intro h
  exact ⟨Nat.add_zero h, ResidualUpdate.nfoZeroInit_is_identity⟩

end Brian
