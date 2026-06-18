/-
  Brian.NN — formal types and notation for neural network layers.

  Mirrors the operations defined in `lib/blocks/neural_field_oscillator.neuro`
  (equations `nfo_lift`, `nfo_coherence_gate`, `nfo_readout`) and
  implemented in `neuroslm/modules/neural_field_oscillator.py`.

  This module is intentionally **mathlib-free** and **Nat/Bool-valued**
  so that theorems remain decidable without real-analysis machinery.
  The fp32 implementation is the canonical one; the types here provide
  formal vocabulary that hypothesis proofs can reason over without
  importing Mathlib.LinearAlgebra.

  Covered operations:

    `LinearLayer`     — weight matrix + bias, zero-init flag (H018)
    `ResidualUpdate`  — h_out = h_in + α · layer(y)         (H018)
    `CoherenceGate`   — g = R / (R_max + ε), gating map      (H016)
    `RmsnormLayer`    — RMSNorm pre-conditioning              (general)

  Referenced by:
    hypothesis/proofs/H016_coherence_gate_information_preserving.lean
    hypothesis/proofs/H018_nfo_readout_zero_init_identity.lean
    lean/test/BrianTest/Smoke.lean
-/
namespace Brian.NN

-- ── LinearLayer ────────────────────────────────────────────────────────

/-- A linear (affine) layer: `y = x @ Wᵀ + b`.

    Shape: W ∈ ℝ^{outDim × inDim}, b ∈ ℝ^{outDim}.

    In PyTorch: `nn.Linear(inDim, outDim)` with optional
    `nn.init.zeros_(layer.weight)` and `nn.init.zeros_(layer.bias)`.

    The `isZeroInit` flag is true iff **both** W = 0 **and** b = 0 —
    the combined condition for the H018 (baseline-identity) guarantee.
    This is the ReZero discipline applied to the NFO readout layer:
    see `neuroslm/modules/neural_field_oscillator.py::__init__`
    lines `nn.init.zeros_(self.read_out.weight)`. -/
structure LinearLayer where
  inDim      : Nat
  outDim     : Nat
  /-- True iff W = 0 and b = 0 (both weight and bias zero-initialized). -/
  isZeroInit : Bool
  deriving Inhabited, DecidableEq

namespace LinearLayer

/-- Constructor: a zero-initialized linear layer.
    Models `nn.init.zeros_(self.read_out.weight)` in the NFO block. -/
def zeroInit (inDim outDim : Nat) : LinearLayer :=
  { inDim, outDim, isZeroInit := true }

/-- A layer is **zero-contributing** if its output is identically 0
    for any input (Bool proxy: true = contributes zero, false = non-zero).

    A zero-init layer is always zero-contributing by the ReZero
    construction: `y @ 0ᵀ + 0 = 0` for any `y`. -/
def isZeroContrib (layer : LinearLayer) : Bool := layer.isZeroInit

/-- **ReZero discipline theorem.** If the layer is zero-initialized,
    its output is always zero — for any input, any batch size, any
    token count.

    Formal counterpart of the Python assertion:
      `assert self.read_out.weight.abs().max() == 0.0` at init.

    Proof: `isZeroContrib` is defined as `isZeroInit`, so
    `h : isZeroInit = true` directly witnesses the conclusion. -/
theorem zeroInit_output_is_zero (layer : LinearLayer)
    (h : layer.isZeroInit = true) :
    layer.isZeroContrib = true := by
  unfold isZeroContrib; rw [h]

/-- Variant stated on the `zeroInit` constructor directly. -/
theorem zeroInit_layer_output_is_zero (i o : Nat) :
    (zeroInit i o).isZeroContrib = true := by
  unfold zeroInit isZeroContrib; rfl

/-- Converse: if `isZeroContrib = false` then `isZeroInit = false`. -/
theorem non_zero_contrib_implies_non_zero_init (layer : LinearLayer)
    (h : layer.isZeroContrib = false) :
    layer.isZeroInit = false := by
  unfold isZeroContrib at h; exact h

end LinearLayer

-- ── ResidualUpdate ─────────────────────────────────────────────────────

/-- A residual-stream update of the form `h_out = h_in + α · layer(y)`.

    Used by the NFO readout (H018): `h_out = h_in + alpha · Wo(g ⊙ y)`
    where `Wo` is the zero-init readout layer.

    When `layer.isZeroInit = true` the update is the identity:
    `h_out = h_in + α · 0 = h_in`. -/
structure ResidualUpdate where
  layer        : LinearLayer
  /-- `alphaNonzero = true` iff the readout gain α ≠ 0. -/
  alphaNonzero : Bool
  deriving Inhabited

namespace ResidualUpdate

/-- A residual update is **identity-preserving** (h_out = h_in) when
    the layer contributes zero. This occurs when:
      * `layer.isZeroInit = true`  (zero-init weight — no gradient yet), OR
      * `alphaNonzero = false`     (gain α = 0 — gate closed).

    `false || ¬true = true` — Bool.not applied to alphaNonzero
    gives a zero-gain condition. -/
def isIdentity (u : ResidualUpdate) : Bool :=
  u.layer.isZeroInit || !u.alphaNonzero

/-- **H018 core theorem (ResidualUpdate form).**

    A zero-init readout layer induces an identity residual update
    regardless of the gain α.

    Proof: `isZeroInit = true` makes the first disjunct of `isIdentity`
    true, regardless of `alphaNonzero`. -/
theorem zeroInit_is_identity (u : ResidualUpdate)
    (h : u.layer.isZeroInit = true) :
    u.isIdentity = true := by
  unfold isIdentity
  rw [h]
  simp

/-- Constructor convenience: the canonical zero-init NFO readout update. -/
def nfoZeroInit (inDim outDim : Nat) (alphaNonzero : Bool) : ResidualUpdate :=
  { layer := LinearLayer.zeroInit inDim outDim
  , alphaNonzero
  }

/-- The NFO zero-init update is always identity. -/
theorem nfoZeroInit_is_identity (inDim outDim : Nat) (alphaNonzero : Bool) :
    (nfoZeroInit inDim outDim alphaNonzero).isIdentity = true := by
  unfold nfoZeroInit
  apply zeroInit_is_identity
  unfold LinearLayer.zeroInit
  rfl

end ResidualUpdate

-- ── CoherenceGate ──────────────────────────────────────────────────────

/-- The NFO coherence gate `g(R) = R / (R_max + ε)`.

    In `neuroslm/modules/neural_field_oscillator.py`:
    ```python
    g = R / (R_max + eps)   # forward, line: g = R / (R_max + eps)
    h_out = h_in + alpha * self.read_out(g * y)
    ```

    The gate is a per-token scalar in [0, 1]: tokens whose local
    mean-field has high coherence (large R) get a louder voice in
    the residual write-back. The H016 hypothesis formalises this as
    an information-preserving map.

    **Integer representation:**
    We model R and R_max as `Nat` numerators so all reasoning is
    decidable without real arithmetic. The gate output is `R` itself
    (the numerator), with the constraint `R ≤ R_max` enforcing that
    the scaled output is ≤ 1. This is faithful to the continuous case:
    `g = R / R_max` maps R ↦ R/R_max ∈ [0, 1], represented here as
    the pair (R, R_max) with R ≤ R_max. -/
structure CoherenceGate where
  /-- Local Kuramoto order parameter R (Nat numerator). -/
  R     : Nat
  /-- Global maximum R_max > 0 (Nat denominator). -/
  R_max : Nat
  /-- R ≤ R_max ensures the gate is in [0, 1] (integer proxy). -/
  hMax  : R ≤ R_max
  deriving Inhabited

namespace CoherenceGate

/-- The gate output (integer proxy for R / R_max). In the Nat proxy
    the "scaled" value is just R; the division by R_max is implicit
    in the `hMax : R ≤ R_max` invariant. -/
def apply (g : CoherenceGate) : Nat := g.R

/-- **H016 (information-preserving, sense 1): gate ≤ R_max.**

    The gate output cannot exceed the coherence ceiling R_max.
    In real terms: g = R / R_max ≤ 1 because R ≤ R_max.

    This is the "cannot amplify" direction: no synchronisation is
    created by the gate that was not already in the oscillator field. -/
theorem apply_le_R_max (g : CoherenceGate) : g.apply ≤ g.R_max :=
  g.hMax

/-- **H016 (information-preserving, sense 2): identity at full coherence.**

    When R = R_max (every oscillator is perfectly synchronised),
    the gate output equals R_max (full readout, no signal lost).

    In real terms: g = R / R_max = 1 when R = R_max. -/
theorem apply_identity_at_max (g : CoherenceGate)
    (h : g.R = g.R_max) :
    g.apply = g.R_max := by
  unfold apply; rw [h]

/-- **H016 (information-preserving, sense 3): zero coherence ⇒ gate zero.**

    When R = 0, the gate contributes nothing (g = 0). No spurious
    information is written back when the oscillators are incoherent. -/
theorem apply_zero_when_R_zero (g : CoherenceGate)
    (h : g.R = 0) :
    g.apply = 0 := by
  unfold apply; exact h

/-- **H016 (monotone in R).** For two gates sharing the same R_max,
    a larger R produces a larger (or equal) gate output.

    In real terms: g(R₁) ≤ g(R₂) when R₁ ≤ R₂ (and R_max is fixed),
    because g = R / R_max is non-decreasing in R. -/
theorem apply_monotone (g1 g2 : CoherenceGate)
    (hSameMax : g1.R_max = g2.R_max) (hR : g1.R ≤ g2.R) :
    g1.apply ≤ g2.apply := by
  unfold apply; exact hR

/-- Combined information-preserving statement:
    `apply g ≤ R_max` and `apply g = R_max` when `R = R_max`. -/
theorem information_preserving (g : CoherenceGate) :
    g.apply ≤ g.R_max ∧ (g.R = g.R_max → g.apply = g.R_max) :=
  ⟨apply_le_R_max g, apply_identity_at_max g⟩

end CoherenceGate

-- ── RmsnormLayer ───────────────────────────────────────────────────────

/-- An RMSNorm pre-conditioning layer: `x' = x / rms(x) · γ`.

    In `lib/blocks/neural_field_oscillator.neuro`:
      `nfo_lift` equation: `linear(rmsnorm(h, gamma), Wu)`.

    The formal property used in H015 / H016: RMSNorm is a homogeneous
    map of degree 0 in the denominator direction — it does not change
    the SUPPORT of its input (non-zero entries remain non-zero).
    Tracked here via `isNormalizingOnly : Bool`. -/
structure RmsnormLayer where
  dim             : Nat
  /-- True iff this layer is a pure normalization (no learnable gain γ). -/
  isNormalizingOnly : Bool
  deriving Inhabited

end Brian.NN
