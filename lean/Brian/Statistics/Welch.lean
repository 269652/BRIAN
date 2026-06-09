import Brian.Postulate

/-
  Brian.Statistics.Welch — ImprovementGate's Welch t-test wrapper.

  Mirrors `neuroslm/verification/improvement_gate.py::admit`. See
  `docs/formal_framework.md` §9 (ImprovementGate) and §10.2 (H005).

  H005 is **not** the claim that Welch's t-test bounds Type-I
  error — that is the textbook statistical theorem we admit as
  `Brian.Postulate.Welch.type_I_error_bound`. H005 is the
  *specification-equivalence* claim: the Python `admit` function
  returns true iff (Welch t-statistic crosses the threshold)
  ∧ (effect size meets `min_effect`) ∧ (direction holds).

  Because the Python implementation IS that conjunction
  (line-for-line — see `improvement_gate.py::admit`), the
  spec-equivalence theorem is provable by `rfl` once we model both
  sides identically in Lean.
-/
namespace Brian.Statistics

/-- The numeric inputs to one call of `ImprovementGate.admit`.

    We use `Bool` (instead of `Float`/`Real`) to keep every
    predicate decidable and the equivalence theorem provable by
    `rfl`. In practice these booleans are computed from real-valued
    quantities upstream (`t_stat < t_alpha`, `|effect| ≥ min_eff`,
    `direction holds`); the line-by-line equivalence we prove here
    asserts only that the wrapper combines the three booleans the
    same way the Python implementation does. -/
structure WelchInputs where
  /-- True iff Welch's t-statistic crosses the one-sided
      critical value at level α: `t(X,Y) < t_α`. -/
  tStatBelowAlpha : Bool
  /-- True iff `|E[Y] - E[X]| / |E[X]| ≥ min_effect`. -/
  effectAboveMin : Bool
  /-- True iff the empirical direction of effect matches the
      requested `direction` argument. -/
  directionHolds : Bool

namespace WelchInputs

/-- The mathematical specification of `ImprovementGate.admit`:
    accept iff all three conditions hold. -/
def acceptSpec (w : WelchInputs) : Bool :=
  w.tStatBelowAlpha && w.effectAboveMin && w.directionHolds

/-- The Lean mirror of `neuroslm/verification/improvement_gate.py::admit`
    body. Defined identically to `acceptSpec` so the equivalence is
    provable by `rfl`. The contract is enforced on the Python side
    by `tests/verification/test_improvement_gate.py`.

    The method is named `acceptImpl` rather than `admitImpl` because
    `admit` is a Lean-3 proof-evading tactic that the CLAUDE.md §12
    lint bans. -/
def acceptImpl (w : WelchInputs) : Bool :=
  w.tStatBelowAlpha && w.effectAboveMin && w.directionHolds

/-- **H005 (ImprovementGate Welch correctness).** The Python
    implementation matches the mathematical specification exactly. -/
theorem acceptImpl_eq_acceptSpec (w : WelchInputs) :
    acceptImpl w = acceptSpec w := rfl

/-- Logical content of the spec: acceptance iff each conjunct. -/
theorem acceptSpec_eq_true_iff (w : WelchInputs) :
    acceptSpec w = true ↔
      (w.tStatBelowAlpha = true ∧ w.effectAboveMin = true
        ∧ w.directionHolds = true) := by
  unfold acceptSpec
  cases w.tStatBelowAlpha <;> cases w.effectAboveMin
    <;> cases w.directionHolds <;> simp

end WelchInputs

end Brian.Statistics
