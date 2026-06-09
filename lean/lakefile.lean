import Lake
open Lake DSL

/-
  Brian — formal proofs for the THSD hypothesis ledger.
  See docs/formal_framework.md §10 for the theorem catalogue and
  CLAUDE.md §12 for the "no sorry" enforcement rule.

  Layout:
    lean/Brian/                       — the Brian core library
    lean/Brian/Thsd/                  — Topological Hyper-Sheaf-Dynamics
    lean/Brian/Verification/          — Triple-Guard, gates
    lean/Brian/Statistics/            — Welch, ImprovementGate
    lean/Brian/Postulate.lean         — admitted empirical claims
    lean/test/                        — TDD-style smoke tests for the
                                        Brian library lemmas

  The hypothesis proofs at hypothesis/proofs/H###_*.lean live OUTSIDE
  this directory by design (they belong with the hypothesis ledger,
  not the proof library). They are checked one-at-a-time by
  `neuroslm.discoveries.lean.verify_lean_proof`, which sets LEAN_PATH
  so that `import Brian.Core` resolves to this library.

  Mathlib is intentionally NOT a hard dependency. The Brian core
  library is mathlib-free so `lake build` stays fast on a fresh
  checkout; hypothesis proofs that need Mathlib's tactics can add
  the `require mathlib` line as they land.
-/

package «brian» where
  -- empty; defaults are fine

@[default_target]
lean_lib «Brian» where
  -- Picks up everything under lean/Brian/**/*.lean

lean_lib «BrianTest» where
  srcDir := "test"
  -- Picks up everything under lean/test/**/*.lean; built explicitly
  -- by `lake build BrianTest`, not part of the default target.
