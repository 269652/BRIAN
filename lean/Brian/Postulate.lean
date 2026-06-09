/-
  Brian.Postulate — the namespace for empirical claims we admit.

  Per CLAUDE.md §12.2, every member of `Brian.Postulate.*`:
    * must have a precise type signature in THSD vocabulary,
    * must be referenced by exactly one hypothesis proof,
    * carries a doc-comment naming the empirical evidence
      (test file, paper, or docs/formal_framework.md §).

  Audit commands (rule 12 enforcement):

      grep -rn 'namespace Brian.Postulate' lean/   # every postulate file
      grep -rn '^axiom '                  lean/Brian/Postulate.lean
      grep -rn '^axiom '                  lean/Brian/Postulate/

  A `Brian.Postulate.X` is a *named admission of incompleteness*.
  It is NOT a `sorry` in disguise — it must have a precise type
  using THSD vocabulary so any audit pass can list, by name, every
  unproven link in the formal chain.
-/
namespace Brian.Postulate

/-! This file is intentionally a namespace stub. Concrete postulates
    live in `Brian/Postulate/<topic>.lean` so each topic's empirical
    evidence is co-located with the postulate that depends on it.

    See:
      * `Brian/Postulate/Cdga.lean`     — H002 OOD-gap contraction
      * `Brian/Postulate/Symbolic.lean` — H003 Gumbel-Softmax collapse
      * `Brian/Postulate/Welch.lean`    — H005 Welch Type-I bound
-/

/-! ## Unimplemented marker

    `Brian.Postulate.Unimplemented` is the autogen scaffold's
    obligation type. It is defined as `Unit` so the scaffold compiles
    cleanly in Lean (no `sorry`), but the Python-side static lint in
    `neuroslm.discoveries.lean` scans every committed `.lean` file
    for the literal `Brian.Postulate.Unimplemented` token and fails
    the verification verdict if it appears.

    The contract:

      * Autogen emits  `theorem H001 : Unimplemented "H001" := unimplemented _`.
      * Lean accepts the file (no errors).
      * Static lint catches the marker → verdict status = `"stub"`.
      * Hand-edit replaces the obligation with a real theorem in
        THSD vocabulary → marker disappears → lint passes →
        once Lean is on PATH, the kernel verifies → `"verified"`.

    Per CLAUDE.md §12.2 this marker is the *only* sanctioned way to
    have an unfinished proof on disk. `sorry` and `: True := by trivial`
    remain banned. -/

/-- Type of the autogen scaffold's obligation. Defined as `Unit` so
    the scaffold compiles without `sorry`; flagged by the Python
    static lint via grep for `Brian.Postulate.Unimplemented`. -/
def Unimplemented (_tag : String) : Type := Unit

/-- The canonical inhabitant of `Unimplemented`. -/
def unimplemented (_tag : String) : Unimplemented _tag := ()

end Brian.Postulate
