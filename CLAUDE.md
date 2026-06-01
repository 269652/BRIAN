# Repo-wide instructions for Claude / agents working in this repo

These rules apply to **every** Claude/agent session on this codebase.
They are non-optional.

---

## 1. TDD first — write tests before implementation

Any non-trivial code change (a new function, a new module, a bug fix
with a clear repro) starts with a failing test:

1. Find the right test file under `tests/`. If none fits, create one.
2. Write the test that captures the desired behaviour. **Run it first**
   and confirm it fails for the right reason.
3. Implement the smallest change that makes it pass.
4. Re-run the test (and the rest of the file's suite) to confirm green.
5. Commit the test + implementation together.

Exemptions (no test required, but still confirm intent in conversation):
- Pure renames, comment-only edits, formatting passes.
- One-shot ops scripts that delete themselves (and are never imported).
- Doc files (`.md`) — but see rule 3 below before adding any.
- vast.ai deploy scripts (verified by deploying, not unit-tested).

If the test is hard to write, ask before skipping it. "It's hard to
test" is the most common smell that the design needs rethinking.

---

## 2. No sycophancy

- Don't preface answers with "Great question!", "You're absolutely
  right!", or "Excellent point!".
- Don't repeat the user's instruction back to them before doing it.
- Don't ask "Would you like me to..." for tasks the user clearly
  already asked for.
- If the user is wrong about something verifiable, say so plainly with
  the evidence.

State results and decisions directly. Brief is good; performative is
bad.

---

## 3. No unneeded `.md` files

Every markdown file in this repo has to justify existing. Before
creating a new one, ask:

- **Is there an existing file this content belongs in?** Append, don't
  fork. (E.g. session summaries belong in `docs/history.md`, never as
  `FINAL_SUMMARY.md` at the root.)
- **Will someone reread this in a week?** If no, it's a chat artifact.
  Write it in the conversation, not as a file.
- **Is this a "completion stamp" or "investigation log"?** Those rot
  fast. They go in `docs/archive/` with a dated name (`YYYY-MM-DD_*.md`)
  the moment they stop being load-bearing, or get deleted entirely.

Allowed top-level `.md`:
- `README.md`
- `CLAUDE.md` (this file)
- Anything else needs explicit user approval.

Allowed `docs/` `.md`:
- `architecture.md` — primary spec (detailed §0–12 for codebase maintainers)
- `technical_report.md` — executive summary (for external AIs + new contributors)
- `findings.md` — running hypothesis ledger (Layer A + Layer B evidence)
- `history.md` — session notes + decisions (auto-maintained)
- `changelog.md` — git-derived (auto-maintained)
- `metrics.md` — auto-updated by `brian analyze-log`
- `dsl.md`, `dsl_nn_language.md`, `dsl_subsystem_roadmap.md` — DSL docs
- `OOD_PUSH_STAGES.md` — deprecated; migrate to archive then delete.

`docs/archive/` is the graveyard. Move-then-cite, never delete-then-forget.

---

## 4. File creation has to serve a purpose

Beyond `.md`, this also covers:
- One-off Python scripts at the repo root: only if they're a deploy
  helper (`_deploy_*.py`) or a debugging shim that gets deleted in
  the same PR. **Never** leave `_test.py`, `_check.py`, `_dbg.py`
  artifacts around — those are scratch.
- Generated artifacts (`.png`, `.html`, large `.json`, large
  notebooks): only if they're explicitly user-requested OR they're
  the canonical output of a tool that the user runs (`compile nfg`,
  `analyze-log`).
- Don't add directories you don't immediately populate with code +
  tests.

The default action when in doubt is **not to create the file**. Use
the conversation.

---

## 5. Cleanup is part of every task

Before declaring a task complete:

- `git status` — anything untracked you created? Either commit it or
  delete it. No "Untracked Files" creep.
- Old session artifacts (`*.tmp`, `_summary*.md`, `scratch.py`)
  belong in `docs/archive/` or `/dev/null`.
- The `brian ai cleanup` skill runs this audit on demand; respect its
  recommendations.

---

## 6. When you must produce a report file

If the user explicitly asks for a long-form document:

- Use the existing file (`docs/history.md`, `findings.md`,
  `architecture.md`) and append a dated section.
- Don't open a new top-level `*_SUMMARY.md`.
- Cite sources by file path + line number.

---

## 7. Code style + conventions

- Match the surrounding file (this isn't a place to push personal
  preferences).
- Default to no comments. Add a comment only when the *why* is
  non-obvious — a hidden constraint, a subtle invariant, a workaround.
- Never write multi-paragraph docstrings unless the function is part
  of a documented API surface.
- Avoid backwards-compatibility shims for code that nothing in the
  repo uses anymore. Delete cleanly.

---

## 8. Operational discipline

- **Don't push without explicit ask.** Commit locally, then surface
  the diff for review.
- **Don't destroy vast.ai instances without explicit ask** — running
  jobs cost money to restart, not to keep going.
- **Don't reset / force-push / amend pushed commits** unless the user
  explicitly says so.

Match the scope of your actions to what was requested. If the user
asks to "fix the bug," they didn't ask for a refactor — they didn't
ask for renaming, they didn't ask for a new abstraction. Stay tight.

---

## 9. Documentation Synchronization

Every architectural change **must** synchronize documentation:

1. **Architecture change → arch.neuro first.** The `.neuro` file is the
   canonical source of truth. Update it before implementation.
2. **Significant spec changes → architecture.md.** If you change the
   design (add a subsystem, alter a mechanism, change equations), update
   the corresponding section in `docs/architecture.md`.
3. **New evidence (test or result) → findings.md.** Every Layer A test
   and Layer B OOD eval result gets a row in the hypothesis ledger with
   an explicit link to the artifact (test name or JSON path).
4. **New public-facing documentation → technical_report.md.** This is the
   report external AIs see. Keep it synchronized with findings.md via:
   ```bash
   python scripts/maintain_technical_report.py --verbose
   ```
   This checks that all evidence links exist and detects drift between
   arch.neuro and the report. Fix any issues before committing.
5. **Commit all synced docs together.** If you change arch.neuro,
   architecture.md, findings.md, or technical_report.md, commit them in
   the same change set. One logical change = one commit with all docs.

Example commit (good):
```
arch: add ReZero zero-init gates (§5.3 fix)

- Add λ_motor, λ_mem, λ_thought scalars to forward injection paths
- Update arch.neuro lines 50–60 (ReZero gate config)
- Update architecture.md §5.3 (new mechanism spec)
- Update findings.md H8 (evidence link to test + OOD result)
- Update technical_report.md §7.2 (current state)
- Audit passed: no drift detected

[EVIDENCE: tests/test_stabilization.py::test_rezero_zero_init]
[EVIDENCE: results/ood_rezero-fixed_107M_step7000.json]
```

**Archive policy.** Old session notes, investigation logs, and abandoned
experiment files (OOD_PUSH_STAGES.md, *_SUMMARY.md) should be moved to
`docs/archive/YYYY-MM-DD_*.md` when they stop being load-bearing. Use:
```bash
python scripts/maintain_technical_report.py --fix
```
to auto-archive known stale files. Then commit the archive move.
