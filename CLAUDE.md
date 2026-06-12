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

**Every file in this repo must earn its place.** Every markdown file has
to justify existing by meeting a durable need. Treat file creation as a
commitment: the file will be maintained, kept in sync, and read
repeatedly. Session artifacts, one-off notes, and temporary summaries do
not belong in the repo.

Before creating any `.md`, ask yourself:

- **Is there an existing file this content belongs in?** Append, don't
  fork. (E.g. session summaries belong in `docs/history.md`, never as
  `FINAL_SUMMARY.md` at the root.)
- **Will someone (including you in 3 months) reread this?** If no, it's a
  chat artifact — write it in the conversation, not as a persistent file.
- **Is this a "completion stamp", "status update", or "investigation log"?**
  Those rot fast. If they must be archived: move to `docs/archive/` with
  a dated name (`YYYY-MM-DD_*.md`), or delete. Do not leave them in
  active docs.
- **Am I creating this file because it's genuinely needed, or just because
  I feel like documenting something?** Default: don't. Use the conversation.

Allowed top-level `.md`:
- `README.md` — project overview
- `CLAUDE.md` — this file (repo-wide rules)
- Anything else needs **explicit user approval** before the first commit.

Allowed `docs/` `.md`:
- `architecture.md` — primary spec (detailed §0–12 for codebase maintainers)
- `technical_report.md` — executive summary (for external AIs + new contributors)
- `findings.md` — running hypothesis ledger (Layer A + Layer B evidence)
- `history.md` — session notes + decisions (auto-maintained)
- `changelog.md` — git-derived (auto-maintained)
- `metrics.md` — auto-updated by `brian analyze-log`
- `dsl.md`, `dsl_nn_language.md`, `dsl_subsystem_roadmap.md` — DSL docs
- `BRIAN.md`, `CLI.md`, `harness.md` — reference documentation
- Anything else needs **explicit user approval**.

`docs/archive/` is the graveyard for dead session notes and investigation
logs. Move stale files there with a `YYYY-MM-DD_` prefix, never delete
silently. Citation trail must be preserved.

---

## 4. Any file added to the repo must have a durable purpose

This is the universal rule. The repo is not a scratchpad. Every file is a
commitment to maintenance, clarity, and justification. Before you create
a file, know why it belongs in the repo and why it will still belong in
6 months.

**Test files & scripts:**
- `.py` test files: belong in `tests/` with a real test suite. Standalone
  scripts like `_test.py`, `_check.py`, `_dbg.py`, or `scratch.py` are
  never committed — those are local experiments.
- One-off Python scripts at the repo root: only if they're labeled
  `_deploy_*.py` (a deploy helper) AND will be used repeatedly. Debugging
  shims and one-shot verification scripts go in the conversation, not the
  repo.
- Test output files: never commit `.log`, `_output.txt`, or `_summary.md`
  files created during testing. Those are transient.

**Generated artifacts:**
- Images, HTML, JSON, notebooks: only if (1) explicitly user-requested,
  OR (2) they're the canonical output of a tool the user runs regularly
  (`compile nfg`, `analyze-log`, `brian deploy --label`). Intermediate
  outputs, screenshots, and demo results do not belong.

**Directories:**
- Don't create a directory unless you populate it with code + tests in the
  same commit. No empty directories, no "reserved for future use."

**Golden rule:**
The default action when in doubt is **not to create the file**. Ask
yourself: "Will a future developer reading this repo in 6 months
understand why this file exists?" If the answer is "probably not," use
the conversation instead.

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

### 9.1 DSL/arch ↔ technical_report.md cross-alignment (anti-drift)

**Trigger:** any change to one of these surfaces must be cross-aligned
with `docs/technical_report.md` **in the same commit**:

- `architectures/*/arch.neuro` (and any `architectures/*/modules/*.neuro`)
- `neuroslm/dsl/*.py` — DSL grammar, parser, training_config, regularization
- `neuroslm/compiler/*.py` — Ribosome / NFG / Hypergraph IR
- `dna/*/*.dna` (compiled snapshots — usually downstream of an arch edit)
- Anything that changes the **schema** of what the DSL accepts, what the
  Hypergraph IR exposes, or what `dna/evol/arch.dna` carries

**What "cross-aligned" means:** every commit touching the surfaces above
must do **one** of the following:

1. **Update `docs/technical_report.md`** so the document still describes
   what the code actually does. The §s most likely to drift:
   - §3 (architecture overview — match arch.neuro topology)
   - §5 (mechanisms — match `multi_cortex`, `regularization`, `hardware`)
   - §6 (DSL surface — match grammar/parser changes)
   - §7.2 (current state — match the active arch + scale + preset)
2. **OR** add a single-line `[NO SPEC IMPACT]` annotation in the commit
   body explaining why no doc update is needed (typo fix, internal
   rename, perf-only refactor with identical observable behaviour).
   Pure perf wins like adding a cache or memoisation typically qualify;
   anything that changes loss, throughput envelopes, or what gets
   logged does not.

**Pre-commit check** (run before you `git add`):

```bash
python scripts/maintain_technical_report.py --verbose
```

The script greps `arch.neuro` for every section/field name that
`technical_report.md` mentions and reports drifted/missing references.
Non-zero exit = doc drift = do not commit. If the script flags a false
positive, fix the script — never silence the warning.

**Why this rule exists.** Earlier sessions added the `experts: [...]`
roster, the `hardware{}` block, the `cheap_2k`/`t4_2k` scales, and the
`brian.toml [defaults]` precedence — all real architectural surface —
without touching `technical_report.md`. External AIs reading the report
then made plans against a 6-month-stale picture of the system. The
report is the contract; if it's wrong, every downstream reasoning step
is wrong.

**Worked example (good commit):**
```
feat(arch): add MoE LM-expert ensemble with vocab bridge

- arch.neuro: replace legacy `weights: "gpt2"` with `experts: [...]`
  roster of 3 frozen LMs + `trunk_tokenizer: "gpt2"`
- dna/evol/arch.dna: recompile (downstream of arch.neuro)
- training_config.py: parse `ExpertSpec`, derive n_cortices from roster
- experts.py: new LMExpertEnsemble routes pretrained heads to trunk vocab
- technical_report.md §5.7: rewrite "Multi-Cortex Ensemble" to describe
  pretrained-head routing (was: random-projection chain)
- maintain_technical_report.py --verbose: clean (no drift)

[EVIDENCE: tests/training/test_lm_expert_harness_integration.py::TestSmokingGunCE]
```

**Worked example (good commit, no-impact):**
```
perf(experts): cache pretrained LMs and tokenizers process-wide

- experts.py: add _load_lm_cached / _load_tokenizer_cached
- 7 LM-expert tests dropped from ~7s each to ~1s each
- No observable behaviour change (frozen experts are stateless during forward)

[NO SPEC IMPACT] perf-only; loss curves and logged metrics unchanged
```

### 9.2 Example commit shapes

Example commit (good — architectural change with full doc sync):
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

---

## 10. arch.neuro change → deploy → observe → record (the scientific loop)

Edits to any `architectures/*/arch.neuro` (or to a mechanism that the DSL
exposes — `neuroslm/dsl/*.py`, `neuroslm/harness.py` training-loop knobs)
are **research experiments**, not refactors. The project's evolution only
has scientific value if every such change is paired with a measurement
and a recorded observation. Follow this loop without exception:

1. **Form the hypothesis explicitly.** Before editing the `.neuro` file,
   write one sentence stating what metric (`train_ppl`, `OOD_ppl`,
   `gap_ratio`, `tok/s`, Φ, …) you expect to move, in which direction,
   and by how much. Note the prior baseline number from `findings.md`.

2. **Commit the `.neuro` change + any wiring together** with a message of
   the form `arch: <mechanism>=<value> (Hxx hypothesis)`. The catalog in
   `docs/OOD_MECHANISMS.md` must list any newly exposed DSL field.

3. **Deploy** with a label that encodes the change:
   `brian deploy --label neuroslm-<change>-v<n>`. Capture the returned
   vast.ai instance id — it is the artifact id for this experiment.

4. **Observe.** Use `brian ps` / `brian logs <id>` to read out the PPL /
   OOD-PPL / gap_ratio trajectory at canonical checkpoints (step 500,
   1000, 2000, 5000, full). Don't trust eyeballs — copy the numbers.

5. **Record in `docs/findings.md`** as a new `Hxx` section *before*
   moving on to the next change. Required fields: hypothesis, spec
   (commit + arch.neuro lines), run (vast id + label + GPU + cost),
   trajectory table, outcome (✅/🟡/🟠/❌), follow-up. Cite the run's
   instance id so the raw log under `logs/vast/` can be retrieved.

6. **Destroy** the instance once the verdict is recorded (cost
   discipline), unless the user explicitly asks to keep it running.

7. **Negative results count.** A FALSIFIED hypothesis is just as
   valuable as a confirmed one and *must* be recorded. Deleting or
   silently overwriting a failed-experiment finding destroys the
   evolutionary signal this project depends on.

Anti-patterns to refuse:
- Editing `arch.neuro`, deploying, and then editing again before the
  first run produces a measurement — the two changes become
  inseparable. Wait for the trajectory, record it, *then* iterate.
- Bundling more than one mechanism change in one experiment without
  noting it explicitly as a "stack" finding (see H13/H14 for the
  pattern). A stack finding *must* spawn a follow-up "single-mechanism
  ablation" backlog entry.
- Recording a finding without the instance id — the raw log is the
  only thing that lets a future reader audit the claim.

---

## 11. No repository clutter — every file earns its place

This is the meta-principle underlying rules 3–4. The repo must remain
clean, navigable, and free of dead weight. Clutter compounds: each
abandoned `.md`, each forgotten test artifact, each "temporary" script
makes the repo harder to reason about and slower for the next person
(often you, 3 months later).

**Apply ruthlessly:**
- Does this file have a clear, permanent purpose? If not, don't commit it.
- Is it tracking a temporary state (session notes, investigation logs, in-progress
  summaries)? Use `docs/archive/` with a dated filename, or delete it entirely.
- Is it code that nobody will reuse (one-shot debug script, test output,
  demo result)? Don't commit it. Use the conversation.
- Are there multiple related files that should be consolidated? Do that
  instead of adding more.

**At task completion:**
- `git status` — anything untracked? Delete it or commit it. No stragglers.
- Any files you added: Can you articulate in one sentence why each one
  belongs in the repo? If you hesitate, delete it.

**The cost of clutter:**
- Future maintainers waste time understanding what each file is for.
- Stale docs become liabilities (wrong info is worse than no info).
- The repo feels poorly maintained, which erodes confidence in the project.

**When in doubt: ask the user.** If you're unsure whether a file belongs,
mention it in conversation before committing. Prefer temporary artifacts
(conversation, branches, local experiments) over repo clutter.

---

## 12. Lean proofs — THSD-annotated, zero `sorry`s, load-bearing

Every `.lean` file in this repo is a **load-bearing proof obligation**, not a
placeholder. Lean is what turns our claims from "passes tests in this
configuration" into "is true under the THSD axioms." The autogen scaffold
emitted by `neuroslm/discoveries/lean.py` is exactly that — a scaffold; it is
**never the committed state**. The committed state is a complete proof.

### 12.1 The THSD Lean library is the only vocabulary allowed

`docs/formal_framework.md` defines the Topological Hyper-Sheaf-Dynamics
annotation framework. Its executable counterpart is `neuroslm/thsd/engine.py`
(`SimplexComplex`, `CellularSheaf`, `CoboundaryOperator`, `PhiDynamicsComputer`,
`SymbolicSimplex`). Its formal counterpart **must** be a Lean library at
`lean/Brian/` (mirror layout: `Brian/Thsd/Simplex.lean`, `Brian/Thsd/Sheaf.lean`,
`Brian/Thsd/Coboundary.lean`, `Brian/Thsd/Phi.lean`, `Brian/Thsd/Symbolic.lean`,
plus `Brian/Core.lean` as the top-level import surface).

Every hypothesis proof under `hypothesis/proofs/H###_*.lean`:

- **must** `import Brian.Core` (or a narrower `Brian.Thsd.*` submodule);
- **must** state its theorem using THSD types (`Sheaf`, `Coboundary`, `Φ`,
  `SymbolicSimplex`) — never ad-hoc redeclarations of the same objects;
- **must** cite, in the file's header docstring, the corresponding
  `docs/formal_framework.md` section number AND the
  `neuroslm/thsd/engine.py` symbol whose behaviour the theorem pins down;
- **must** type-check against the same Mathlib + `Brian.Core` pin recorded in
  `lean/lean-toolchain` and `lean/lakefile.lean`.

### 12.2 `sorry` is banned. So is `: True`.

No committed `.lean` file may contain the token `sorry`, nor may any theorem
have the trivial obligation `theorem X ... : True := by trivial`. The CI gate
(`brian hypothesis verify --all` once the Lean toolchain is provisioned)
fails the build on either pattern. The autogen stub in
`neuroslm/discoveries/lean.py` currently produces both — **that file is a
template, not a destination**. Replacing the `sorry` with the real proof,
and the `: True` with the real obligation in THSD types, is part of
delivering the hypothesis, not a follow-up.

Allowed escape hatches (rare, must be justified):

- `axiom`  — only for genuinely axiomatic THSD facts that the framework
  posits (e.g. "$\partial \circ \partial = 0$" if we choose to take it as
  axiom rather than derive it from `Brian.Thsd.Simplex`). Every `axiom`
  declaration in `lean/Brian/` requires a one-line comment naming the
  source (paper, formal_framework.md §, or "design choice") and a
  matching entry in `docs/formal_framework.md` §10.2's axiom table.
- `Brian.Postulate` namespace — wrap empirical conjectures we have not
  yet derived. Must be referenced by exactly one hypothesis and tagged
  `@[brian_postulate]` so a future audit pass can list every unproven
  link. A `Brian.Postulate` is a *named admission of incompleteness*; it
  is not a `sorry` in disguise — it must have a precise type signature
  in THSD vocabulary.

Nothing else. No `admit`, no `sorry`, no `_ : True`, no `Classical.choice`
inside a proof body without a comment naming the lemma it discharges.

### 12.3 What "complete description" means per `.lean` file

A `.lean` file is complete when, reading top-to-bottom, a stranger fluent in
Lean + the THSD framework can:

1. **Identify the obligation** — the file header docstring states the claim
   in prose, cites `formal_framework.md §X`, and names the
   `neuroslm/thsd/engine.py` symbol.
2. **Read the THSD context** — every imported name resolves to either
   Mathlib or `Brian.*`; no opaque external constants.
3. **Verify each step** — every `theorem`, `lemma`, `def`, and `axiom`
   has either a complete proof body OR a single-line comment naming the
   `Brian.Postulate` / `Mathlib` lemma it relies on.
4. **Check the closure** — the final `theorem H###_*` discharges its goal
   with zero open obligations (`lean --json` reports `messages: []` and
   `n_sorry: 0`).

### 12.4 Building out the THSD Lean library is part of the work

The first hypothesis whose proof needs a THSD construct that doesn't yet
exist in `lean/Brian/` is responsible for **landing that construct** in the
same commit. Examples:

- H001 (`PhiMonotone`) needs `Brian.Thsd.Phi.Phi`, `Brian.Thsd.Sheaf.add_coupling`,
  and a `Phi_monotone_under_add` lemma. The H001 commit ships all three +
  the proof that consumes them, with TDD-style Lean test files under
  `lean/test/` exercising the new lemmas in isolation.
- H002 (`OodGapDecrease`) needs `Brian.Cdga` + the gap-non-increase lemma.
  Same rule: land them in the H002 commit.

No "stub now, prove later" — that pattern is what produced today's five
`sorry`-laden files. A hypothesis without a complete `.lean` proof is
**unverified** and may not be cited as evidence in `docs/findings.md`,
`docs/technical_report.md`, or any commit message claiming "[VERIFIED]".

### 12.5 Until the Lean toolchain is on the CI host

The local + CI fallback `LeanVerdict(status="skipped")` is acceptable **only**
as a transient state. The moment the Lean toolchain (`lean`, `lake`,
`Mathlib`) lands in the CI image, the gate hardens: `status="skipped"`
becomes a build failure. Every existing `hypothesis/proofs/*.lean` must
have been promoted to `status="verified"` by that point or the hypothesis
gets demoted in the ledger.

The corresponding tracking row lives in `docs/formal_framework.md` §10.2;
update it in the same commit that lands a Lean proof, never separately.

---

## 13. ONE venv. Never create another.

This repo has exactly **one** Python virtual environment: `./.venv`.
It runs Python 3.13.x and contains every dependency the project needs
(see `pyproject.toml` + `[ml]` extras). Use it, fix it, install into it.
**Never create `.venv-2`, `.venv-deploy`, `.venv-test`, `venv/`, or any
sibling.** This rule exists because earlier sessions silently created
12 of them (totalling ~4.9 GB) trying to work around transient install
failures — every one of those failures was solvable inside the single
canonical venv.

**Allowed operations:**
- `pip install <pkg>` into `.venv` (after activating it or via
  `.venv\Scripts\python.exe -m pip install ...`).
- `pip install -e .[ml,dev]` to reinstall everything from
  `pyproject.toml` after a clean.
- Wipe `.venv` entirely and rebuild it from scratch — same name, same
  Python, same purpose. This is a reset, not a new venv.

**Forbidden:**
- Creating any second venv directory (`.venv-2`, `.venv-deploy`, `env/`,
  `.virtualenv/`, anything). If a tool insists on a separate
  environment, that's a code smell — the tool should be installed into
  `.venv` like everything else, OR a Conda env named `brian-<purpose>`
  may be used (but only with explicit user approval).
- Hard-coding `.venv-2` (or any sibling) anywhere in the codebase. The
  `_find_deploy_python()` helper and similar interpreter-routing logic
  must use the active interpreter (`sys.executable`) or `./.venv` —
  never a numbered sibling.
- Committing `.venv*` to git. `.gitignore` already excludes it; if
  a future agent edits `.gitignore` to "un-ignore" a venv, revert that
  change.

**If `.venv` is broken:**
1. `Remove-Item .venv -Recurse -Force` (or `rm -rf .venv` on Linux).
2. `py -3.13 -m venv .venv` (Windows) or `python3.13 -m venv .venv`.
3. `.venv\Scripts\python.exe -m pip install -e .[ml,dev]`.
4. Verify: `.venv\Scripts\python.exe -m pytest tests/ -x` runs.

That sequence is the **only** way to get a fresh environment. Adding a
new venv next to the old one is never the answer — it just hides the
real problem and consumes another gigabyte of disk.

---

## 14. No stubs, no scaffolds — always FULL TDD implementations

§1 says "tests first". This rule says **the implementation that lands
behind those tests must be the real thing.** No placeholders, no
pass-through shims, no `TODO: implement later`, no `if not enabled:
return x` as a substitute for actually building the mechanism.

**What counts as a stub (and is therefore banned):**

- A function whose body is `pass`, `return None`, `raise NotImplementedError`,
  or `return x` (identity pass-through) when the test only checks
  shape / type / "doesn't crash".
- A "default-OFF" toggle that, when ON, runs a trivial linear layer
  instead of the documented mathematics. The toggle controls **whether
  the real implementation is wired in**, never **what the implementation
  does**.
- A test that only asserts the module instantiates / has the right
  attribute names / preserves shape. Shape-only contracts are a stub
  smell — write the test that pins the actual mathematical behaviour
  (a known input/output pair, an invariant, an analytical limit).
- A docstring promising behaviour that the body does not deliver.
- Wiring a mechanism into `arch.neuro` whose backing Python class is
  any of the above. The DSL must never reference vapourware.

**What "FULL TDD implementation" requires for any new mechanism:**

1. **Write the contract.** Tests that pin the actual math — e.g. for a
   Poincaré-disc attention head, assert that `‖projected_q‖ < 1` for
   arbitrary input, that two vectors with hyperbolic distance 0 produce
   maximum attention weight, that the gradient flows through the Möbius
   addition. Not "the output has shape `[B, T, D]`."
2. **Run them — confirm RED for the right reason.** "Test fails because
   the function doesn't exist" is fine; "test fails because of a typo in
   the test" is not.
3. **Implement the real mechanism.** The formula in the equation block
   must match the code. The code must be numerically sound (no NaNs on
   edge cases the test checks). It must run end-to-end on CPU at minimum
   (CUDA-only code is not allowed until a CPU fallback is tested).
4. **Run the contract — confirm GREEN.** Plus the surrounding suite
   (`pytest tests/<area>/ -q`) to confirm no regression.
5. **Wire it in.** The `feature` block in `arch.neuro` references the
   real equation; the mechanism is reachable from the model. If the
   feature defaults to `active: false`, that's a research choice (clean
   baseline for A/B), **not** a license to leave the implementation
   half-built.
6. **Commit the test + implementation + wiring together.** One logical
   change, one commit. Per §10, if this is a research mechanism, also
   log the H## hypothesis row in `findings.md` describing what the
   first A/B run is meant to measure.

**Allowed exceptions (rare, must be acknowledged in conversation):**

- A genuine algorithmic primitive that needs an upstream library
  decision (e.g. "geoopt vs hand-rolled Möbius math"). Document the
  decision point in the conversation, then implement one path fully.
  Do not commit a `try: import geoopt except: pass` placeholder.
- A mechanism whose mathematics genuinely is "identity unless an
  external signal flips it" (e.g. an actual no-op residual connection).
  These exist but are rare; if you find yourself writing one for a
  research mechanism, double-check you haven't accidentally designed
  the feature out of existence.

**Why this rule exists.** Stub-driven development produces repos where
half the modules are decorative — they exist in the DSL, they have
tests, they have docs, but flipping their toggle does literally nothing.
That's worse than the mechanism not existing: it lets hypothesis rows in
`findings.md` claim "A/B tested, no effect" when the truth is "the A and
B sides ran identical code." Every claim in `findings.md` and
`technical_report.md` must trace back to a real implementation that does
what its equation says it does.
