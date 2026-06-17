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

**Strict TDD is non-negotiable.** "I'll add tests after" is the
single biggest source of regressions in this repo. RED-GREEN-COMMIT
in that order. No exceptions for "small" changes — small changes are
exactly where regressions hide.

---

## 1b. Read the existing pieces FIRST

Before writing any new code:

1. **Search the codebase** for existing helpers that solve part of the
   problem. Use `grep_search`, `semantic_search`, or `list_code_usages`
   on the relevant function/class names.
2. **Read the file you're about to edit**, top to bottom of the
   relevant region, before making the first edit. Match the surrounding
   conventions (naming, error handling, logging style, type-hint style).
3. **Reuse before reinventing.** If `_find_latest_log` already walks
   `logs/vast/`, the new `--latest` flag calls it; it does NOT define
   a new walker. If `_detect_exit_reason` already parses the tail of a
   training log, the new "destroyed instance" code path calls it; it
   does NOT re-grep the log inline.
4. **Honour the file's existing structure.** Helpers go next to other
   helpers. Argparse subparsers go in the argparse block. Tests go in
   the file that already covers the module under test.

Why this matters: every uncoordinated reinvention is two implementations
of the same thing that will silently diverge. The bug is then not in
either copy individually — it's in the *gap between them*. That bug is
the hardest class to diagnose because it doesn't live in any single
file.

The grep-first / read-first / reuse-first discipline takes ~2 minutes
per change and saves ~2 hours per regression. Always pay the 2 minutes.

---

## 1c. Use `brian test`, not raw `pytest`

The repo has a unified test driver — **always** use it instead of
calling `pytest` directly. The three subcommands match the loop you
actually want:

| Command            | What it runs                                    | When to use it                                              |
| ------------------ | ----------------------------------------------- | ----------------------------------------------------------- |
| `brian test quick` | The 30 most-recently-modified test files (mtime) | After editing a handful of tests — re-checks your active edits |
| `brian test fast`  | The 30 fastest individual tests (cached durations) | Near-zero-cost smoke check; loads most of the import graph in <5 s |
| `brian test full`  | Canonical full sweep + refresh duration cache   | Before commit / push; ~8 min on a workstation               |
| `brian test PATH`  | (legacy) pytest on a path/pattern               | Targeted re-runs of a single file or directory              |

Why this matters:

- **One source of truth for the exclusion list.** Files like
  `tests/test_feature_flag_ablation.py`, `tests/test_brian_compile.py`,
  and `tests/training/` are excluded by `full` (they're broken or very
  slow) and ALSO skipped by `quick`/`fast` so they never block your loop.
  Raw `pytest` calls don't share this list and routinely trip on the
  same broken files.
- **`fast` is free**. The duration cache at `.neuro/test_durations.json`
  is rewritten on every `brian test full`. After one full sweep you get
  a 30-test smoke check for the price of about 5 seconds of wall time.
- **`quick` tracks your work**. Picks files by mtime, so the moment you
  edit a test file it jumps into the next `quick` run automatically —
  no need to remember to pass `-k` or a path.
- **No more "I ran pytest on a subset and missed a regression"**. The
  `full` sweep is the same invocation every time, on every machine.

**Forbidden patterns** (will produce drift between contributors):

```
# DON'T:
pytest tests/foo.py
.venv\Scripts\python.exe -m pytest tests/ -q
python -m pytest tests/dsl -k something

# DO:
brian test tests/foo.py            # targeted
brian test full                    # canonical sweep
brian test quick                   # what I'm actively editing
brian test fast                    # smoke check after rebase
```

The only legitimate reason to invoke `pytest` directly is **inside**
the CLI driver itself (`neuroslm/cli.py::cmd_test_*`). Everywhere else
— shell sessions, CI scripts, doc snippets, agent commands — go
through `brian test`.

When writing new tests, RED-confirm them with
`brian test tests/path/to/new_test.py` (legacy form), then GREEN-confirm
with the same. Don't reach for `pytest`.

---

## 1d. Use the existing `.venv` — **never** create a new one

> **STRICT RULE — ONE VENV, ONE NAME, NO EXCEPTIONS.**
> The only Python environment that exists for this repo is
> **`./.venv/`**. You may not create, suggest, configure, switch to,
> or "temporarily try" *any* other Python environment under *any*
> circumstance. Not `venv/`, not `env/`, not `.venv2/`, not a conda
> env, not `uv venv`, not `poetry shell`, not a "fresh interpreter
> just to test something". If `.venv/` exists, **use it**. If it
> doesn't, **re-create it at the exact path `./.venv/`** with
> `python -m venv .venv && .venv\Scripts\pip.exe install -e .` and
> then use it. There is no second case.
>
> This rule applies to **tool calls too**: do not call any tool
> that would create, register, or activate a new environment
> (`configure_python_environment` against a fresh path, `uv venv`,
> `python -m venv <anything but .venv>`, `conda create`, etc.).
> When asked to "set up the environment", the answer is always
> "use `./.venv/`".

The repo ships a single, canonical virtual environment at
`./.venv/` with every dependency the project needs (torch, transformers,
pytest, vastai, the editable `brian` console script, etc.). **Always
use it.** Every drift between environments has, historically, produced
a day of "works on my machine" debugging.

**The one and only Python interpreter for this repo:**

```
.venv\Scripts\python.exe        # Windows (PowerShell / cmd)
.venv/bin/python                # macOS / Linux
```

**Everything routes through it:**

| What you want                  | The right invocation                                  |
| ------------------------------ | ----------------------------------------------------- |
| Run the CLI                    | `.venv\Scripts\brian.exe <subcommand>`                |
| Run tests                      | `.venv\Scripts\brian.exe test <quick\|fast\|full\|PATH>` |
| Run a one-off Python script    | `.venv\Scripts\python.exe path\to\script.py`          |
| Install a new package          | `.venv\Scripts\pip.exe install <pkg>` (then update `requirements.txt` / `pyproject.toml`) |
| Activate interactively         | `.venv\Scripts\Activate.ps1`                          |

**Forbidden patterns** (immediate red flag in review):

```
# DON'T:
python -m venv venv              # creates a SECOND env
python -m venv .venv2            # same problem, different name
python -m venv ./env             # same problem, different name
uv venv                          # ditto
uv venv .venv-new                # ditto
poetry shell                     # creates a poetry-managed env
poetry install                   # ditto (installs into poetry's env, not .venv)
conda create -n brian ...        # ditto
mamba create -n brian ...        # ditto
python script.py                 # bare `python` -> whichever python is on PATH (often Python 2.7 or a system python)
pip install foo                  # bare `pip` -> installs into the wrong env

# DO:
.venv\Scripts\python.exe script.py
.venv\Scripts\pip.exe install foo
.venv\Scripts\brian.exe test quick
```

**Tool-call equivalents that are equally forbidden** (the rule applies
to *any* path that materialises a second environment, including via
agent tools and IDE configuration):

```
# DON'T (tool-shaped):
configure_python_environment  → pointed at a fresh path (creates a new venv)
install_python_packages       → into anything other than the resolved .venv
create_and_run_task           → that invokes `python -m venv ...` for a non-.venv path
notebook_install_packages     → into a kernel whose interpreter is not .venv

# DO (tool-shaped):
configure_python_environment  → only when the resolved env is already ./.venv/
install_python_packages       → only after confirming the active env is ./.venv/
```

The bare command `python` on Windows frequently resolves to
`C:\Python27\python.exe` or the Microsoft Store stub — neither of
which has any of this repo's dependencies. **Always use the absolute
`.venv\Scripts\python.exe` path** (or activate first). If something
"isn't installed", the answer is almost always *wrong interpreter*,
not *missing package*.

If `.venv` is somehow corrupted or missing, the correct fix is to
delete it and re-create it **at the same path** with
`python -m venv .venv && .venv\Scripts\pip.exe install -e .` —
never to create it under a different name.

---

## 1e. **NEVER** deploy to vast.ai without explicit permission

> **STRICT RULE — NO UNAUTHORISED VAST.AI DEPLOY, EVER.**
> You may not, under *any* circumstance, launch, rent, provision,
> resume, restart, or otherwise spend money on a vast.ai instance
> without the user explicitly saying so in the **current** turn.
> "The user said yes 20 turns ago" does **not** count. "It looks
> like the obvious next step" does **not** count. "I'll just spin
> up a tiny one to test the deploy script" does **not** count.
> "The previous run died and I'll just relaunch it" does **not**
> count. **Every** vast.ai launch is a fresh, explicit, in-turn
> authorisation — or it does not happen.

This rule applies to **every** path that costs real money on vast.ai:

```
# DON'T (any of these without explicit "yes, deploy" in the current turn):
.venv\Scripts\brian.exe deploy ...
.venv\Scripts\brian.exe vast launch ...
.venv\Scripts\brian.exe vast resume ...
.venv\Scripts\brian.exe vast restart ...
vastai create instance ...
vastai start instance ...
python deploy/train_dsl.py            # any path that ssh-launches a remote run
python _deploy_train.py
bash deploy/*.sh                       # any deploy shell script
ssh root@<vast-ip> ...                 # provisioning / rsync into a running instance
rsync ... root@<vast-ip>:...           # ditto
```

Tool-call equivalents that are equally forbidden without explicit
in-turn permission:

```
# DON'T (tool-shaped, without explicit "deploy" / "launch" / "rent" in this turn):
run_in_terminal  → invoking `brian deploy`, `brian vast launch/resume/restart`,
                    `vastai create/start instance`, `python _deploy_train.py`,
                    any `ssh root@<vast-ip>`, any `rsync ... root@<vast-ip>:`
create_and_run_task → that wraps any of the above
```

**Safe operations** (read-only, no spend, never need permission):

```
# OK any time:
.venv\Scripts\brian.exe vast list           # list instances
.venv\Scripts\brian.exe vast status         # status of a running instance
.venv\Scripts\brian.exe vast logs --latest  # tail the most-recent run's log
.venv\Scripts\brian.exe vast tail ...       # tail a specific log
vastai show instances                       # raw list
ssh root@<vast-ip> 'tail -f /workspace/training.log'   # read-only inspection
```

**Stop / destroy** also requires explicit permission (it interrupts a
paid run that may be the user's most expensive resource of the day):

```
# DON'T without explicit "stop" / "destroy" / "kill" in the current turn:
.venv\Scripts\brian.exe vast stop ...
.venv\Scripts\brian.exe vast destroy ...
vastai destroy instance ...
```

If you *think* a deploy is the obvious next step — **say so in the
conversation and stop**. Wait for the user to say "yes, deploy" (or
equivalent unambiguous green light) in the **same** turn before
invoking any spend-incurring tool. The cost of a wrongly-launched
H100 is non-trivial; the cost of asking one clarifying question is
zero.

Why this matters: agent action that costs real money is the single
category of mistake that cannot be undone by a `git revert`. Treat
every vast.ai spend as load-bearing on the user's wallet — because
it is.

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
- **Don't deploy to vast.ai without explicit ask** — every `brian deploy`
  creates a billable instance. Even a "quick test" deploy costs real money.
  Always confirm before calling `brian deploy` or any equivalent.
- **Don't destroy vast.ai instances without explicit ask** — running
  jobs cost money to restart, not to keep going.
- **Don't reset / force-push / amend pushed commits** unless the user
  explicitly says so.

Match the scope of your actions to what was requested. If the user
asks to "fix the bug," they didn't ask for a refactor — they didn't
ask for renaming, they didn't ask for a new abstraction. Stay tight.

### 8.1 Secrets never enter the repo

- **API tokens / write-keys / passwords belong in `.env`** (gitignored;
  `.env.example` is the public template). The chain `_deploy_train.py`
  uses is:
  1. Process env (`$HF_TOKEN`, `$GITHUB`, `$VAST_API_KEY`) wins.
  2. `os.environ.setdefault(...)` falls back to the matching line in
     `.env`.
  3. For runtime-only auth (HF Hub), `~/.huggingface/token` (from
     `huggingface-cli login`) is the last-ditch fallback.
- **Never** add a literal token to source, tests, fixtures, docstrings,
  comments, log files, `.md` evidence files, or commit messages.
  `grep -r "hf_[A-Za-z0-9]\{20,\}" .` and the GitHub-side equivalents
  for `ghp_` / `vast_` should always return zero matches inside
  `git ls-files`.
- **Never** print a rendered token-containing string to stdout/stderr
  (the `ONSTART` script in `_deploy_train.py` is the canonical example
  of what NOT to log — it gets POSTed to vast.ai's API but is never
  echoed locally).
- **When checking in a new secret-bearing flow:** add the field to
  `.env.example` with a comment explaining what it auths against, add
  a pre-flight warning in the consumer (so a missing token surfaces
  at deploy-time, not 500 steps into a $1.50/hr GPU run), and the
  consumer must "fail open" — print a clean skip message, never crash
  the training loop.

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

### 9.1 DSL/arch ↔ technical_report.md ↔ README.md cross-alignment (anti-drift)

**Trigger:** any change to one of these surfaces must be cross-aligned
with **both** `docs/technical_report.md` *and* the top-level `README.md`
**in the same commit**:

- `architectures/*/arch.neuro` (and any `architectures/*/modules/*.neuro`)
- `neuroslm/dsl/*.py` — DSL grammar, parser, training_config, regularization
- `neuroslm/compiler/*.py` — Ribosome / NFG / Hypergraph IR
- `neuroslm/experts.py` — multi-cortex routing, abstain logits, KL distill
- `neuroslm/harness.py` — fusion gate, inhibition, loss composition
- `dna/*/*.dna` (compiled snapshots — usually downstream of an arch edit)
- Anything that changes the **schema** of what the DSL accepts, what the
  Hypergraph IR exposes, what `dna/evol/arch.dna` carries, or what loss
  terms / telemetry the trainer emits

**What "cross-aligned" means:** every commit touching the surfaces above
must do **one** of the following:

1. **Update `docs/technical_report.md`** so the document still describes
   what the code actually does. The §s most likely to drift:
   - §3 (architecture overview — match arch.neuro topology)
   - §5 (mechanisms — match `multi_cortex`, `regularization`, `hardware`)
   - §6 (DSL surface — match grammar/parser changes)
   - §7.2 (current state — match the active arch + scale + preset)
2. **AND update `README.md`** — the README is the human-facing entry
   point that every external visitor reads first. The §s most likely to
   drift on an architectural change:
   - `## System Architecture` (block diagram, layer count, dims)
   - `## The .neuro DSL` (DSL snippets must compile against current grammar)
   - `## Multi-Cortex Fusion` (experts roster, fusion gate semantics,
     abstain logit formula, KL-distill schedule)
   - `## Loss composition` (every loss term name + weight that
     `harness.py` actually emits)
   - `## Parameter presets` (preset names and param counts must match
     `brian.toml` / `training_config.py`)
   - `## Quick start` (CLI flags must match `train_dsl.py` / `brian`)
3. **OR** add a single-line `[NO SPEC IMPACT]` annotation in the commit
   body explaining why no doc update is needed (typo fix, internal
   rename, perf-only refactor with identical observable behaviour).
   Pure perf wins like adding a cache or memoisation typically qualify;
   anything that changes loss, throughput envelopes, what gets logged,
   what the CLI accepts, or what the architecture diagram should show
   does not.

**Pre-commit check** (run before you `git add`):

```bash
python scripts/maintain_technical_report.py --verbose
```

The script greps `arch.neuro` for every section/field name that
`technical_report.md` mentions and reports drifted/missing references.
Non-zero exit = doc drift = do not commit. If the script flags a false
positive, fix the script — never silence the warning.

> **README audit:** the script currently audits `technical_report.md`
> only. README drift must be checked by eye: open `README.md` next to
> the modified file (`arch.neuro` / `experts.py` / etc.) and confirm
> every claim still holds. If you add a new mechanism to the trunk,
> ask: "would a first-time visitor reading the README know this exists,
> and would the snippet they copy-paste still run?" If no, edit README.

**Why this rule exists.** Earlier sessions added the `experts: [...]`
roster, the `hardware{}` block, the `cheap_2k`/`t4_2k` scales, and the
`brian.toml [defaults]` precedence — all real architectural surface —
without touching `technical_report.md`. External AIs reading the report
then made plans against a 6-month-stale picture of the system. Worse,
external humans landing on the GitHub README read the *headline* of the
project and tried to reproduce a setup that no longer existed. The
README is the contract with the outside world; the technical_report is
the contract with collaborating AIs. If either is wrong, every
downstream reasoning step is wrong.

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
- README.md §Multi-Cortex Fusion: update experts list + telemetry
  example so visitors see the actual roster, not the legacy one
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

### 9.3 README/template proofreading — technical accuracy is mandatory

**CRITICAL:** Every statement added to `README.template.md` must be
technically accurate. The README is the first thing external visitors
see. Incorrect claims damage credibility.

**Data-driven requirement:** Every number, metric, or empirical claim in
`README.template.md` MUST use a `${METRIC}` placeholder from
`docs/readme_metrics.toml`. NO hardcoded numbers except:
- Version numbers (Python 3.10+, PyTorch 2.x, IIT 4.0)
- Architectural constants (11-stage bowtie, 10×10 GridWorld, 28 populations)
- Code example literals (count: 32, gain: 0.6)
- Section/anchor references (§12, #12-the-neuro)
- Hypothesis IDs (H1, H6.5, H22)

**Smoke test enforcement:** Run `pytest tests/test_readme_quality.py` before
committing README changes. The tests enforce:
1. No hardcoded empirical claims (use ${METRIC})
2. All ${METRIC} exist in readme_metrics.toml
3. No unrendered ${...} in final README.md
4. All metrics are properly formatted

**Common errors to avoid:**

1. **Wrong compilation pipeline:**
   - ❌ "arch.neuro compiles to PyTorch"
   - ✅ "arch.neuro → Hypergraph IR → PyTorch"
   - **The Hypergraph IR is the source of truth for wiring**, not arch.neuro
     directly. arch.neuro is parsed into IR, then IR generates PyTorch.

2. **Oversimplified mechanism claims:**
   - ❌ "cortex experts teach the trunk"
   - ✅ "KL distillation from detached cortex logits into trunk (gradient
     flows trunk-only)"
   - Precision matters. The mechanism must match the actual code.

3. **Hardcoded numbers instead of metrics:**
   - ❌ "1511 tests passing"
   - ✅ "${TOTAL_TESTS} tests passing" (in template) → "1511 tests passing" (in rendered README)
   - Update `docs/readme_metrics.toml` when results change

**Proofreading checklist before editing README.template.md:**

- [ ] Is this claim actually true? Check the code/tests/logs.
- [ ] Does the compilation pipeline match reality? (arch.neuro → IR → PyTorch)
- [ ] Are all metrics using `${...}` placeholders from readme_metrics.toml?
- [ ] Did I run `pytest tests/test_readme_quality.py` and fix all violations?
- [ ] Would a first-time visitor misunderstand this sentence?
- [ ] Does the code snippet actually run against current HEAD?

**If you add claims to the README, proofread them.** No exceptions.

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

### 10.1 Log naming + boot-stamp forensics (audit trail contract)

**Every** training log **must** be named with step range in the filename
and **every** log body **must** open with a 3-line boot stamp. This is
the contract that makes a deploy auditable months later without git
archaeology.

**Filename format** (enforced by `scripts/log_pusher.sh::_compose_logfile`
and the `0002_log_name_refactor` migration):

```
logs/<YYYYMMDD>/<arch>/<HHMMSS>_<start>_<end>.log
```

Two-level hierarchy: **day → arch**, with step-range filenames. The day
comes from `${BOOT_TIMESTAMP:0:8}`, the arch from `$ARCH_NAME` (the same
value used by `brian train --arch <name>`), and the filename encodes:
- `HHMMSS` = boot time (UTC) from `${BOOT_TIMESTAMP:9:6}`
- `start` = first step number (0 for fresh runs, >0 for resumed)
- `end` = last/current step number

The hierarchy mirrors how a human searches the directory: `ls logs/`
shows the days, `ls logs/<day>/` shows which architectures ran that
day, `ls logs/<day>/<arch>/` shows each run with its boot time and
step range visible at a glance.

Worked example:

```
logs/20260615/neuroslm-full-dna-arch/175931_0_10000.log
└─── day ─┴──────── arch ────────┴─ time ─┴─ steps ─┘
```

- The HHMMSS boot time provides unique per-run identity (two runs on
  the same day still get distinct filenames without coordinating ids).
- The step range makes completion status immediately visible without
  opening the file (e.g., `175931_0_7800.log` vs `175931_0_10000.log`).
- The filename updates on each push as training progresses — old files
  with lower step counts are removed automatically.
- `BOOT_TIMESTAMP` is set ONCE in `scripts/vast_train.sh` via
  `date -u +%Y%m%dT%H%M%SZ` and propagated as an env var to both the
  background `log_pusher.sh` and the final one-shot push.

**Layout history** (kept here so legacy log paths still parse):

| Era                | Path                                                                                                         |
| ------------------ | ------------------------------------------------------------------------------------------------------------ |
| pre-2026-06-15     | `logs/vast/<stamp>_<id>_..._stepNofN.log` (flat)                                                             |
| 2026-06-15-am      | `logs/vast/<YYYYMMDD>/<ARCH>/<stamp>_<id>_..._stepNofN.log` (nested, mutating leaf name)                     |
| 2026-06-15-pm      | `logs/<YYYYMMDD>-<HHMMSS>_<arch>_<sha>/train.log` (flat per-run folder)                                      |
| 2026-06-15-eve     | `logs/<YYYYMMDD>/<arch>/<HHMMSS>_<sha>/train.log` (3-level hierarchy)                                        |
| 2026-06-16         | `logs/<YYYYMMDD>/<arch>/<HHMMSS>_<start>_<end>.log` (step-range filename — **current**)                     |

The `0001_logs_to_run_folders` migration normalised eras 1-3 into the
3-level layout. The `0002_log_name_refactor` migration converts era 4
into the current step-range format. Both are idempotent (tracked in
`.brian/migrations.json`).

**Key log reference** (for migration/clean protection):

- `logs/20260615/neuroslm-full-dna-arch/175931_0_10000.log` —
  H22 SmolLM2 upgrade run, 1.12B total params (146.9M trainable),
  WikiText-103 PPL 155.0, train PPL 23.6, gap_ratio 6.55. First
  complete 10k run with the new cortex fusion stack. Checkpoint:
  `hf://moritzroessler/BRIAN/checkpoints/20260615-175931_c19bf629_neuroslm-full-dna-arch/step10000.pt`

**Boot stamp format** (enforced by
`neuroslm/train_dsl.py::_print_boot_stamp`, called from `main()`):

```
[train_dsl] boot @ 2026-06-14T16:04:23Z
[train_dsl] git_commit a22eecc4e7b9c8d6f5a3b2e1d0c9b8a7f6e5d4c3 (master)
[train_dsl] arch_dsl_sha256 7f4a2b8e1c9d3a5f6b7e8c9d0a1b2c3d4e5f6a7b8c9d0e1f2a3b4c5d6e7f8a9b (architectures/rcc_bowtie)
```

- **Line 1**: UTC ISO-8601 with `Z` suffix — when the trainer booted.
- **Line 2**: 40-hex git sha + branch — pin the exact commit that
  produced this run, even if it was later rebased or force-pushed away.
- **Line 3**: 64-hex SHA-256 of the canonical concatenation of all
  `*.neuro` files under `arch_root`, sorted by relative path with a
  `--- {relpath} ---` separator between files. Pins the **unfolded
  DSL state**, including any uncommitted local edits at deploy time.

**Why both?** The git sha catches "what commit did this come from".
The DSL sha catches "what arch did the trainer actually see" — useful
when you deploy with uncommitted changes (DEFAULT for fast iteration)
or when the arch_root resolution picks a different DNA than expected.
Two runs with identical git shas but different DSL shas → someone
modified the working tree between launches. Two runs with identical
DSL shas but different metrics → a non-DSL change (data, seed, tokeniser).

**Pre-commit check.** Boot stamp + filename are pinned by:

- `tests/training/test_train_dsl_boot_stamp.py` (9 contracts)
- `tests/training/test_log_pusher_naming.py` (4 contracts: timestamp
  prefix, per-run folder layout, day→arch→`<time>_<sha>` hierarchy,
  no `_PREV_LOG` cleanup)
- `tests/test_migration_0001.py` (14 contracts: parsing, planning,
  reference gate, applying — same folder shape as the pusher writes)

All three must stay GREEN. If you rename a helper, update the test. If
you change the format, write the new contract first (TDD per §1) — the
forensic value is the format itself, not the implementation.

**Anti-patterns to refuse:**

- Stripping the timestamp prefix because "the instance id is enough" —
  it isn't; vast.ai recycles ids on destroy/recreate within hours.
- Putting the timestamp at the END of the filename — defeats the
  chronological sort that makes incidents triageable at a glance.
- Logging the git sha but skipping the DSL sha because "git already
  has it" — git only has it for **committed** state. Deploys with
  uncommitted edits (common during a rapid debug loop) leave no other
  trace of what the trainer actually saw.
- Catching and swallowing an exception inside `_print_boot_stamp` to
  print `?` for one of the lines — the helper is wrapped in a broad
  try/except that already returns `"-"` placeholders; if you see `-`
  in real logs, fix the helper, do not hide the failure.

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
4. Verify: `brian test full` (or `brian test quick` for a fast spot-check) runs.

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
   (`brian test tests/<area>/`) to confirm no regression. Never call
   `pytest` directly — see §1c.
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

---

## 15. Always use `brian` CLI commands for repo operations

Every workspace-level operation — training, deploying, destroying,
checking status, compiling DNA, rendering an NFG, running OOD eval —
has a `brian` subcommand. **Use it.** Do not call the underlying
shell scripts or Python entry points directly.

| Operation                | ✅ Use this                                  | ❌ Do not call this directly                        |
|--------------------------|---------------------------------------------|-----------------------------------------------------|
| Launch training run      | `brian deploy [--steps N] [--branch X]`     | `python _deploy_train.py` · `bash scripts/vast_train.sh` |
| Resume previous run      | `brian deploy --resume <path-or-hf-uri>`    | manually editing env vars                          |
| Resume latest from HF    | `brian deploy --latest [--hf-prefix RUN]`   | manual `huggingface_hub` download                  |
| Long-horizon run         | `brian deploy-100k`                         | same                                                |
| Kill a vast.ai instance  | `brian destroy <ID>`                        | `bash scripts/vast.sh destroy instance <ID> -y`     |
| List active instances    | `brian ps [--it]` · `brian status`          | `vastai show instances`                             |
| Tail container logs      | `brian logs <ID>`                           | `vastai logs <ID>`                                  |
| List HF checkpoints      | `brian hf list [--prefix RUN]`              | `huggingface_hub.HfApi().list_repo_files(...)`     |
| Download HF checkpoint   | `brian hf pull <path>` · `brian hf pull --latest` | `hf_hub_download(...)` + manual copy         |
| Newest HF checkpoint URI | `brian hf latest`                           | manual repo listing                                |
| Always-on chat daemon    | `brian chat [<ckpt>] [--latest]`            | `python -m neuroslm.chat_daemon`                   |
| Compile arch → DNA       | `brian dna compile [<arch>] [-o FILE]`      | `python -m neuroslm.compiler.ribosome ...`          |
| Unfold DNA → DSL         | `brian dna unfold <dna> [-o FILE]`          | same                                                |
| Render NFG diagram       | `brian compile nfg --current` · `brian nfg` | `python -m neuroslm.graphviz ...`                   |
| OOD evaluation           | `brian ood eval <checkpoint>`               | `python deploy/ood_eval.py`                         |
| Train locally            | `brian train [--steps N]`                   | `python -m neuroslm.train_dsl ...`                  |
| Analyse arch             | `brian analyze <arch>`                      | `python -m neuroslm.analyzer ...`                   |

**Why this rule exists.**

1. **brian.toml is the single source of truth.** Every `brian`
   subcommand consults `brian.toml [current]` + `[defaults]` so an
   unflagged invocation produces a fully-defined operation. Bypassing
   the CLI means re-typing `--steps` / `--branch` / `--dna` every time,
   and worse: the bypassed scripts have stale hardcoded fallbacks
   (`_deploy_train.py` defaults to `BRANCH=arch/rcc-p4-loss-clip` from
   May 2026, deeply wrong now). The `brian` CLI layer is what keeps
   those drift bugs from reaching production.
2. **Canonical pipeline guarantees.** `brian deploy --dna ...` calls
   `prepare_run_workspace` LOCALLY before any vast.ai network call, so
   broken DNA fails fast and you never pay for provisioning. The raw
   `_deploy_train.py` skips that check.
3. **Two-file split avoidance.** `brian dna compile` (no args) writes
   to `brian.toml [current].dna` — the exact path the deploy reads.
   Calling the underlying `RibosomeCompiler` directly forces you to
   pick an output path manually, and on 2026-06-14 that misalignment
   shipped a wasted-compute deploy on the stale legacy roster (see
   findings.md H22 sidebar). The CLI closes that gap.
4. **Telemetry, logging, label suffixes.** `brian deploy` automatically
   builds the vast.ai instance label (`neuroslm-full-<scale>-<label>-<source>`),
   sets `LABEL_SUFFIX`, picks the right Python interpreter, scrubs the
   environment of stale `BRIAN_*` vars. Reproducing that by hand is
   exactly the kind of accidental-complexity work that produces silent
   regressions.

**Exceptions.** The CLI layer doesn't have to be a hard wall — there
are legitimate one-offs (debugging a specific subcommand by stepping
through it with `python -m pdb -m neuroslm.cli ...`, calling
`compiler.run_workspace` from a test fixture, etc.). The rule is "use
the CLI for anything you would do MORE THAN ONCE." If you find
yourself reaching for the raw script, either (a) you're debugging and
that's fine, or (b) the CLI is missing a flag — in which case ADD the
flag instead of bypassing.

