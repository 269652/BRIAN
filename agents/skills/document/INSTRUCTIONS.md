# Documentation-maintenance instructions for `brian ai document`

This file is the prompt loaded by `brian ai document`. It's a recurring
chore: given the repo's current state (commit history, training logs,
OOD eval JSONs), produce a clean, evidence-backed paper trail of what
we tried, what worked, and what didn't.

`brian ai document` invokes the `claude` CLI with this file as the
prompt and the repo root as the working dir, so you can read any file
to gather evidence.

---

## Files you maintain

- **`docs/history.md`** — chronological narrative of every architectural
  hypothesis we made, **with citations** (commit SHAs + log file paths
  + line numbers). Each entry is shaped as:

  ```
  ### YYYY-MM-DD · <Hypothesis name>

  **Claim:** <what we expected — verbatim if possible>
  **Source:** <commit SHA / doc filename>
  **Test:** <which run/log proved or disproved it>
  **Result:** <PROVEN | DISPROVEN | INCONCLUSIVE> — <one-line summary>
  **Evidence:**
    - logfile: `logs/vast/<file>.log:<line>` — "<quoted line>"
    - ood JSON: `logs/vast/benchmarks/ood/<file>.json` — `gap_ratio: X`
    - metric: `docs/metrics.md` row `<run_id>`
  ```

  When a claim is **DISPROVEN** you must link the specific log line
  that contradicts it (file path + line number + quoted text).

- **`docs/changelog.md`** — generated from `git log` since project start.
  One section per month. Per commit: SHA (short), date, scope tag
  (arch/cli/harness/fix/etc.), one-line summary. Skip merge commits.

- **`docs/archive/`** — historical docs that aren't load-bearing. When
  you find a doc in `docs/` whose claims have been answered (PROVEN or
  DISPROVEN), move it here, renaming to `YYYY-MM-DD_original-name.md`
  and add a one-line header citing the entry in `history.md` that
  superseded it.

- **`docs/metrics.md`** — already auto-maintained by `brian analyze-log`;
  don't rewrite, but verify rows match the runs cited in `history.md`.

---

## Inputs you can read

- `git log --oneline` (commit history)
- `git log -p -- <file>` (per-file history)
- `git show <SHA>` (specific commit)
- Every file under `docs/`, `logs/vast/`, `architectures/`,
  `neuroslm/`, `tests/`
- OOD eval results: `logs/vast/benchmarks/ood/*.json`
- Training logs: `logs/vast/*__neuroslm-*.log` and
  `logs/vast/*_step*of*.log`

---

## Output discipline

- **Cite or omit.** Never write "X improved Y" without naming the run.
  If you can't find the evidence, mark it INCONCLUSIVE and link the run
  you'd need to check.
- **Quote verbatim.** When citing a log line, paste it as a code block
  with the path + line number, e.g.:
  ```
  logs/vast/101ceb95a960__neuroslm-full.log:822
  > [train_dsl] PASS-MARK EARLY EXIT @ step 5000: ood_not_falling: ood_ppl not falling
  ```
- **Don't invent metrics.** If a number isn't in a log or JSON, don't
  put it in `history.md`.
- **Date everything.** Use `YYYY-MM-DD` from the log's first timestamp
  or the commit date — not "today".

---

## Recurring task (what to do each invocation)

1. Read `docs/history.md` if it exists (else start fresh from this list).
2. Walk every `.md` file under `docs/` (excluding `archive/` and the
   ones you maintain: `history.md`, `changelog.md`, `metrics.md`,
   `architecture.md`, `dsl.md`).
3. For each claim or hypothesis you find:
   a. Search `logs/vast/` and `logs/vast/benchmarks/ood/` for matching
      evidence.
   b. Add or update the entry in `docs/history.md` with the citation.
   c. If the doc is fully answered, move it to `docs/archive/` and
      add a one-line `> Superseded by docs/history.md#<anchor>` header
      at the top of the archived copy.
4. Refresh `docs/changelog.md` from `git log`:
   - Show all commits since the last entry in changelog.md (or since
     the project root commit if changelog.md is empty).
   - Group by month.
   - Strip the `Co-Authored-By:` lines from messages.
5. Sanity-check: every run mentioned in `history.md` should have at
   least one row in `metrics.md` (or a note explaining why it doesn't).

If you can't find evidence for a claim that was previously marked
PROVEN, downgrade it to INCONCLUSIVE and explain.

---

## What NOT to do

- Don't delete `docs/architecture.md`, `dsl.md`, `dsl_nn_language.md`,
  `dsl_subsystem_roadmap.md`, `findings.md` — those are load-bearing.
- Don't rewrite `docs/metrics.md` — `brian analyze-log` owns it.
- Don't fabricate citations. A missing evidence link is better than a
  wrong one.
- Don't change source code, run experiments, or push to git. This task
  is documentation-only.

---

## When you're done

Print a short summary:

```
Updated docs/history.md (+N entries / M revised).
Updated docs/changelog.md (+K commits since last entry).
Archived: <list of files moved>
INCONCLUSIVE claims awaiting evidence: <count>
```

Then stop. The user reviews + commits manually.
