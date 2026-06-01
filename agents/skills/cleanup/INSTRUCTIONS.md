# Repo declutter — instructions for `brian ai cleanup`

You are running a one-shot declutter pass. The goal is to **reduce the
cognitive load on the human** by archiving or deleting files that have
served their purpose. Be conservative — when in doubt, archive instead
of delete.

Read the top-level `CLAUDE.md` rules **first** (especially sections 3,
4, and 5). They define what "lives in this repo" and what doesn't.

---

## What you're allowed to touch

- Top-level `*.md` files that are NOT `README.md` or `CLAUDE.md`.
- `docs/*.md` files that are NOT `architecture.md`, `dsl.md`,
  `history.md`, `changelog.md`, `metrics.md`, `findings.md`,
  `dsl_nn_language.md`, `dsl_subsystem_roadmap.md`.
- `*_SUMMARY.md`, `*_COMPLETE.md`, `FINAL_*.md`, `COMMIT_*.md`,
  `*_INVESTIGATION.md` anywhere — these are session artifacts.
- `_scratch.py`, `_check.py`, `_dbg.py`, `_test*.py` at the repo root
  (anything matching `^_[a-z]+(test|check|dbg|scratch).*\.py$`).
- `*.tmp`, `*.bak` anywhere outside `node_modules` / `.venv`.
- Empty or near-empty (<5 lines) untracked notebooks.
- `logs/vast/inspect.log`, `logs/vast/temp.log`.

## What you must NEVER touch

- `architectures/`, `neuroslm/`, `tests/`, `scripts/` source files.
- Anything under `lfs_checkpoints/`.
- Git-tracked log files unless they've been superseded (very rare —
  prefer archive over delete for logs).
- Anything in `docs/archive/` (already archived).
- `colab_run.ipynb`.

---

## Procedure

1. Read `git status -uall` to enumerate untracked files. These are the
   prime candidates for cleanup.
2. List every `*.md` at the repo root + under `docs/` (NOT `docs/archive/`).
3. For each candidate, classify it:
   - **DELETE** if it's a session artifact with no historical value
     (e.g. another agent's "I finished task X" stamp that's already
     captured in commit history).
   - **ARCHIVE** if it has historical value but isn't load-bearing
     (debug investigations, completion reports). Rename to
     `docs/archive/YYYY-MM-DD_<original-name>.md` using the file's
     mtime as the date.
   - **KEEP** if it's referenced by code, listed in `CLAUDE.md`'s
     allowed list, or actively maintained.
4. Apply the moves/deletes. Use `git mv` for tracked files, plain `mv`
   for untracked.
5. Print a summary in this exact form:

```
== brian ai cleanup ==

Deleted (N files):
  - path/to/file.md  (reason: session artifact, content in commit abc1234)

Archived (M files):
  - path/from -> docs/archive/2026-xx-xx_name.md

Kept (K files):
  - path/to/file.md  (load-bearing: <reason>)

Disk freed: <rough estimate>
```

6. STOP. Do not commit, do not push. The user reviews + commits.

---

## What NOT to do

- Don't run experiments, train, or eval — pure file ops.
- Don't fabricate content. If a file's purpose is unclear, mark it KEEP
  with "unclear purpose, leaving for human review" and move on.
- Don't touch the agent skills under `agents/skills/`.
- Don't restructure folder layouts — only file-level moves.
- Don't delete `logs/vast/*.log` files — those are training evidence.
