# `hooks/` — declarative CLI lifecycle hooks

Cross-platform extension points for the BRIAN CLI. A hook is a pair
of scripts (`.sh` for Unix, `.ps1` for Windows) selected by an
event-name YAML manifest in this folder.

## Why

Some CLI ceremony points — `brian deploy` chief among them — have
project-specific pre/post flight checks that should NOT live inside
the binary distribution. Examples:

- Recompile `architectures/master` → `dna/evol/arch.dna` before each
  deploy so the on-disk DNA never drifts from the source-of-truth
  `.neuro` files (this is the shipped `pre-deploy` hook).
- Push a Slack notification when a deploy lands.
- Sync a heatmap from `architectures/current` after a successful run.
- Refresh a local LLM-as-judge inference server before an OOD eval.

A YAML hook folder gives you all of that with zero changes to
`neuroslm/cli.py`.

## Event-name → file mapping

| Hook event       | File                       | Triggered by                                    |
| ---------------- | -------------------------- | ----------------------------------------------- |
| `pre-deploy`     | `hooks/pre-deploy.yaml`    | `cmd_deploy` (before any vast.ai network call)  |

The shipped `pre-deploy` implements the 5-step "roundtrip + publish"
pipeline:

1. **Clean-check** — refuse if `git status --porcelain` is non-empty.
2. **Compile** `architectures/master` → `dna/master/arch.dna`
3. **Unfold** `dna/master/arch.dna` → `architectures/current/arch.neuro`
4. **Commit** — `git add -A` + `git commit -m "chore: roundtrip recompile of current architecture"` (no-op when the roundtrip is idempotent)
5. **Push** — `git push`. Only on success does the hook return 0, unblocking the deploy.

More events (`post-deploy`, `pre-train`, `pre-compile`, …) can be
added by wiring a single `_run_hook("event-name")` line at the right
spot in `cli.py`; the runner module is event-agnostic.

## YAML schema

```yaml
name: pre-deploy                 # informational; filename is authoritative
description: One-liner.          # shown in --help and the [hook] banner
enabled: true                    # default true; false → silent skip
fail_on_error: true              # default true; false → log + return 0
timeout_seconds: 300             # default 300; 0 = no timeout
scripts:
  windows: hooks/scripts/pre-deploy.ps1
  unix:    hooks/scripts/pre-deploy.sh
```

## What the runner gives your script

| Env var            | Meaning                                                |
| ------------------ | ------------------------------------------------------ |
| `BRIAN_HOOK_NAME`  | Name of the hook event being fired                    |
| `BRIAN_REPO_ROOT`  | Absolute path of the repo root                         |

All output streams live (stdout + stderr are NOT captured), so a
long-running compile remains observable in the user's terminal.

## Locked behavioural contract

`tests/test_hooks.py` covers:

- YAML loading (5 cases)
- Runner OS dispatch (powershell on Windows, bash on Unix)
- `fail_on_error` propagation
- `disabled` short-circuit
- `cmd_deploy` runs `pre-deploy` BEFORE any vast.ai call
- A non-zero pre-deploy aborts `cmd_deploy` with the hook's exit code
- The shipped `pre-deploy` YAML + both scripts exist and invoke
  `brian dna compile` + `brian dna unfold`

Bump those tests when you change the contract.
