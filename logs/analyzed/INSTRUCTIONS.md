# Analysis SOP — turn raw vast.ai logs into evidence

> **Audience.** An LLM (Copilot / Claude / Cursor agent) processing one or
> more raw stdout captures under `logs/vast/`.
>
> **Output.** For each input log, this folder gains one Markdown analysis
> file with a speaking name, the raw log is renamed to the same stem,
> and `docs/FINDINGS.md` is updated with any new insight that affects
> a hypothesis row. All actions are reproducible and grounded in the
> log content — no metric appears in the analysis without a line
> citation back to the raw log.
>
> **Trigger.** Run this SOP whenever new files appear in `logs/vast/`
> (after `npm run sync:logs` finishes), or on-demand against any
> already-renamed log file.

---

## Inputs

```
logs/vast/<instance-id>.log                    # fresh capture (id-named)
logs/vast/<instance-id>__<label>.log           # if sync got a label
logs/vast/<descriptive-name>.log               # already-analyzed (skip)
```

Plus, always read:
- `docs/FINDINGS.md` — the live hypothesis ledger.
- `docs/architecture.md` — to disambiguate §-references that appear in
  training/eval logs (e.g. "§5.2 trunk-iso", "§5.5 PCT").
- `results/*.json` — to cross-reference any train_ppl / OOD_ppl
  numbers cited in the log against the canonical eval JSONs.

---

## Outputs

```
logs/vast/<descriptive-name>.log               # renamed input
logs/analyzed/<descriptive-name>.md            # the analysis
docs/FINDINGS.md                               # updated if new insights
```

`<descriptive-name>` is a kebab-case slug, ≤ 64 chars, of the form:
```
<role>_<branch-short>_<ckpt-or-step>_<UTC-date>
```
where:
- `role` ∈ {`train`, `ood-eval`, `bootstrap-fail`, `dpo`, `ablation`,
  `smoke`, `bench`, `synth`, `pct`} — chosen by reading the log.
- `branch-short` is the active git branch slug minus the `arch/`
  `stabilize/` `docs/` namespace prefix (`predictive-coding-trunk` →
  `pct-trunk`; `trunk-grad-isolation` → `trunk-iso`;
  `recursive-reasoning` → `recursive`; `synthesis-v1` → `synth-v1`).
- `ckpt-or-step` is the most-advanced step number seen, or the ckpt
  filename stem (`mix_best`, `mix_7000`).
- `UTC-date` is `YYYYMMDD` of the *first* log timestamp.

Examples that have actually been produced:
- `ood-eval_trunk-iso_baseline-mix-80000_20260525.log`
- `ood-eval_pct-trunk_pct-30m-mix-best_20260525.log`
- `train_synth-v1_mix-best_20260525.log`

If two distinct phases appear in one log (e.g. bootstrap → training →
checkpoint push), keep them in one file; the descriptive name names
the *dominant* phase.

---

## Procedure (do these steps in order; no parallelism)

### 1. Inventory the input

```bash
ls -la logs/vast/
```

For each `<id>.log` (or `<id>__<label>.log`) that **does not yet** have
a matching `logs/analyzed/<descriptive-name>.md`:

1. Read the **first 100 lines** for: clone URL / branch / commit /
   env vars (`BRANCH=`, `CKPT=`, `ROLE_TAG=`, `MAX_OOD_WINDOWS=`,
   `--baseline` flag in any python command, etc.).
2. Read the **last 200 lines** for: final exit status, final loss /
   PPL / gap_ratio printout, JSON pushed back to origin, instance
   labels in the wrap-up.
3. Sample the **middle** every ~5 000 lines to catch divergences,
   NaN spikes, CUDA OOM, or rapid log-spam regions (commonly
   per-step or per-batch training prints).

### 2. Classify

Decide the role from log content:

| Signal in log | Role |
|---|---|
| `brian_ood_test.py` invocation, "OOD PPL" / "train PPL" lines | `ood-eval` |
| `python -m neuroslm.train ...` + per-step `lm_loss` lines | `train` |
| Errors before any training output (apt failures, pip failures, `git lfs pull` errors) | `bootstrap-fail` |
| `tests/test_*.py::test_` listing, pytest summary | `smoke` |
| `dpo` / `--reward_model` / `pi_loss` / `kl_to_ref` | `dpo` |
| Repeated short runs with `--ablation` / `ROLE_TAG=baseline` / `ROLE_TAG=experiment` pairs | `ablation` |

If the log is empty, truncated, or shows only the vast.ai onstart
banner with no real work, name it `bootstrap-fail` and note the
failure mode in the analysis.

### 3. Extract evidence

Pull these fields whenever they appear. Cite each with a line range
back to the raw log (`L1234-L1236`).

**Identity / setup**
- branch, commit hash (from `git rev-parse` or clone output)
- checkpoint loaded (`--checkpoint` arg or `Resuming from ...` line)
- preset / params (`n_params: ...`, `cfg.preset = ...`)
- GPU model, instance label, instance id, vast offer id
- tokenizer + vocab size + ctx_len (from the `[ood] loaded BRIAN: ... ctx=...` line for evals)

**Training metrics (per-step or per-checkpoint)**
- step, lm_loss, total_loss
- Φ value, ignition rate, gnorm (if logged)
- maturity index / MAT
- per-aux-loss weights and their gated values
- λ values (`lambda_motor`, `lambda_mem`, `lambda_thought`) if printed
- the `best_loss` / `best_step` recorded at each save

**Eval metrics (OOD runs)**
- train_ppl, OOD_ppl, gap_ratio, verdict — these are the canonical
  numbers; cross-check against `results/ood_*.json`.

**Failures / anomalies**
- any NaN / Inf / "training diverged" / loss explosion
- gradient norm spikes (>10×) and their step
- vast.ai disconnects, OOM, network failures
- "missing keys" / "unexpected keys" from `load_state_dict`
- the legacy-default-fallback diag messages (`[diag] saved cfg missing 'X' -> legacy False`)

**Pushes / commits**
- any `git commit` / `git push` lines (record the SHA pushed)
- any `ood_results_*.json` filename pushed back to origin

### 4. Pick the descriptive name

Following the schema in §Outputs. If two analyses would collide on
the same name (e.g. two ablation runs the same day), append a `_a` /
`_b` suffix in load order.

### 5. Rename the raw log

```bash
git mv logs/vast/<id>.log logs/vast/<descriptive-name>.log
```

(Use `git mv` not `mv` so the rename is recorded as a rename, not a
delete + add, when the file is already tracked. For an untracked
file just `mv` is fine.)

### 6. Write `logs/analyzed/<descriptive-name>.md`

Use this template — fill every field; write `(not in log)` where the
field genuinely doesn't appear:

```markdown
# <descriptive-name>

**Raw log:** [logs/vast/<descriptive-name>.log](../vast/<descriptive-name>.log)
**Instance id:** <id> (vast.ai)
**Role:** <role>
**Branch / commit:** <branch> @ <sha-short>
**Started (UTC):** <first timestamp in log>
**Completed (UTC):** <last timestamp> — <exit status>

## Setup
- Checkpoint loaded: `<path>` (step <N>, <params>M)
- Preset: `<preset>`
- GPU: <model>
- Tokenizer: <name> (vocab <V>, ctx <C>)

## Headline numbers
| Metric | Value | Cited from |
|---|---|---|
| train_ppl | … | L<line>-L<line> |
| OOD_ppl | … | L<line>-L<line> |
| gap_ratio | … | L<line>-L<line> |
| (for train runs: final lm_loss / step) | … | L<line> |

## Trajectory (training runs only)
Brief prose of how loss moved — e.g. "lm_loss 9.2 → 4.1 over 7000
steps, smooth except for a +0.6 spike at step 3247 (L4192) that
recovered within 80 steps."

## Anomalies / failures
- (list each with L<line> citation; "(none)" if clean)

## Pushes
- Pushed `ood_results_<tag>.json` to `<branch>` (L<line>)
- Committed checkpoint(s): … (L<line>)

## Cross-checks
- The train_ppl in this log (X.Y) **matches / does not match** the
  committed value in `results/ood_<…>.json` (Y.Z). If mismatch, note
  why (different harness invocation, seed, etc.).

## Insights for FINDINGS.md
- (one bullet per insight that affects a hypothesis row in FINDINGS.md;
  "(none — confirms existing data)" if nothing new)

## Verbatim excerpts
```
<one or two short, load-bearing log excerpts that the metrics
above cite — keeps the analysis self-contained even if the raw log
is later pruned by checkpoint-rotation logic>
```
```

### 7. Update `docs/FINDINGS.md` only if there's a real insight

A "real insight" means one of:
- A new metric for an existing row in the **Reference table**
  (different ckpt step, different mix, different harness flag).
- A revealed failure mode (e.g. legacy-default fallback firing, NaN
  in a specific module) that fits in "Things that broke" or refines
  a row's status.
- A reproducibility hole (e.g. an `ood_results_*.json` cited in the
  log but missing from `results/`).
- Evidence that contradicts a stated verdict.

**Do not** update FINDINGS for "ran the same eval again and got the
same number" — link the new analysis md from the existing row
instead.

When updating, follow these rules:
- Cite the analysis file: `(see logs/analyzed/<descriptive-name>.md)`.
- Don't restate full metrics in FINDINGS — keep FINDINGS terse;
  the analysis md is the long form.
- Status downgrades (✅ → 🟡, 🟡 → ❌) require *explicit* evidence
  in the log; otherwise note it in the row's caveat list instead.

### 8. Sanity-check the result

```bash
# These all should be true:
ls logs/vast/<descriptive-name>.log
ls logs/analyzed/<descriptive-name>.md
git status logs/ docs/FINDINGS.md
```

The analyzed md should be 80–250 lines. If it's shorter, the log
probably has more to say. If it's longer than 400 lines, you're
summarising too much — push detail into the verbatim excerpts and
keep the prose tight.

### 9. Commit

One commit per analyzed log:

```
git add logs/vast/<descriptive-name>.log logs/analyzed/<descriptive-name>.md
git add docs/FINDINGS.md   # only if §7 updated it
git commit -m "logs: analyze <descriptive-name> (<role>, <branch-short>)"
```

Don't batch multiple unrelated logs into one commit — bisectability
matters for the analysis layer too.

---

## Naming rules — the boring details

| Aspect | Rule |
|---|---|
| Case | lowercase only |
| Separator within tokens | hyphens |
| Separator between tokens | single underscore |
| Date format | `YYYYMMDD` (no separators) |
| Max length | 64 chars |
| Allowed chars | `[a-z0-9._-]` |
| Forbidden | spaces, slashes, uppercase, `@`, `#`, `:` |

A name that violates these is rejected — re-derive.

---

## What this SOP deliberately does not do

- **Does not run code or training.** Analysis is read-only on the
  log; if a metric is missing from the log, write "(not in log)"
  — do not attempt to recompute it from the checkpoint.
- **Does not delete raw logs.** Logs are evidence. The rename is the
  only mutation. Pruning is a separate operation done by humans.
- **Does not modify `results/*.json`.** Those are the canonical
  benchmark outputs; analysis may *cite* them but never rewrite them.
- **Does not edit `docs/architecture.md`.** Architecture spec is
  authored, not derived. If a log reveals a discrepancy with the
  spec, note it in the analysis md and as a new hypothesis row in
  FINDINGS — but the spec edit is a human decision.

---

## Example walk-through (the first analysis to ship under this SOP)

The log `logs/vast/37764350.log` is the first input. After this SOP
runs over it, you should see:

```
logs/vast/<chosen-name>.log         # renamed
logs/analyzed/<chosen-name>.md      # filled-in template
docs/FINDINGS.md                    # possibly updated audit-notes / backlog
```

The choice of `<chosen-name>` is the LLM's job — derive it from the
log contents per §4. Don't reuse `37764350` in any committed
filename; the instance id lives only inside the analysis md as a
metadata field.
