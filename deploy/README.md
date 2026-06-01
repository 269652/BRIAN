# `deploy/` — vast.ai deployment scripts

**Load-bearing. Do not delete.** These are the only paths through which
training runs and OOD evals are launched onto vast.ai.

Both scripts wrap `vastai create instance` directly in Python (bypassing
the bash equivalents under `scripts/`, which stall on Windows due to
multi-venv vastai-import detection).

| Script              | Invoked by                  | Purpose                                                  |
| ------------------- | --------------------------- | -------------------------------------------------------- |
| `train_dsl.py`      | `brian deploy` / manual run | Launch a DSL training run from arch.neuro `scales {...}` |
| `ood_eval.py`       | `brian eval ood ...`        | Spin a throwaway A100 to eval a checkpoint via WikiText  |

## Hardware envelope (both scripts)

Restricted to **A100_SXM4** only with:

- `reliability > 0.995`
- `gpu_ram >= 40 GB` (post-filtered after `vastai search` — CLI's
  `gpu_ram>=N` query is unreliable; cheap MIG slices show up as A100s)
- `inet_down >= 200 Mbps`
- `verified=true`, `rentable=true`

These constraints match `architectures/rcc_bowtie/arch.neuro`'s
`hardware { ... }` block and are the result of multiple stuck-loading
incidents on cheaper PCIE / unverified offers (e.g. instance 38965155
sat in `loading` for 30+ min on a $0.69/hr PCIE A100 and had to be
destroyed manually).

## How they self-destroy

Each onstart sets up a self-destroy at the end:

```bash
yes y | vastai destroy instance "$SELF_ID"
```

So the instance stops billing once training/eval completes and logs +
checkpoints are pushed to the branch. Without this, instances idle at
$1+/hr after finishing.

## What NOT to do

- Don't run these as background daemons — they're one-shot launchers.
  They exit ~5s after `vastai create` returns.
- Don't add interactive prompts — they're invoked from Python.
- Don't hardcode credentials. Both read `VAST_API_KEY` + `GITHUB` from
  `.env` at the repo root.
