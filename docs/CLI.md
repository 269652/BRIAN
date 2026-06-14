# NeuroSLM Command-Line Reference

Complete reference for all CLI tools and training scripts in NeuroSLM.

---

## Training

### `python -m neuroslm.train_dsl` — DSL-Driven Training (Recommended)

Train using the declarative `.neuro` architecture files.

**Usage:**
```bash
python -m neuroslm.train_dsl \
    --arch architectures/rcc_bowtie \
    --scale 30m_p4 \
    --steps 10000 \
    --device cuda:0 \
    --ckpt_dir lfs_checkpoints
```

**Key arguments:**

| Flag | Default | Description |
|------|---------|-------------|
| `--arch` | required | Architecture folder (e.g., `architectures/rcc_bowtie`) |
| `--scale` | `30m_p4` | Preset (30m_p4, 100m, 300m, 1b, 7b) from arch.neuro scales block |
| `--steps` | 10000 | Total training steps |
| `--batch` | — | Override batch size (from arch.neuro by default) |
| `--seq_len` | — | Override sequence length |
| `--device` | `cpu` | Device: `cpu`, `cuda:0`, `cuda:1`, etc. |
| `--ckpt_dir` | `checkpoints/` | Where to save checkpoints |
| `--keep_last_n_ckpt` | 3 | Keep N most recent checkpoints (others pruned) |
| `--prune_git` | True | Use git rm to prune old checkpoints (atomic) |
| `--overwrite_ckpt` | False | Single `_latest.pt` mode (no rotation) |
| `--mode` | `"mix"` | Data source: "mix" (FineWeb+OpenHermes), "wikitext", "synthetic" |

**Example: quick smoke test (10 steps, synthetic data)**
```bash
python -m neuroslm.train_dsl \
    --arch architectures/rcc_bowtie \
    --scale tiny \
    --steps 10 \
    --device cpu
```

**Example: 100M model on A100 (fp32, 4 grad-accum steps)**
```bash
python -m neuroslm.train_dsl \
    --arch architectures/rcc_bowtie \
    --scale 100m \
    --steps 50000 \
    --device cuda:0
```

**Checkpoint naming:**
```
neuroslm_<preset>_<params>M_<optimizer>[_baseline]_<step>.pt
neuroslm_pct_30m_68M_adamw_mix_best.pt
neuroslm_rcc_bowtie_30m_p4_step10000.pt
```

---

### `python -m neuroslm.train` — Hand-Written Model Training (Legacy)

Train the hand-crafted `Brain` class directly (not DSL-compiled).

**Usage:**
```bash
python -m neuroslm.train \
    --preset xl \
    --steps 100000 \
    --batch_size 4 \
    --optimizer adamw
```

**Key arguments:**

| Flag | Default | Description |
|------|---------|-------------|
| `--preset` | `large` | tiny, small, medium, large, xl, xxl |
| `--steps` | 10000 | Total training steps |
| `--batch_size` | 4 | Batch size |
| `--device` | `cuda` | Device specification |
| `--optimizer` | `adamw` | adamw or adafactor |
| `--learning_rate` | 0.0003 | Initial LR |
| `--baseline` | False | Vanilla transformer ablation at matched params |
| `--resume` | None | "latest" to continue last run, or explicit ckpt path |

**Example: Baseline ablation (vanilla transformer at matched params)**
```bash
python -m neuroslm.train --preset xl --baseline --steps 80000
```

---

## Inference & Generation

### `python -m neuroslm.generate` — Interactive Generation

Generate text from a trained checkpoint.

**Usage:**
```bash
python -m neuroslm.generate \
    --ckpt lfs_checkpoints/neuroslm_pct_30m_68M_adamw_mix_best.pt \
    --prompt "Once upon a time" \
    --max_tokens 256 \
    --temperature 0.7
```

**Key arguments:**

| Flag | Default | Description |
|------|---------|-------------|
| `--ckpt` | None | Checkpoint path (required) |
| `--prompt` | "" | Starting text |
| `--max_tokens` | 256 | Max tokens to generate |
| `--temperature` | 0.7 | Sampling temperature (0 = greedy) |
| `--top_p` | 0.9 | Nucleus sampling threshold |
| `--device` | `cuda` | Device for inference |
| `--interactive` | False | REPL mode (keep generating) |

**Example: Interactive mode**
```bash
python -m neuroslm.generate \
    --ckpt lfs_checkpoints/neuroslm_pct_30m_68M_adamw_mix_best.pt \
    --interactive
# Prompts you for input after each generation
```

---

## Evaluation

### `python scripts/vast_show_logs.py` — Fetch & Stream Vast.ai Logs

Fetch stdout/stderr logs from a running vast.ai instance. Supports streaming (follow) and polling modes.

**Usage:**
```bash
# Fetch logs for an instance
python scripts/vast_show_logs.py --instance-id 37240129 --dest logs/37240129.log

# Stream logs (follow mode, if vastai CLI supports it)
python scripts/vast_show_logs.py --instance-id 37240129 --follow

# Poll for new logs every 5 seconds
python scripts/vast_show_logs.py --instance-id 37240129 --poll 5.0

# Show last 100 lines
python scripts/vast_show_logs.py --instance-id 37240129 --tail 100

# Find by label and combine with polling
python scripts/vast_show_logs.py --label "my-training-run" --poll 10 --dest logs/my-run.log

# Fetch all instances in parallel
python scripts/vast_show_logs.py --all --workers 4
```

**Key arguments:**

| Flag | Type | Default | Description |
|------|------|---------|-------------|
| `--instance-id` | str | None | Instance ID (required if --label not set) |
| `--label` | str | None | Label substring to match instances |
| `--follow` | bool | False | Stream logs using vastai CLI (requires CLI installed) |
| `--tail` | int | None | Show last N lines only |
| `--poll` | float | 0 | Poll interval in seconds (polling mode; 0 = no polling) |
| `--dest` | str | None | Local file to write/append logs to |
| `--all` | bool | False | Fetch logs for all instances (parallel) |
| `--workers` | int | 4 | Number of parallel workers for --all |

**Examples:**

```bash
# Live stream with follow
python scripts/vast_show_logs.py --instance-id 12345 --follow

# Poll every 10 seconds, save to file
python scripts/vast_show_logs.py --instance-id 12345 --poll 10 --dest training.log

# Show last 50 lines
python scripts/vast_show_logs.py --instance-id 12345 --tail 50

# Poll + tail combo: last 100 lines, update every 5 seconds
python scripts/vast_show_logs.py --instance-id 12345 --tail 100 --poll 5
```

---

### `bash scripts/vast_ood_eval.sh` — Out-of-Distribution Evaluation

Evaluate OOD perplexity on WikiText-103-v1 (academic prose).

**Usage:**
```bash
CKPT=lfs_checkpoints/neuroslm_pct_30m_68M_adamw_mix_best.pt \
BRANCH=arch/my-feature \
ROLE_TAG=my-feature-eval \
bash scripts/vast_ood_eval.sh
```

**Environment variables:**

| Variable | Required | Description |
|----------|----------|-------------|
| `CKPT` | Yes | Checkpoint path |
| `BRANCH` | Yes | Branch name (for result naming) |
| `ROLE_TAG` | Yes | Descriptive tag for this run (my-feature-eval) |
| `VAST_GPU_QUERY` | No | GPU filter (default: A100 verified, reliability >0.99) |

**Output:** Result JSON in `results/ood_<tag>_<params>M_step<step>.json` with:
```json
{
  "train_ppl": 400.9,
  "ood_ppl": 1806.6,
  "gap_ratio": 4.51,
  "checkpoint": "lfs_checkpoints/...",
  "branch": "arch/my-feature",
  "timestamp": "2026-06-01T15:30:00Z"
}
```

---

### `python scripts/maintain_technical_report.py` — Documentation Audit

Audit technical_report.md against the codebase for drift.

**Usage:**
```bash
# Check for issues
python scripts/maintain_technical_report.py --verbose

# Apply auto-fixes (archive old docs)
python scripts/maintain_technical_report.py --fix
```

**What it checks:**
- Presets in `arch.neuro` are documented in the report
- Key hyperparameters (loss_clipping, dropout, pct_trunk) are sync'd
- Stale files (OOD_PUSH_STAGES.md) are identified for archiving

---

## Analysis & Logging

### `brian analyze-log` — Analyze Training Logs (Brian Skill)

Parse vast.ai training logs and extract metrics.

**Usage:**
```bash
brian analyze-log logs/vast/b49e69448613_rcc_bowtie_134M_lm-first-v11-100m_step7500of40k.log
```

**Output:** Structured analysis in `logs/analyzed/<name>.md`:
- Convergence trajectory (PPL @ step)
- Loss composition (lm, world, Φ, etc.)
- Auxiliary metric trends (gnorm, NT levels, MAT)
- Failure modes or interesting transitions

---

## Utility Commands

### `python -m neuroslm.tools.prune_ckpts` — Checkpoint Rotation

Manually prune old checkpoints (usually automatic in train loop).

**Usage:**
```bash
python -m neuroslm.tools.prune_ckpts \
    --dirs lfs_checkpoints checkpoints \
    --keep 3 \
    --git
```

**Arguments:**

| Flag | Default | Description |
|------|---------|-------------|
| `--dirs` | required | Directories to prune |
| `--keep` | 3 | Number of recent checkpoints to retain per stream |
| `--git` | False | Use `git rm` for atomic cleanup |

**Grouping:** Checkpoints are grouped by stream (preset, optimizer, mode). Each stream is pruned independently.

---

## Deployment (vast.ai)

### `bash scripts/vast_train_dsl_loop.sh` — Deploy Training Job

Launch a training job on vast.ai (runs inside the instance).

**Setup (on your local machine):**
```bash
export ARCH_PATH=architectures/rcc_bowtie
export SCALE=30m_p4
export NUM_STEPS=10000

# Creates instance (you configure GPU query in vast.ai UI)
vast create instance ... 
# Then SSH in and run the script inside
```

**Inside the instance:**
```bash
bash scripts/vast_train_dsl_loop.sh
# Handles:
# - Data download
# - Training loop with auto-resume
# - Checkpoint upload (git lfs push)
# - Graceful shutdown
```

**Configuration (edit script):**
- ARCH_PATH: which architecture to train
- SCALE: preset size (30m_p4, 100m, etc.)
- NUM_STEPS: training horizon
- BATCH_SIZE, SEQ_LEN: from arch.neuro by default

---

### Cheap 2k-step smoke run from `dna/evol/arch.dna`

Use this when you've changed `architectures/rcc_bowtie/` (or the MoE
experts roster in `multi_cortex`) and want a short, cheap end-to-end
verification on vast.ai before committing to a full A100 run. Costs
~$0.30-0.60 for the whole run (RTX 3090 @ $0.15-0.25/hr × ~2 hours).

**Prerequisites:**
1. The DNA must be current. If you touched `arch.neuro`, recompile per
   CLAUDE.md §14:
   ```powershell
   python -c "from neuroslm.compiler.ribosome import RibosomeCompiler; `
              RibosomeCompiler().compile_file('architectures/evol', `
                                              'dna/evol/arch.dna')"
   ```
2. Verify the invariant: `python -m pytest tests/test_evol_dna_has_experts.py -q`
3. `.env` carries `VAST_API_KEY=...` and `GITHUB=<token>`.

**Local dry-run (free, sanity-only):**
```powershell
# Unfolds dna/evol/arch.dna into .neuro/arch/temp/ and lifts the
# Hypergraph IR — same path the vast.ai job will take, just without
# the actual training.
brian train --dna dna/evol/arch.dna --steps 10 --preset cheap_2k
```

**Deploy to vast.ai (2k steps, RTX 3090):**
```powershell
# _deploy_train.py reads `multi_cortex.experts` from the DNA's
# embedded arch.neuro and provisions an RTX 3090 instance via the
# `cheap_2k` scale's hardware{} override.
$env:DNA = "dna/evol/arch.dna"
$env:SCALE = "cheap_2k"
$env:STEPS = "2000"
$env:OOD_EVERY = "500"   # 4 OOD evals over the run
python _deploy_train.py
```

**What the run produces:**
- 4 OOD evals (steps 500, 1000, 1500, 2000) — enough to see whether
  loss is still falling without paying for a full run.
- ~4 checkpoints at `lfs_checkpoints/dsl_arch_cheap_2k_stepXXXXX.pt`.
- `/workspace/train.log` containing per-step `train_ppl`, `ood_ppl`,
  `cortex_alpha`, MoE router-weight distribution per domain.
- Total token budget: 2000 × 2 × 1024 × 2 (grad_accum) ≈ 8M training
  tokens — about 1/65 of GPT-2's 524M-per-step pretraining batch.

**When this run is enough:**
- Initial CE on natural English at step 0 should be **3-5 nats** (not
  ~10.85 as on the old random-projection path). If it's >7 nats, the
  MoE wiring is silently broken — abort and check
  `tests/training/test_lm_expert_harness_integration.py::TestSmokingGunCE`.
- `train_ppl` should fall from ~50-100 at step 0 to ~20-30 by step 2000.
- Router weights should NOT all collapse onto a single expert (telemetry
  field `cortex/router_weights/*`); if they do, lexical bias is too low.

**When you need more than 2k:**
- `train_ppl` hasn't dropped below 60 by step 2000 → architectural
  problem, not a learning-rate problem. Don't extend the run; investigate.
- `ood_ppl` is still falling fast at step 2000 → switch to the full
  `30m_p4` scale (or `100m`) on an A100 for the 10k-40k step trajectory.

**Record the result** per CLAUDE.md §10 — even a 2k smoke run is an
experiment with a verdict; add a row to `docs/findings.md` with the
vast.ai instance id and the cost.

---

## Git LFS Checkpoint Management

### Skip LFS downloads (recommended for laptops)

```bash
# Download only pointer stubs (~130 B each)
git lfs install --local --skip-smudge
git clone <repo>
```

### Pull a specific checkpoint

```bash
# Fetch a single file
git lfs pull --include="lfs_checkpoints/neuroslm_30m_p4_step10000.pt"

# Or by glob (all 10k-step files)
git lfs pull --include="lfs_checkpoints/*_step10000*"
```

### Rehydrate entire LFS repo

```bash
git lfs install --local --force  # re-enable smudge
git lfs pull                      # fetch all
```

---

## Tips & Troubleshooting

### "No valid GPU found" on vast.ai

**Solution:** Reduce `min_gpu_mem_gib` or `min_reliability` in arch.neuro's hardware block.

### Training diverges after step 5k

**Likely cause:** Loss clipping or maturity phasing misconfigured. Check `arch.neuro` training block (loss_clipping, pct_trunk, dropout).

**Verify:** Run `python scripts/maintain_technical_report.py --verbose` to check hyperparams are synced.

### Checkpoint resume fails ("state_dict key mismatch")

**Solution:** Manually load with `--strict=False` (ignores missing keys) or re-run from step 0.

### OOD eval is slow

**Reason:** WikiText-103-v1 eval runs 200 windows × 1024 seq_len, ~5-10 min per checkpoint on A100.

**Workaround:** Use a smaller test set (modify `scripts/vast_ood_eval.sh`).

---

## Full Workflow Example

```bash
# 1. Clone and setup
git clone <repo>
cd neuroslm
pip install -r requirements.txt
py -3 -m pytest tests/test_phi.py -v  # verify setup

# 2. Quick smoke test (1 min)
python -m neuroslm.train_dsl --arch architectures/rcc_bowtie --scale tiny --steps 10

# 3. Train locally (CPU, 30 min)
python -m neuroslm.train_dsl --arch architectures/rcc_bowtie --scale 30m_p4 --steps 1000 --device cpu

# 4. Train on GPU (A100, ~8 hours for 10k steps)
python -m neuroslm.train_dsl --arch architectures/rcc_bowtie --scale 30m_p4 --steps 10000 --device cuda:0

# 5. Generate from checkpoint
python -m neuroslm.generate --ckpt lfs_checkpoints/neuroslm_30m_p4_*.pt --prompt "Once upon"

# 6. Audit docs before commit
python scripts/maintain_technical_report.py --verbose

# 7. Commit
git add architectures/rcc_bowtie/arch.neuro docs/technical_report.md ...
git commit -m "feat: new mechanism or hyperparameter change"
```

---

**For more details:** See [`CONTRIBUTING.md`](../CONTRIBUTING.md) (contributor guide) or [`technical_report.md`](technical_report.md) (project overview).
