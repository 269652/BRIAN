# BRIANHarness — Training Infrastructure

The `BRIANHarness` wraps the compiled `Brain` model and orchestrates the full training loop with loss clipping, gradient accumulation, maturity gating, auxiliary loss scheduling, and periodic OOD evaluation.

---

## Overview

```
BRIANHarness(brain_module, config)
    ↓
    ├─ Micro-batch loop (gradient accumulation)
    │   ├─ brain.forward_lm(ids, targets)  → loss terms
    │   ├─ Clip per-sample losses
    │   ├─ Accumulate gradients (GA steps)
    │   └─ optimizer.step() → brain.step() → grad_clip, trophic update, NT dynamics
    │
    ├─ Maturity gating (soft awakening)
    │   ├─ MAT = 1 - lm_loss / L_random (EMA)
    │   ├─ Phase gates at MAT levels 0.35, 0.45, 0.55, 0.60
    │   └─ Aux-loss ramp: α(t) = maturity_aux_gate(MAT)
    │
    ├─ Per-step metrics (logged)
    │   ├─ lm_loss, train_ppl, gnorm
    │   ├─ Φ, λ₁ (Fiedler value), NT levels
    │   ├─ MAT, oscillation rates, osc (8-freq activity)
    │   └─ Expert routing entropy, trophic activity
    │
    └─ Periodic OOD eval (every N steps)
        ├─ Load WikiText-103-v1 hold-out batches
        ├─ Forward in eval mode (no grad, deterministic)
        ├─ Compute OOD perplexity
        └─ Check early-exit criteria (OOD trend rule)
```

---

## Architecture

### 1. Loss Clipping (`BRIANHarness.forward`)

**Problem:** Large loss spikes cause gradient explosions that diverge training mid-run.

**Solution:** Per-sample loss clipping with configurable method.

```python
# arch.neuro
loss_clipping: {
    enabled: true,
    method: "per_sample",        # or "global", "adaptive"
    factor: 3.0                  # clip all losses ∈ [0, 3.0]
}
```

**Mechanism:**
```python
def clip_loss(loss_per_sample, method="per_sample", factor=3.0):
    if method == "per_sample":
        clipped = torch.clamp(loss_per_sample, 0, factor)
    elif method == "global":
        clipped = loss_per_sample.clamp(0, factor)
    elif method == "adaptive":
        # scale factor based on running mean
        factor_eff = factor * (1 + 0.1 * torch.randn(1))
        clipped = torch.clamp(loss_per_sample, 0, factor_eff)
    return clipped
```

**Evidence:** P4 preset runs to step 10k stably with factor=3.0 on 30M model; factor=0 (no clipping) diverges by step 7-8k.

### 2. Gradient Accumulation

**Purpose:** Train on larger effective batch without OOM by accumulating gradients over micro-batches.

```python
# arch.neuro
grad_accum: 1    # accumulate over N micro-batches before step
```

**In train loop:**
```python
for step in range(num_steps):
    for micro_batch in range(grad_accum):
        ids, targets = next(data_loader)
        loss = brain(ids, targets)
        (loss / grad_accum).backward()  # scale for averaging
    
    optimizer.step()
    optimizer.zero_grad()
```

**Trade-off:** Higher grad_accum = larger effective batch = more stable gradients, but less frequent updates = slower convergence per wall-clock time.

### 3. Maturity Phasing (`brain._phase_gate`)

**Problem:** At step ~5k, auxiliary losses suddenly engage (via old all-or-nothing gating), causing gnorm spike → divergence.

**Solution:** Per-subsystem smooth sigmoid phase gates centered at different MAT (maturity) levels.

```python
MAT = 1 - lm_loss / L_random  # proxy: LM "comprehension" (EMA)

def phase_gate(mat, center, width=0.10):
    return 0.5 * (1 + torch.tanh((mat - center) / width))

# Each aux loss is gated independently:
loss_world = phase_gate(MAT, 0.45) * w_world * L_world
loss_motor = phase_gate(MAT, 0.50) * w_motor * L_motor
loss_phi = phase_gate(MAT, 0.60) * w_phi * L_phi
loss_kl = phase_gate(MAT, 0.60) * w_kl * L_kl
# ... etc
```

**Subsystem onset windows:**

| Subsystem | Phase center | Onset logic |
|-----------|--------------|------------|
| `pred_coding` | 0.35 | Cheap internal supervision; engages earliest |
| `world` | 0.45 | World model grounding once LM bootstraps |
| `motor` / `forward` | 0.50 | Action objectives need working world model |
| `novel_aux` / `cpc` | 0.55 | Contrastive objectives |
| `kl_world` / `Φ` | 0.60 | Heaviest objectives; last to engage |

**Evidence:** With phase gates, training reaches step 10k+ stably. Without them (single on/off switch), divergence happens at ~7-8k during the awakening transition.

### 4. Gradient Isolation (Trunk vs Bio)

**Problem:** Bio-module losses (world, motor, Φ, etc.) corrupt the shared trunk representation the LM head depends on.

**Solution:** `sem = sem.detach()` before feeding to bio pipeline (§5.2 architecture.md).

```python
# In Brain.forward_lm:
sem_bio = sem.detach()  # Stop backward flow here
# bio modules read from detached sem
world_output = world_model(sem_bio)
motor_output = motor_cortex(sem_bio)
phi_loss = compute_phi(brain.modules)  # Still shaping weights via other paths
```

**Effect:** Trunk gradient comes *only* from LM loss + pred_coding. Aux losses shape *their own* modules, not the trunk.

**Evidence:** Test `test_stabilization.py::test_trunk_gradient_invariance` — trunk gradient is invariant to aux-loss weights when isolation is ON.

### 5. ReZero Forward Gates (λ scalars)

**Problem:** Module contributions to the LM forward (motor bias, memory xattn, thought conditioning) were gated by maturity phase-gates, causing a discontinuity at awakening: PPL ~90 → ~370 in one step.

**Solution:** Zero-init learnable scalars λ. LM gradient self-discovers whether the injection helps.

```python
h_biased = h_lang + λ_motor · motor_lang_bias    # was: _motor_phase · …
mem_kv = λ_mem · memory_kv_proj(mem)             # was: _mem_phase · …
lang_thought = λ_thought · lang_thought          # gates from_sem path
```

**Properties:**
- **Identity at init:** λ=0 ⟹ module contribution is zero ⟹ LM behaves like baseline
- **Self-discovery:** ∂L_lm/∂λ is real; if injection helps, λ grows; if not, stays small
- **Bootstrap:** While module output is near-zero (early training), ∂L_lm/∂λ ≈ 0, so λ stays zero—correct, nothing to gate yet

**Evidence:** gap_ratio 5.22 (ReZero) < 6.34 (plain recursive), modest improvement without absolute-OOD win.

### 6. Auxiliary Loss Weighting & Ramp

```python
# arch.neuro
mechanisms: {
    enabled: true
}

# In train loop:
aux_w_scale = maturity_aux_gate(MAT, awaken_floor=0.30)
# Then for each loss term:
total_loss = lm_loss + aux_w_scale * (w_world*L_world + w_motor*L_motor + ...)
```

**Ramp strategy:**
1. **Infancy (MAT < 0.30):** aux_w_scale ≈ 0.001, LM dominates
2. **Awakening (0.30 ≤ MAT < 1.0):** aux_w_scale ramps smoothly to 1.0
3. **Full strength (MAT ≥ 1.0):** aux_w_scale = 1.0, all aux losses at nominal weight

---

## Key Hyperparameters (In `arch.neuro`)

### Loss Clipping
```neuro
loss_clipping: {
    enabled: true,
    method: "per_sample",
    factor: 3.0
}
```
**Impact:** Prevents gradient spikes. Factor=0 → divergence; factor=3.0 → stable to 10k+.

### Regularization
```neuro
dropout: 0.12                  # single biggest OOD lever
stochastic_depth: 0.1         # random block skip during training
flooding_level: 4.0           # prevent train loss from collapsing (maintains grads)
label_smoothing: 0.05
```

**Evidence:** dropout=0.0 → OOD gap blows out to 13×. dropout=0.12 is sweet spot for P4.

### Predictive Coding
```neuro
pct_trunk: 0.4                # use top-down predictors on 40% of samples
pct_strength: 0.3             # weight of PCT loss
```

**Purpose:** Force compositionality via top-down generative predictions. Reduces gap_ratio from 6.12 (baseline) to 4.51 (PCT best).

### Maturity Gating
```neuro
mechanisms: {}                # enable/disable by subsystem
maturity_awaken_floor: 0.30   # MAT below this → infancy gating
```

### Optimizer & LR Schedule
```neuro
optimizer: "adamw"
learning_rate: 0.0003
weight_decay: 0.12
warmup_steps: 300
min_lr_ratio: 0.1             # LR decays to 0.1x initial by end
```

---

## Per-Step Metrics & Logging

Logged at each step:

| Metric | Interpretation |
|--------|-----------------|
| `step` | Training step number |
| `lm_loss` | Cross-entropy loss (main objective) |
| `train_ppl` | exp(lm_loss) |
| `gnorm` | Global gradient norm (clip target ≈1.0) |
| `loss_total` | Sum of all losses (after clipping) |
| `phi` | Integrated information (IIT MIP) |
| `lambda1` | Fiedler value (graph connectivity proxy) |
| `mat` | Maturity (LM comprehension proxy) |
| `nt_*` | Neurotransmitter levels (dopamine, 5HT, etc.) |
| `osc_*` | Oscillation rates (8 frequencies) |
| `lr` | Current learning rate |

### Visualization

Standard TensorBoard or CSV logging:

```bash
# If using TensorBoard (setup in train.py)
tensorboard --logdir runs/

# Or load CSV directly
import pandas as pd
log_df = pd.read_csv("logs/metrics.csv")
log_df[["step", "train_ppl", "mat", "phi"]].plot()
```

---

## OOD Evaluation Loop

Periodically (every N steps), the harness evaluates on held-out WikiText-103-v1:

```python
if step % eval_interval == 0:
    ood_ppl = evaluate_ood(brain, wikitext_val)
    gap_ratio = ood_ppl / train_ppl
    
    # Log and check early-exit criteria
    log_metric("ood_ppl", ood_ppl, step)
    log_metric("gap_ratio", gap_ratio, step)
    
    # Criterion: "OOD PPL not falling over last N evals"
    if should_exit_early(ood_history, tol=0.02, window=4000):
        save_checkpoint()
        exit()
```

**Evidence:** B3 (PCT-30M) reaches best OOD at step 4000, then plateaus. Early exit prevents wasted compute.

---

## Checkpoint Management

### Save Format

```
neuroslm_<preset>_<params>M_<optimizer>[_baseline]_<step>.pt
```

Each `.pt` file contains:
- `state_dict`: weights + biases
- `cfg.__dict__`: hyperparameters (for resume)
- Optionally: `.mem` (narrative memory state) and `.mem.json` (metadata)

### Rotation

After each save, old checkpoints are pruned (keeping last N per stream):

```python
from neuroslm.tools.prune_ckpts import prune_old_checkpoints

prune_old_checkpoints(
    [Path(ckpt_dir)],
    keep=3,              # keep 3 most recent
    use_git=True         # use `git rm` for atomic cleanup
)
```

**Streams:** Grouped by (preset, optimizer, mode). Each rotated independently so AdamW runs don't prune Adafactor checkpoints.

---

## Troubleshooting

### "Model diverges at step ~7-8k"

**Likely cause:** Maturity phasing disabled or phase gates not configured.  
**Fix:** Check `arch.neuro` mechanisms block, ensure `loss_clipping.factor > 0`.

### "OOD perplexity is very high (>2000)"

**Normal at early training.** Models need 5-10k steps to develop usable representations.  
**If persistent:** Check dropout (should be 0.1–0.15), check data mix (should be FineWeb-Edu+OpenHermes for this threshold).

### "Training is slower than expected"

**Check:** Grad accumulation (high GA = fewer updates per step), sequence length (O(seq²) attention), batch size (GPU memory constraint).  
**Profile:** Add `torch.profiler` around forward/backward to find bottlenecks.

### "Checkpoint resume fails with key mismatch"

**Cause:** Config mismatch between saved and loaded model.  
**Fix:** Load with `--strict=False` (ignores extra keys) or check that `arch.neuro` hasn't changed drastically between runs.

---

## References

- **Architecture design:** `docs/architecture.md` §5 (five architectural fixes)
- **Hyperparameter tuning:** `docs/technical_report.md` § 5 (current model state + history)
- **Training stability fixes:** `docs/findings.md` (H7–H10 hypotheses with evidence)
- **CLI reference:** `docs/CLI.md` (command-line usage for train, eval, etc.)

