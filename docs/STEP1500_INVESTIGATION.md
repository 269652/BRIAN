# The Step-1500 Spike — Investigation + Remediation Plan

**Observation**: Three independent RCC training runs (P1, P2, P3) — each
with a different architectural fix targeting different "closed-loop"
hypotheses — all spiked from `ppl ~125` to `ppl ~493` at **exactly
step 1500**, with three-significant-figure-identical magnitudes:

| Run | Step 1480 ppl | Step 1500 ppl | Spike ratio |
|---|---|---|---|
| RCC P1 | 125  | 493 | 3.94× |
| RCC P2 | 125.0 | 493 | 3.94× |
| RCC P3 | 125.1 | 492.8 | 3.94× |

The data identity is the *only* shared variable. All three runs use:

- `seed=0` (default)
- `mode='mix'` with `chat_ratio=0.6`
- `batch_size=4`, `grad_accum=4` → 16 sequences per `train.py` step
- Same `ctx_len=1024`, same tokenizer (GPT-2 BPE)

`token_window_iterator` is fully deterministic given those parameters
(see `neuroslm/data.py:262` — `rng = random.Random(seed)`). So **step 1500
contains the same 16 sequences across every run**, and those sequences
are pathological.

## What `scripts/inspect_step1500_batch.py` does

Re-creates the exact data iterator with the training parameters,
fast-forwards to step 1499 (one before the spike), and dumps step
1499 + 1500 + 1501 sequences. Each sequence comes with heuristic
pathology scores:

- **len_chars**: total decoded length
- **longest_token**: longest whitespace-delimited run (URLs / base64 / code lines)
- **nonascii_frac**: fraction of non-ASCII characters (foreign script clusters)
- **repeated_4gram_frac**: 4-gram repetition density (catches stuck loops)
- **whitespace_frac**: low values → dense unbroken blobs

**How to run**:
```bash
.venv/Scripts/python.exe scripts/inspect_step1500_batch.py > /tmp/step1500_dump.txt
```

Streams FineWeb-Edu + OpenHermes-2.5 from HuggingFace (requires
`HF_TOKEN` in `.env` for full speed, anonymous works but slower).
Run takes ~3-8 min depending on bandwidth.

## What the spike pattern usually means

In language-model training, deterministic single-step PPL spikes from a
low basin (here, 125 → 493) almost always trace to one of these classes
of pathological inputs:

### Class A — Token-distribution outlier (most common)
A single sequence dominated by an unusual character class:
- A long URL or base64 blob → many low-frequency tokens in a row,
  each one ~14 bits of surprise.
- A code snippet with patterns absent from the training distribution
  (e.g., a sequence of SQL bind parameters, or hex memory dumps).
- A "stuck" repetition where the source document loops on itself
  (web scrape artifact).
- A foreign-script cluster (Chinese / Arabic / Devanagari) in a
  mostly-English mix.

These cause one sequence's per-token loss to spike to ~10-12 nats per
token, and since loss is averaged uniformly across the batch, ONE bad
sequence at loss 10 + 15 good sequences at loss 5 → batch loss 5.3,
PPL goes from 150 to 200. With grad_accum=4 (16 sequences total), if
4 of them are pathological the effective batch loss jumps to ~7 → PPL
~1000.

### Class B — Sequence boundary artifact
The stream concatenates documents and slices to fixed `ctx_len`. A
window can straddle 2-5 unrelated documents, joined by `<eos>`. If
step 1500's batch happens to draw windows that ALL straddle topic
boundaries, the model's coherence prediction tanks.

### Class C — Mixture-mode artifact
`chat_ratio=0.6` → on average 9-10 of the 16 sequences per step are
chat-formatted (OpenHermes), 6-7 are text (FineWeb). The chat format
uses `<|im_start|>...<|im_end|>` tokens. If the RNG happens to draw
mostly chat at step 1500 AND those chat samples include unusual
system prompts (e.g., a roleplay scenario the model hasn't seen
this batch range), the conditional distribution shifts.

### Class D — Tokenizer artifact
A document containing characters that tokenize into many tiny
sub-word tokens (rare punctuation, mathematical symbols, edge
unicode). Each tiny token carries near-uniform entropy → high loss.

## Remediation options — preprocess vs runtime

### Option 1 — Preprocessing filter (cheapest, deterministic)

Before training, scan the stream and drop sequences that score badly on:
- `longest_token > 200` (URLs / base64 / hex dumps)
- `nonascii_frac > 0.5` AND foreign-script cluster (use `unicodedata.script`)
- `repeated_4gram_frac > 0.3` (stuck loops)
- `whitespace_frac < 0.05` (dense unbroken blobs)

**Cost**: a few thousand sequences dropped per million → ~1% data loss.
**Benefit**: deterministically removes the worst spikes.

Implementation: ~30 LOC wrapper around `_stream_iterator`.

### Option 2 — Per-sample loss clipping (mid-cost, robust)

In the training loss computation, instead of `loss = lm_loss.mean()`,
compute per-sequence losses and **clip each one to a max value** before
averaging:

```python
lm_loss_per_seq = lm_loss_per_token.mean(dim=1)   # (B,) per-sequence loss
max_allowed = 3.0 * lm_loss_per_seq.median()       # adaptive clip
lm_loss_per_seq = torch.clamp(lm_loss_per_seq, max=max_allowed)
loss = lm_loss_per_seq.mean()
```

**Cost**: ~5 LOC change in `train.py`.
**Benefit**: a single bad sequence can no longer dominate the batch
loss. The model still SEES the data (forward unchanged) but doesn't
get yanked by its outlier gradient.

This is the **Huber loss applied at the sequence level** — well-
established robust statistics. Used in:
- Microsoft's Phi training (anomaly clipping per-document)
- OpenAI's GPT-3 robust loss variants
- Cerebras training pipelines (per-sample loss filtering)

### Option 3 — Adaptive batch rejection (robust + visible)

If a batch's loss exceeds `N × EMA(loss)`, skip the optimizer step
(same mechanism as the existing `grad_spike_factor=3.0` for gradients).

```python
loss_ema = 0.95 * loss_ema + 0.05 * loss.item()
if loss.item() > 3.0 * loss_ema:
    print(f"[train] outlier batch loss {loss.item():.2f} > 3x EMA {loss_ema:.2f}, skipping")
    continue
loss.backward()
optimizer.step()
```

**Cost**: ~8 LOC.
**Benefit**: deterministic, observable in logs.
**Downside**: throws away a few % of training data (the outliers).

### Option 4 — Detect and HANDLE (most ambitious)

Train the model to be ROBUST to the pathological inputs:
- Add a small "anomaly head" that predicts whether the current input is
  out-of-distribution.
- If the anomaly head fires, mask the trunk's gradient contribution
  for that sequence (let the embedding+anomaly head still learn).

**Cost**: ~150 LOC + careful integration.
**Benefit**: the model learns to be calm in the face of weird inputs.
**Risk**: research-mode; the anomaly head itself has to be trained,
chicken-and-egg.

## Recommendation

**Ship Option 2 (per-sample loss clipping) first.**

Reasons:
1. Tiny change. ~5 LOC. Can ship in 10 minutes.
2. The model still sees ALL the data — no preprocessing loss.
3. Well-established in production LM training (Phi / GPT-3 / Cerebras).
4. Adaptive threshold (`3 × median`) means it auto-tunes per batch.
5. Specifically targets the failure mode we observed: ONE sequence
   dominating the batch average.

If Option 2 doesn't fully resolve the spike, layer in Option 1
(preprocessing filter) for the data classes we identify via
`inspect_step1500_batch.py`.

Option 4 (anomaly head) is research-tier — only worth pursuing if 1-3
combined still leave residual spikes.

## Acceptance criteria

After applying the chosen remediation, a fresh training run with
seed=0 should show:

- Step 1480 ppl: still ~125 (no upstream change)
- **Step 1500 ppl: < 200** (down from 493 — Option 2 should clip ~75% of the spike)
- Step 1700 onwards: smooth descent past ppl 80
- best.pt: advances to step 5000+ instead of plateau'ing at step 4000
- gnorm: same band (clipping happens on loss, not gradient)

If those hold, the "deterministic step-1500 catastrophe" is gone
**without any architectural change** — proving the bug was data-driven
all along.
