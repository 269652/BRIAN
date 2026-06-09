"""Empirical diagnosis of catastrophic init loss on rcc_bowtie_30m_p4.

User observation: step-20 loss ≈ 13.84 while ln(50257) = 10.82.
That is 3 nats WORSE than uniform, which means the model is making
confidently wrong predictions at init. Three suspects ranked by prior:

  H1  GPT-2 rogue-dimension anisotropy: GPT-2's residual stream has a
      well-documented "outlier" dimension with magnitude >> 1 even
      after final LayerNorm (Timkey & van Schijndel 2021). A random
      Linear(768 -> d_sem) projection bakes that outlier into the
      first principal direction of sem-space, then the tied
      cortex_lm_head (= lm.embed.T, std=0.02) re-emits it as a
      logit spike on whichever vocab token happens to align with
      that direction. Result: high-confidence wrong predictions.

  H2  Scale mismatch between lm_logits (xavier_uniform head, std ~0.14)
      and cortex_logits (tied head over a non-normalised projection).
      The (1-α)·lm + α·cortex mix at α=0.5 inherits the larger
      cortex variance, so the trunk's small-but-correct signal is
      drowned out by the cortex's larger-but-random signal.

  H3  Tied-head sign bug: cortex_lm_head.weight = embed.weight only
      makes sense if the cortex projection lives in the SAME semantic
      space as the embedding. cortex_proj is randomly initialised at
      training start, so the projection does NOT map cortex features
      into embed-space, defeating the tying assumption.

We probe all three with one forward pass on real GPT-2 (the smallest
sub-cortex used in rcc_bowtie). If any of {cortex_logits.std,
per-token max, cross-entropy vs random labels} blows up, the
hypothesis is confirmed.
"""

from __future__ import annotations

import math
import sys

import torch
import torch.nn as nn
import torch.nn.functional as F

torch.manual_seed(0)
BASELINE_NATS = math.log(50257)  # uniform random predictor
print(f"baseline ln(50257) = {BASELINE_NATS:.4f} nats")
print("=" * 70)


# ---------------------------------------------------------------------------
# Step 1: empirical anisotropy of GPT-2 last_hidden_state
# ---------------------------------------------------------------------------
print("\n[1/4] GPT-2 hidden-state anisotropy probe")
try:
    from transformers import GPT2Model
except Exception as e:  # pragma: no cover - diagnostic only
    print(f"transformers not available: {e}; skipping live GPT-2 probe")
    GPT2Model = None  # type: ignore[assignment]

if GPT2Model is not None:
    gpt2 = GPT2Model.from_pretrained("gpt2").eval()
    B, T = 2, 32
    ids = torch.randint(0, 50257, (B, T))
    with torch.no_grad():
        h = gpt2(input_ids=ids).last_hidden_state  # (B, T, 768)

    per_dim_std = h.std(dim=(0, 1))  # (768,)
    print(f"  overall  : mean={h.mean():.4f} std={h.std():.4f}")
    print(f"  per-dim  : median std={per_dim_std.median():.4f}")
    print(f"  per-dim  : max std={per_dim_std.max():.4f}")
    print(f"  ROGUE    : max/median ratio = {(per_dim_std.max() / per_dim_std.median()):.1f}x")
    top5 = per_dim_std.topk(5)
    print(f"  top-5 std: {[f'{v:.2f}' for v in top5.values.tolist()]}")
    print(f"  top-5 dim: {top5.indices.tolist()}")
    cortex_hidden = h
else:
    # Fall back to synthetic isotropic hidden (best-case)
    cortex_hidden = torch.randn(2, 32, 768)


# ---------------------------------------------------------------------------
# Step 2: cortex_proj scale
# ---------------------------------------------------------------------------
print("\n[2/4] cortex_proj output scale (default nn.Linear init)")
d_sem = 512
vocab = 50257

cortex_proj = nn.Linear(768, d_sem, bias=False)  # default kaiming_uniform
with torch.no_grad():
    sem = cortex_proj(cortex_hidden)  # (B, T, 512)
sem_per_dim = sem.std(dim=(0, 1))
print(f"  sem        : mean={sem.mean():.4f} std={sem.std():.4f}")
print(f"  per-dim std: median={sem_per_dim.median():.4f} max={sem_per_dim.max():.4f}")
print(f"  ratio      : {(sem_per_dim.max()/sem_per_dim.median()):.1f}x  (still anisotropic?)")


# ---------------------------------------------------------------------------
# Step 3: tied cortex_lm_head (= embed.T, std=0.02)
# ---------------------------------------------------------------------------
print("\n[3/4] cortex_lm_head (tied to embed, std=0.02 like the harness)")
embed = nn.Embedding(vocab, d_sem)
nn.init.normal_(embed.weight, std=0.02)
cortex_lm_head = nn.Linear(d_sem, vocab, bias=False)
cortex_lm_head.weight = embed.weight  # tied (like harness._make_cortex_lm_head)
with torch.no_grad():
    cortex_logits = cortex_lm_head(sem)  # (B, T, 50257)
print(f"  cortex_logits        : mean={cortex_logits.mean():.4f} std={cortex_logits.std():.4f}")
print(f"  cortex_logits max    : {cortex_logits.max().item():.2f}")
print(f"  cortex_logits min    : {cortex_logits.min().item():.2f}")
print(f"  per-token max (mean) : {cortex_logits.max(dim=-1).values.mean():.4f}")


# ---------------------------------------------------------------------------
# Step 4: cross-entropy vs random labels (the actual training signal)
# ---------------------------------------------------------------------------
print("\n[4/4] cross-entropy of mixed logits vs RANDOM labels (init step)")

# Trunk path: hidden from DSL LM is roughly N(0,1) after LayerNorm.
# lm_head is xavier_uniform.
lm_head = nn.Linear(d_sem, vocab, bias=False)
nn.init.xavier_uniform_(lm_head.weight)
trunk_hidden = torch.randn_like(sem)
with torch.no_grad():
    lm_logits = lm_head(trunk_hidden)

# Fusion: (1-α)·lm + α·cortex, α=sigmoid(0)=0.5 at init.
alpha = torch.sigmoid(torch.zeros(1)).item()
mixed = (1 - alpha) * lm_logits + alpha * cortex_logits

labels = torch.randint(0, vocab, (lm_logits.shape[0], lm_logits.shape[1]))


def ce(logits: torch.Tensor) -> float:
    return F.cross_entropy(logits.reshape(-1, vocab), labels.reshape(-1)).item()


ce_lm = ce(lm_logits)
ce_cortex = ce(cortex_logits)
ce_mixed = ce(mixed)

print(f"  ln(50257) baseline        : {BASELINE_NATS:.4f}")
print(f"  CE(lm_logits only)        : {ce_lm:.4f}  Δ={ce_lm-BASELINE_NATS:+.4f}")
print(f"  CE(cortex_logits only)    : {ce_cortex:.4f}  Δ={ce_cortex-BASELINE_NATS:+.4f}")
print(f"  CE(mixed, α=0.5)          : {ce_mixed:.4f}  Δ={ce_mixed-BASELINE_NATS:+.4f}")

print("\n" + "=" * 70)
print("VERDICT (current code):")
if ce_mixed > BASELINE_NATS + 0.3:
    print(f"  mixed CE {ce_mixed:.4f} > baseline {BASELINE_NATS:.4f} + 0.3")
    print("  cortex pathway is ACTIVELY HARMFUL at init (vs trunk alone).")
    suspect = "cortex" if ce_cortex > ce_lm + 0.5 else "lm"
    print(f"  dominant source of harm: {suspect}_logits")
else:
    print(f"  mixed CE {ce_mixed:.4f} is near baseline; cortex is benign.")


# ---------------------------------------------------------------------------
# Step 5: candidate fix — LayerNorm between cortex_proj and cortex_lm_head
# ---------------------------------------------------------------------------
print("\n" + "=" * 70)
print("[5/5] CANDIDATE FIX: cortex_pre_head_norm = LayerNorm(d_sem)")
print("       (normalises projected sem features before tied head)")

pre_head_norm = nn.LayerNorm(d_sem)
with torch.no_grad():
    sem_normed = pre_head_norm(sem)
    cortex_logits_fixed = cortex_lm_head(sem_normed)

normed_per_dim = sem_normed.std(dim=(0, 1))
print(f"\n  sem_normed       : mean={sem_normed.mean():.4f} std={sem_normed.std():.4f}")
print(f"  per-dim std      : median={normed_per_dim.median():.4f} max={normed_per_dim.max():.4f}")
print(f"  per-dim ratio    : {(normed_per_dim.max()/normed_per_dim.median()):.2f}x")
print(f"\n  cortex_logits FIX: mean={cortex_logits_fixed.mean():.4f} std={cortex_logits_fixed.std():.4f}")
print(f"  max              : {cortex_logits_fixed.max().item():.2f}")
print(f"  per-token max    : {cortex_logits_fixed.max(dim=-1).values.mean():.4f}")

ce_cortex_fixed = ce(cortex_logits_fixed)
ce_mixed_fixed = ce((1 - alpha) * lm_logits + alpha * cortex_logits_fixed)
print(f"\n  CE(cortex_fixed) : {ce_cortex_fixed:.4f}  Δ={ce_cortex_fixed-BASELINE_NATS:+.4f}")
print(f"  CE(mixed_fixed)  : {ce_mixed_fixed:.4f}  Δ={ce_mixed_fixed-BASELINE_NATS:+.4f}")
print(f"  CE(lm_only)      : {ce_lm:.4f}  Δ={ce_lm-BASELINE_NATS:+.4f}  (reference)")

print("\n" + "=" * 70)
print("FIX VERDICT:")
gap_before = ce_mixed - ce_lm
gap_after = ce_mixed_fixed - ce_lm
print(f"  cortex-induced excess CE  before fix: {gap_before:+.4f} nats")
print(f"  cortex-induced excess CE  after  fix: {gap_after:+.4f} nats")
if abs(gap_after) < 0.05:
    print("  ✅ FIX RESOLVES the catastrophic-loss bug at init")
    sys.exit(0)
else:
    print("  ❌ FIX is INCOMPLETE — additional measures required")
    sys.exit(1)
