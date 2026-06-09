"""Diagnose why ``rcc_bowtie_30m_p4`` starts at loss > log(V).

Builds the production config end-to-end (DSL trunk + multi-cortex GPT-2
ensemble + tied fusion head + z-loss + per-sample clip), runs a single
forward pass on a random batch, and dissects WHERE the logit-magnitude
budget is being blown.

Use ``--no-gpt2`` to substitute random-init StubSubCortex backbones so
the diagnostic runs without downloading 600 MB of pretrained weights.
"""
from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

# Make `neuroslm` importable when run from the repo root.
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch
import torch.nn.functional as F


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--no-gpt2", action="store_true",
                        help="Use stub cortices instead of HF GPT-2 weights.")
    parser.add_argument("--batch", type=int, default=2)
    parser.add_argument("--seq-len", type=int, default=128)
    parser.add_argument("--vocab", type=int, default=50257)
    parser.add_argument("--d-sem", type=int, default=512)
    args = parser.parse_args()

    torch.manual_seed(0)

    print("=" * 78)
    print("DIAGNOSE rcc_bowtie_30m_p4 — loss > log(V) investigation")
    print("=" * 78)
    print(f"log(V) for V={args.vocab}: {math.log(args.vocab):.4f} nats")
    print()

    # ── Build the DSL LM trunk ────────────────────────────────────────
    from neuroslm.dsl.nn_lang import DSLLanguageModel
    trunk = DSLLanguageModel(
        vocab=args.vocab,
        d_model=args.d_sem,
        n_layers=8,
        n_heads=8,
        d_head=64,
        d_kv=512,
        d_ff=2048,
        max_seq=args.seq_len,
        dropout=0.0,
    )
    trunk.eval()

    # ── Build the cortex ensemble ─────────────────────────────────────
    from neuroslm.cortex import (
        build_default_ensemble,
        DEFAULT_GPT2_VARIANTS,
        build_gpt2_ensemble,
    )
    if args.no_gpt2:
        print("[ensemble] using stub cortices (random init)")
        ensemble = build_default_ensemble(
            vocab=args.vocab,
            d_model=args.d_sem,
            domains=("math", "code", "chat", "general"),
            lexical_bias_weight=2.0,
            bema_tau=0.5,
        )
    else:
        print("[ensemble] loading GPT-2 family from HF...")
        ensemble = build_gpt2_ensemble(
            d_target=args.d_sem,
            variants=DEFAULT_GPT2_VARIANTS,
            freeze_weights=True,
            lexical_bias_weight=2.0,
            bema_tau=0.5,
        )
    ensemble.eval()

    # ── Build the tied fusion head (mirror of harness logic) ──────────
    import torch.nn as nn
    cortex_lm_head = nn.Linear(args.d_sem, args.vocab, bias=False)
    cortex_lm_head.weight = trunk.embed   # hard tie
    cortex_mix_logit = nn.Parameter(torch.tensor([0.0]))  # fusion_init=0.5
    alpha = float(torch.sigmoid(cortex_mix_logit).item())
    print(f"[fusion] alpha (initial) = {alpha:.4f}")
    print()

    # ── Random batch ──────────────────────────────────────────────────
    ids = torch.randint(0, args.vocab, (args.batch, args.seq_len))
    targets = torch.randint(0, args.vocab, (args.batch, args.seq_len))

    # ── Stage 1: trunk only ───────────────────────────────────────────
    with torch.no_grad():
        lm_logits = trunk(ids)
    print("─── STAGE 1: DSL trunk logits (random init) ─────────────────")
    print(f"  shape={tuple(lm_logits.shape)}")
    print(f"  mean={lm_logits.mean().item():+.4f}  std={lm_logits.std().item():.4f}")
    print(f"  min={lm_logits.min().item():+.4f}   max={lm_logits.max().item():+.4f}")
    ce_trunk = F.cross_entropy(
        lm_logits.reshape(-1, args.vocab), targets.reshape(-1)
    ).item()
    print(f"  CE(trunk_only) = {ce_trunk:.4f}  "
          f"(log(V)={math.log(args.vocab):.4f}, "
          f"excess={ce_trunk - math.log(args.vocab):+.4f})")
    print()

    # ── Stage 2: cortex hidden ────────────────────────────────────────
    with torch.no_grad():
        cortex_h = ensemble(ids)
    print("─── STAGE 2: MultiCortexEnsemble output ─────────────────────")
    print(f"  shape={tuple(cortex_h.shape)}")
    print(f"  mean={cortex_h.mean().item():+.4f}  std={cortex_h.std().item():.4f}")
    print(f"  min={cortex_h.min().item():+.4f}   max={cortex_h.max().item():+.4f}")
    print(f"  per-element abs-mean = {cortex_h.abs().mean().item():.4f}")
    print()

    # Inspect each sub-cortex + projection individually
    print("─── STAGE 2b: per-cortex breakdown ──────────────────────────")
    with torch.no_grad():
        for i, (sub, proj) in enumerate(
            zip(ensemble.sub_cortices, ensemble.projections)
        ):
            h_native = sub(ids)
            h_proj = proj(h_native) if not isinstance(proj, nn.Identity) else h_native
            print(f"  [{i}] {sub.name:18s} d_native={sub.d_native:4d}  "
                  f"native_std={h_native.std().item():.3f}  "
                  f"proj_std={h_proj.std().item():.3f}")
            if hasattr(proj, "weight"):
                print(f"      proj.weight std = {proj.weight.std().item():.6f}  "
                      f"(shape={tuple(proj.weight.shape)})")
    print()

    # ── Stage 3: cortex logits ───────────────────────────────────────
    with torch.no_grad():
        cortex_logits = cortex_lm_head(cortex_h)
    print("─── STAGE 3: cortex_lm_head(cortex_h) = cortex_logits ───────")
    print(f"  shape={tuple(cortex_logits.shape)}")
    print(f"  mean={cortex_logits.mean().item():+.4f}  "
          f"std={cortex_logits.std().item():.4f}")
    print(f"  min={cortex_logits.min().item():+.4f}   "
          f"max={cortex_logits.max().item():+.4f}")
    ce_cortex = F.cross_entropy(
        cortex_logits.reshape(-1, args.vocab), targets.reshape(-1)
    ).item()
    print(f"  CE(cortex_only) = {ce_cortex:.4f}")
    print()

    # ── Stage 4: fused logits ────────────────────────────────────────
    with torch.no_grad():
        final = (1.0 - alpha) * lm_logits + alpha * cortex_logits
    print("─── STAGE 4: final fused logits ─────────────────────────────")
    print(f"  shape={tuple(final.shape)}")
    print(f"  mean={final.mean().item():+.4f}  std={final.std().item():.4f}")
    print(f"  min={final.min().item():+.4f}   max={final.max().item():+.4f}")
    ce_fused = F.cross_entropy(
        final.reshape(-1, args.vocab), targets.reshape(-1)
    ).item()
    z_loss = (torch.logsumexp(final.reshape(-1, args.vocab), dim=-1) ** 2).mean().item()
    print(f"  CE(fused)         = {ce_fused:.4f}")
    print(f"  z_loss(fused)     = {z_loss:.4f}  (* z_w=1e-4 -> {z_loss*1e-4:+.4f})")
    print(f"  loss(fused+z)     = {ce_fused + z_loss*1e-4:.4f}")
    print()

    # ── Verdict ──────────────────────────────────────────────────────
    print("=" * 78)
    print("VERDICT")
    print("=" * 78)
    log_v = math.log(args.vocab)
    if ce_fused > log_v + 0.5:
        excess = ce_fused - log_v
        print(f"  FUSED loss exceeds log(V) by {excess:+.3f} nats.")
        print(f"  Production observed: ~13.84 (excess ~ {13.84-log_v:+.3f})")
        if cortex_logits.std().item() > lm_logits.std().item() * 2:
            print(f"  ⇒ CORTEX logits dominate by std: "
                  f"{cortex_logits.std().item():.2f} vs trunk "
                  f"{lm_logits.std().item():.2f}")
            print(f"  ⇒ Fix: scale-control the cortex output BEFORE the tied lm_head")
            print(f"         (e.g. final LayerNorm on cortex_h, OR zero-init the")
            print(f"          per-cortex projections so cortex_h ≈ 0 at step 0).")
        else:
            print(f"  ⇒ TRUNK logits are the culprit (std={lm_logits.std().item():.2f})")
    else:
        print(f"  CE(fused) = {ce_fused:.3f} is within log(V) = {log_v:.3f}")
        print(f"  ⇒ ROOT CAUSE IS ELSEWHERE (data path? optimizer? loss reduction?)")


if __name__ == "__main__":
    main()
