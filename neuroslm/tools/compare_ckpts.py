"""Compare baseline vs full-model checkpoints on held-out LM loss.

The ablation comparison wants a single number per model — the mean
language-modelling loss on a fixed held-out slice — so baseline (vanilla
param-matched transformer) and full (bio modules) can be ranked directly.

Each checkpoint is loaded with its OWN saved cfg (so architecture matches
the weights exactly — see benchmarks.py §0.18), then evaluated on N fixed
batches drawn with a fixed seed (identical data for both models).

Usage (Colab cell):

    !cd /content/brian && python -m neuroslm.tools.compare_ckpts \
        --full     /content/brian/lfs_checkpoints/neuroslm_large_107M_adamw_mix_1000.pt \
        --baseline /content/checkpoints_baseline/neuroslm_large_107M_adamw_baseline_mix_1000.pt \
        --eval_batches 20 --device cuda --mode mix --chat_ratio 0.6

Either --full or --baseline may be omitted; whichever is present is
evaluated. Missing files are reported, not fatal.
"""
from __future__ import annotations
import argparse
import glob
import math
import os
import re
from pathlib import Path

import torch

_STEP_RE = re.compile(r"_(\d+)\.pt$")


def _find_latest(directory: str, preset: str, baseline: bool) -> str | None:
    """Newest (highest-step) checkpoint in `directory` for this preset.

    baseline=True  → only `*_baseline_*` files (vanilla param-matched).
    baseline=False → only NON-baseline files (full bio model).
    `_latest.pt` files are considered only if no step-numbered file exists.
    """
    if not directory or not os.path.isdir(directory):
        return None
    cands = []
    for p in glob.glob(os.path.join(directory, f"neuroslm_{preset}*.pt")):
        name = os.path.basename(p)
        is_baseline = "_baseline" in name
        if is_baseline != baseline:
            continue
        m = _STEP_RE.search(name)
        step = int(m.group(1)) if m else -1   # -1 → _latest.pt, lowest rank
        cands.append((step, os.path.getmtime(p), p))
    if not cands:
        return None
    # Highest step wins; mtime breaks ties (and ranks _latest.pt among itself).
    cands.sort(key=lambda x: (x[0], x[1]))
    return cands[-1][2]


def _load_brain(ckpt_path: str, device: str, fallback_preset: str | None):
    """Rebuild a Brain from the checkpoint's saved cfg and load weights.

    Returns (brain, info_dict) or (None, info_dict) on failure.
    """
    from ..config import PRESETS, BrainConfig
    from ..brain import Brain
    from ..tokenizer import Tokenizer

    info = {"path": ckpt_path, "ok": False}
    if not Path(ckpt_path).exists():
        info["error"] = "file not found"
        return None, info

    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    tok = Tokenizer()

    if "cfg" in ckpt and isinstance(ckpt["cfg"], dict):
        cfg = BrainConfig()
        valid = set(cfg.__dict__.keys())
        for k, v in ckpt["cfg"].items():
            if k in valid:
                setattr(cfg, k, v)
    else:
        cfg = PRESETS[fallback_preset or "large"]()
    cfg.vocab_size = tok.vocab_size

    brain = Brain(cfg).to(device)
    # Reuse the train-time adapter-rank-aware loader so BDNF-grown
    # checkpoints don't crash on shape mismatch.
    try:
        from ..train import _load_compatible
        _load_compatible(brain, ckpt["model"], label=os.path.basename(ckpt_path))
    except Exception:
        brain.load_state_dict(ckpt["model"], strict=False)
    brain.eval()

    info.update(
        ok=True,
        baseline=bool(getattr(cfg, "baseline", False)),
        params=sum(p.numel() for p in brain.parameters()),
        step=ckpt.get("step", "?"),
        cfg=cfg,
        tok=tok,
    )
    return brain, info


@torch.no_grad()
def _eval_loss(brain, tok, cfg, device: str, n_batches: int,
               mode: str, chat_ratio: float, seed: int = 1234) -> float:
    """Mean LM loss over `n_batches` fixed held-out windows."""
    from ..data import batch_iterator

    ctx = cfg.lang_ctx
    it = batch_iterator(tok, ctx, batch_size=1, seed=seed,
                        mode=mode, chat_ratio=chat_ratio)
    total, n = 0.0, 0
    for _ in range(n_batches):
        batch = next(it).to(device)
        ids, tgt = batch[:, :-1], batch[:, 1:].contiguous()
        out = brain.forward_lm(ids, tgt)
        lm = out.get("lm_loss", out.get("loss"))
        total += float(lm.item())
        n += 1
    return total / max(1, n)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--full", default=None, help="Explicit path to full-model .pt")
    ap.add_argument("--baseline", default=None, help="Explicit path to baseline .pt")
    ap.add_argument("--full_dir", default=None,
                    help="Directory to auto-pick the NEWEST full checkpoint from "
                         "(highest step). Overridden by --full if given.")
    ap.add_argument("--baseline_dir", default=None,
                    help="Directory to auto-pick the NEWEST baseline checkpoint "
                         "from. Overridden by --baseline if given.")
    ap.add_argument("--eval_batches", type=int, default=20)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--mode", default="mix", choices=["text", "chat", "mix"])
    ap.add_argument("--chat_ratio", type=float, default=0.6)
    ap.add_argument("--seed", type=int, default=1234,
                    help="Fixed eval seed — identical data for both models.")
    ap.add_argument("--preset", default="large",
                    help="Preset used for auto-discovery + cfg fallback.")
    args = ap.parse_args()

    # Auto-discover newest checkpoints when explicit paths aren't given.
    if not args.full and args.full_dir:
        args.full = _find_latest(args.full_dir, args.preset, baseline=False)
        if args.full:
            print(f"[compare] auto-selected FULL: {args.full}", flush=True)
    if not args.baseline and args.baseline_dir:
        args.baseline = _find_latest(args.baseline_dir, args.preset, baseline=True)
        if args.baseline:
            print(f"[compare] auto-selected BASELINE: {args.baseline}", flush=True)

    if not args.full and not args.baseline:
        ap.error("provide at least one of --full / --baseline / --full_dir / "
                 "--baseline_dir (no matching checkpoint found)")

    results = {}
    for label, path in (("FULL", args.full), ("BASELINE", args.baseline)):
        if not path:
            continue
        print(f"\n[compare] loading {label}: {path}", flush=True)
        brain, info = _load_brain(path, args.device, args.preset)
        if brain is None:
            print(f"[compare] {label}: {info.get('error', 'load failed')}",
                  flush=True)
            results[label] = info
            continue
        print(f"[compare] {label}: {info['params']/1e6:.2f}M params, "
              f"step {info['step']}, baseline={info['baseline']} — "
              f"evaluating {args.eval_batches} batches ...", flush=True)
        loss = _eval_loss(brain, info["tok"], info["cfg"], args.device,
                          args.eval_batches, args.mode, args.chat_ratio,
                          seed=args.seed)
        info["loss"] = loss
        info["ppl"] = math.exp(min(loss, 20))
        results[label] = info
        # Free before loading the next model.
        del brain
        if args.device == "cuda":
            torch.cuda.empty_cache()

    # ── Report ──
    print("\n" + "=" * 62)
    print("  ABLATION COMPARISON — held-out LM loss "
          f"({args.eval_batches} batches, seed {args.seed})")
    print("=" * 62)
    print(f"  {'model':10s} {'params':>10s} {'step':>8s} {'loss':>9s} {'ppl':>10s}")
    print("  " + "-" * 58)
    for label in ("BASELINE", "FULL"):
        r = results.get(label)
        if not r or not r.get("ok") or "loss" not in r:
            print(f"  {label:10s} {'—':>10s} {'—':>8s} "
                  f"{'(no ckpt)':>9s} {'—':>10s}")
            continue
        print(f"  {label:10s} {r['params']/1e6:>9.1f}M {str(r['step']):>8s} "
              f"{r['loss']:>9.4f} {r['ppl']:>10.1f}")
    print("  " + "-" * 58)

    b = results.get("BASELINE", {})
    f = results.get("FULL", {})
    if b.get("loss") is not None and f.get("loss") is not None:
        delta = b["loss"] - f["loss"]
        if delta > 0:
            print(f"  ✓ FULL beats BASELINE by {delta:.4f} nats "
                  f"→ bio modules help.")
        else:
            print(f"  ✗ BASELINE beats FULL by {-delta:.4f} nats "
                  f"→ tune loss weights / train longer.")
    print("=" * 62)


if __name__ == "__main__":
    main()
