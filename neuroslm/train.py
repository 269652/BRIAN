"""Training loop for NeuroSLM.

Streams a Phi-style open dataset (Cosmopedia by default) and runs the brain's
multi-objective forward pass.

Usage:
    python -m neuroslm.train --preset small --steps 2000 --batch_size 4
"""
from __future__ import annotations
import argparse
import math
import os
import time
from pathlib import Path
import torch
from torch.optim import AdamW

from .config import PRESETS
from .tokenizer import Tokenizer
from .brain import Brain
from .data import batch_iterator


def cosine_lr(step: int, warmup: int, total: int, peak: float) -> float:
    if step < warmup:
        return peak * step / max(1, warmup)
    progress = (step - warmup) / max(1, total - warmup)
    return peak * 0.5 * (1.0 + math.cos(math.pi * min(progress, 1.0)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--preset", default="small", choices=list(PRESETS.keys()))
    ap.add_argument("--steps", type=int, default=2000)
    ap.add_argument("--batch_size", type=int, default=4)
    ap.add_argument("--ctx", type=int, default=None,
                    help="override context length (must be <= cfg.lang_ctx)")
    ap.add_argument("--ckpt_dir", default="checkpoints")
    ap.add_argument("--save_every", type=int, default=500)
    ap.add_argument("--log_every", type=int, default=10)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--resume", default=None)
    ap.add_argument("--transfer", default=None,
                    help="Load only matching tensors from a previous checkpoint "
                         "(use when architecture changed).")
    ap.add_argument("--device", default=None)
    ap.add_argument("--mode", default="text", choices=["text", "chat", "mix"],
                    help="text=narrative, chat=multi-turn dialogue, "
                         "mix=interleave (recommended once base LM is decent)")
    ap.add_argument("--chat_ratio", type=float, default=0.75,
                    help="(mix only) fraction of windows from chat datasets")
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[train] device={device}", flush=True)

    cfg = PRESETS[args.preset]()
    tok = Tokenizer()
    cfg.vocab_size = tok.vocab_size
    ctx_len = args.ctx or cfg.lang_ctx
    assert ctx_len <= cfg.lang_ctx

    brain = Brain(cfg).to(device)
    n_params = brain.num_parameters()
    print(f"[train] model params: {n_params/1e6:.2f}M (preset={args.preset})", flush=True)

    optim = AdamW(brain.parameters(), lr=cfg.lr,
                  weight_decay=cfg.weight_decay, betas=(0.9, 0.95))

    start_step = 0
    if args.resume and Path(args.resume).exists():
        ckpt = torch.load(args.resume, map_location=device)
        brain.load_state_dict(ckpt["model"])
        optim.load_state_dict(ckpt["optim"])
        start_step = ckpt["step"]
        if "gene_pool" in ckpt:
            from .dna import GenePool
            brain.gene_pool = GenePool.from_state(ckpt["gene_pool"])
        print(f"[train] resumed from {args.resume} @ step {start_step}", flush=True)
    elif args.transfer and Path(args.transfer).exists():
        ckpt = torch.load(args.transfer, map_location=device)
        brain.load_partial(ckpt["model"])
        if "gene_pool" in ckpt:
            from .dna import GenePool
            brain.gene_pool = GenePool.from_state(ckpt["gene_pool"])
        print(f"[train] transferred matching tensors from {args.transfer}", flush=True)

    Path(args.ckpt_dir).mkdir(parents=True, exist_ok=True)

    print(f"[train] starting; ctx={ctx_len}, batch={args.batch_size}, "
        f"steps={args.steps}, mode={args.mode}"
        + (f" (chat_ratio={args.chat_ratio})" if args.mode == "mix" else ""), flush=True)
    it = batch_iterator(tok, ctx_len, args.batch_size, seed=args.seed,
                        mode=args.mode, chat_ratio=args.chat_ratio)
    t0 = time.time()
    running_loss = 0.0
    running_lm = 0.0
    n_obs = 0

    for step in range(start_step, args.steps):
        try:
            batch = next(it)
        except StopIteration:
            print("[train] dataset exhausted; restarting iterator", flush=True)
            it = batch_iterator(tok, ctx_len, args.batch_size,
                                seed=args.seed + step,
                                mode=args.mode, chat_ratio=args.chat_ratio)
            batch = next(it)

        batch = batch.to(device)
        ids, targets = batch[:, :-1], batch[:, 1:].contiguous()

        # LR schedule
        for pg in optim.param_groups:
            pg["lr"] = cosine_lr(step, cfg.warmup_steps, args.steps, cfg.lr)

        out = brain.forward_lm(ids, targets)
        loss = out["loss"]
        optim.zero_grad(set_to_none=True)
        loss.backward()
        gnorm = torch.nn.utils.clip_grad_norm_(brain.parameters(), cfg.grad_clip)
        optim.step()

        # Slow homeostatic regulation of NT baselines & gains.
        brain.homeostasis.observe(brain.transmitters,
                                  float(out["lm_loss"].item()), float(gnorm))

        running_loss += float(loss.item())
        running_lm += float(out["lm_loss"].item())
        n_obs += 1

        if (step + 1) % args.log_every == 0:
            dt = time.time() - t0
            avg = running_loss / n_obs
            avg_lm = running_lm / n_obs
            ppl = math.exp(min(avg_lm, 20))
            tok_per_s = args.log_every * args.batch_size * ctx_len / max(dt, 1e-3)
            nt_str = " ".join(f"{k}={v:.2f}" for k, v in (brain.last_nt or {}).items())
            lg = float(brain.last_learning_gain.mean()) if brain.last_learning_gain is not None else 0.0
            print(f"step {step+1:5d} | loss {avg:.4f} | lm {avg_lm:.4f} "
                  f"| ppl {ppl:.1f} | lr {optim.param_groups[0]['lr']:.2e} "
                  f"| {tok_per_s:.0f} tok/s | mesoLG {lg:.2f} | NT[{nt_str}]", flush=True)
            running_loss = running_lm = 0.0
            n_obs = 0
            t0 = time.time()

        if (step + 1) % args.save_every == 0 or (step + 1) == args.steps:
            tag = "" if args.mode == "text" else f"_{args.mode}"
            path = Path(args.ckpt_dir) / f"neuroslm_{args.preset}{tag}_{step+1}.pt"
            torch.save({
                "model": brain.state_dict(),
                "optim": optim.state_dict(),
                "cfg": cfg.__dict__,
                "step": step + 1,
                "preset": args.preset,
                "gene_pool": brain.gene_pool.state(),
                "trophic_stats": brain.trophic.stats(),
            }, path)
            print(f"[train] saved {path} | genome={brain.gene_pool.active().id} "
                  f"gen={brain.gene_pool.active().generation} | "
                  f"trophic={brain.trophic.stats()}", flush=True)

    print("[train] done.", flush=True)


if __name__ == "__main__":
    main()
