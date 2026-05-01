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

import sys

# Extra safety: assert Python version early with a helpful message.
if getattr(sys, 'version_info', (0,)) < (3, 8):
    raise RuntimeError("neuroslm training requires Python 3.8+. Please run using your venv python or 'py -3'.")

from .config import PRESETS
from .tokenizer import Tokenizer
from .brain import Brain
from .data import batch_iterator
from torch.nn.utils import stateless


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
    ap.add_argument("--meta", action="store_true",
                    help="Enable one-step meta-training of the learned optimizer")
    ap.add_argument("--meta_lr", type=float, default=1e-4,
                    help="Learning rate for the meta-optimizer (updates learned optimizer)")
    ap.add_argument("--mode", default="text", choices=["text", "chat", "mix"],
                    help="text=narrative, chat=multi-turn dialogue, "
                         "mix=interleave (recommended once base LM is decent)")
    ap.add_argument("--chat_ratio", type=float, default=0.75,
                    help="(mix only) fraction of windows from chat datasets")
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[train] device={device}", flush=True)

    # Meta-mode requires higher-order gradients through the inner update.
    # Some CPU attention kernels (flash attention variants) do not implement
    # higher-order derivatives on CPU. Detect this early and emit a helpful
    # error to avoid confusing low-level runtime failures.
    if args.meta and device != "cuda":
        raise RuntimeError(
            "Meta-training (--meta) requires a CUDA-capable device because some CPU\n"
            "attention kernels lack support for higher-order gradients.\n"
            "Please run with --device cuda on a machine with an NVIDIA GPU (or on Colab),\n"
            "or disable --meta for CPU runs."
        )

    cfg = PRESETS[args.preset]()
    tok = Tokenizer()
    cfg.vocab_size = tok.vocab_size
    ctx_len = args.ctx or cfg.lang_ctx
    assert ctx_len <= cfg.lang_ctx

    brain = Brain(cfg).to(device)
    n_params = brain.num_parameters()
    print(f"[train] model params: {n_params/1e6:.2f}M (preset={args.preset})", flush=True)

    # Separate optimizers: model optimizer updates model params; meta optimizer
    # updates the learned optimizer (`brain.learned_opt`). This avoids double
    # updating learned_opt with the main optimizer.
    named = list(brain.named_parameters())
    model_params = [p for n, p in named if not n.startswith('learned_opt.')]
    optim = AdamW(model_params, lr=cfg.lr,
                  weight_decay=cfg.weight_decay, betas=(0.9, 0.95))

    meta_opt = None
    if args.meta:
        # meta optimizer updates the learned optimizer + geometry adapters
        meta_params = list(brain.learned_opt.parameters())
        # Include geometry adapter parameters so the neural topology is meta-learned
        for name, p in brain.language.named_parameters():
            if 'adapter' in name:
                meta_params.append(p)
        meta_opt = AdamW(meta_params, lr=args.meta_lr)

    start_step = 0
    if args.resume and Path(args.resume).exists():
        ckpt = torch.load(args.resume, map_location=device)
        brain.load_state_dict(ckpt["model"])
        optim.load_state_dict(ckpt["optim"])
        start_step = ckpt["step"]
        if "gene_pool" in ckpt:
            from .genome import GenePool
            brain.gene_pool = GenePool.from_state(ckpt["gene_pool"])
        print(f"[train] resumed from {args.resume} @ step {start_step}", flush=True)
    elif args.transfer and Path(args.transfer).exists():
        ckpt = torch.load(args.transfer, map_location=device)
        brain.load_partial(ckpt["model"])
        if "gene_pool" in ckpt:
            from .genome import GenePool
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


    # --- Memory system integration ---
        # 1. Record episodic memory for this batch (use first sample for simplicity)
        content = tok.decode(ids[0].tolist())
        content_vec = ids[0].float().cpu().numpy()  # Placeholder: use embedding for real system
        nt_state = brain.transmitters.vector()[0].cpu().numpy()
        emotion = None  # Placeholder: can be inferred from NT or model output
        tags = []
        context = {'self': True}  # Placeholder: can be set based on batch/task
        brain.record_episode(content, content_vec, nt_state, emotion, tags, context)

        # 2. Forward pass
        out = brain.forward_lm(ids, targets)
        loss = out["loss"]

        # If meta-training is enabled, perform a one-step differentiable unroll
        if args.meta:
            # Compute comprehension delta: how much LM loss improved vs last step
            current_lm = float(out["lm_loss"].item())
            if not hasattr(main, '_prev_lm'):
                main._prev_lm = current_lm
            comp_delta = main._prev_lm - current_lm  # positive = improvement
            main._prev_lm = current_lm

            # get a meta batch
            try:
                meta_batch = next(it)
            except StopIteration:
                it = batch_iterator(tok, ctx_len, args.batch_size,
                                    seed=args.seed + step,
                                    mode=args.mode, chat_ratio=args.chat_ratio)
                meta_batch = next(it)
            meta_batch = meta_batch.to(device)
            meta_ids, meta_targets = meta_batch[:, :-1], meta_batch[:, 1:].contiguous()

            # Meta-learn language module parameters (including geometry adapters)
            model_named = list(brain.language.named_parameters())
            model_params = [p for _, p in model_named]
            grads = torch.autograd.grad(loss, model_params, create_graph=True, allow_unused=True)

            # neuromodulatory vector (DA, NE, 5HT, ACh)
            nm = brain.transmitters.vector().mean(dim=0)[:4].to(device)

            # form virtual updated parameters for the language module
            virtual_map = {}
            for (name, p), g in zip(model_named, grads):
                if g is None:
                    g = torch.zeros_like(p)
                mult = brain.learned_opt(g, p, nm,
                                         comprehension_delta=comp_delta,
                                         param_name=name)
                transformed = g * mult
                virtual = p - cfg.lr * transformed
                virtual_map[name] = virtual

            # Evaluate meta-loss under virtual language params
            meta_out = stateless.functional_call(brain.language, virtual_map, (meta_ids,))
            logits_meta, _, _ = meta_out
            meta_loss = torch.nn.functional.cross_entropy(
                logits_meta.reshape(-1, logits_meta.size(-1)),
                meta_targets.reshape(-1), ignore_index=-100)

            # Update meta-parameters (learned optimizer + geometry adapters)
            meta_opt.zero_grad(set_to_none=True)
            meta_loss.backward()
            meta_opt.step()

            # Reset learned optimizer hidden states after meta step
            brain.learned_opt.reset_state()

            # Apply transformed gradients for real model update
            optim.zero_grad(set_to_none=True)
            for (name, p), g in zip(model_named, grads):
                if g is None:
                    g = torch.zeros_like(p)
                mult = brain.learned_opt(g, p, nm,
                                         comprehension_delta=comp_delta,
                                         param_name=name)
                transformed = (g * mult).detach()
                p.grad = transformed
            # gradient clip and step (only on language params we modified)
            gnorm = torch.nn.utils.clip_grad_norm_(model_params, cfg.grad_clip)
            optim.step()
        else:
            optim.zero_grad(set_to_none=True)
            loss.backward()
            gnorm = torch.nn.utils.clip_grad_norm_(brain.parameters(), cfg.grad_clip)
            optim.step()

        # 3. Tag memory with reward/insight (mesolimbic)
        reward = float(out["learning_gain"][0].item()) if "learning_gain" in out else 0.0
        insight = None  # Placeholder: can be set if new pattern detected
        brain.tag_memory(len(brain.episodic.buffer)-1, reward, insight)

        # 4. Consolidate and update narratives every 500 steps
        if (step + 1) % 500 == 0:
            brain.consolidate_memory()
            brain.update_narratives()

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
