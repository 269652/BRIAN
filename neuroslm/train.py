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
from contextlib import nullcontext
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
from torch.func import functional_call


def cosine_lr(step: int, warmup: int, total: int, peak: float) -> float:
    if step < warmup:
        return peak * step / max(1, warmup)
    progress = (step - warmup) / max(1, total - warmup)
    return peak * 0.5 * (1.0 + math.cos(math.pi * min(progress, 1.0)))


def push_checkpoint_to_lfs(ckpt_path: str, repo_root: str | None = None):
    """Copy a checkpoint to lfs_checkpoints/ and push via Git LFS."""
    try:
        import shutil, subprocess
        if repo_root is None:
            repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        lfs_dir = os.path.join(repo_root, "lfs_checkpoints")
        os.makedirs(lfs_dir, exist_ok=True)
        basename = os.path.basename(ckpt_path)
        dest = os.path.join(lfs_dir, basename)
        shutil.copy2(ckpt_path, dest)
        # Also copy .mem file if it exists
        mem_src = ckpt_path.replace('.pt', '.mem')
        if os.path.exists(mem_src):
            shutil.copy2(mem_src, os.path.join(lfs_dir, os.path.basename(mem_src)))
        subprocess.run(["git", "add", "lfs_checkpoints/"], cwd=repo_root,
                       capture_output=True, timeout=30)
        subprocess.run(["git", "commit", "-m", f"chkpt: {basename}"],
                       cwd=repo_root, capture_output=True, timeout=30)
        subprocess.run(["git", "push"], cwd=repo_root,
                       capture_output=True, timeout=120)
        print(f"[train] ✓ pushed {basename} to Git LFS", flush=True)
    except Exception as e:
        print(f"[train] ⚠ LFS push failed: {e}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--preset", default="xl", choices=list(PRESETS.keys()))
    ap.add_argument("--steps", type=int, default=100000)
    ap.add_argument("--batch_size", type=int, default=4)
    ap.add_argument("--ctx", type=int, default=None,
                    help="override context length (must be <= cfg.lang_ctx)")
    ap.add_argument("--ckpt_dir", default="checkpoints")
    ap.add_argument("--save_every", type=int, default=500)
    ap.add_argument("--log_every", type=int, default=10)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--resume", default=None,
                    help="Path to checkpoint, or 'latest' to auto-find the most recent.")
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
    ap.add_argument("--baseline", action="store_true",
                    help="Train vanilla transformer only (no bio modules) for ablation")
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
    if args.baseline:
        cfg.baseline = True
    ctx_len = args.ctx or cfg.lang_ctx
    assert ctx_len <= cfg.lang_ctx

    brain = Brain(cfg).to(device)
    n_params = brain.num_parameters()
    mode_label = "BASELINE (vanilla transformer)" if cfg.baseline else "FULL (bio modules)"
    print(f"[train] {mode_label} | params: {n_params/1e6:.2f}M (preset={args.preset})", flush=True)

    # AMP (mixed precision) — halves memory for activations on CUDA
    use_amp = (device == "cuda")
    scaler = torch.amp.GradScaler('cuda', enabled=use_amp)
    amp_ctx = lambda: torch.amp.autocast('cuda', enabled=use_amp)

    # Separate optimizers: model optimizer updates model params; meta optimizer
    # updates the learned optimizer (`brain.learned_opt`). This avoids double
    # updating learned_opt with the main optimizer.
    named = list(brain.named_parameters())
    model_params = [p for n, p in named if not n.startswith('learned_opt.')]
    optim = AdamW(model_params, lr=cfg.lr,
                  weight_decay=cfg.weight_decay, betas=(0.9, 0.95))

    meta_opt = None
    if args.meta and not cfg.baseline:
        # meta optimizer updates the learned optimizer + geometry adapters
        meta_params = list(brain.learned_opt.parameters())
        # Include geometry adapter parameters so the neural topology is meta-learned
        for name, p in brain.language.named_parameters():
            if 'adapter' in name:
                meta_params.append(p)
        meta_opt = AdamW(meta_params, lr=args.meta_lr)

    start_step = 0
    if args.resume:
        resume_path = args.resume
        if resume_path == "latest":
            # Auto-find the most recent checkpoint
            import glob as _glob
            candidates = sorted(
                _glob.glob(os.path.join(args.ckpt_dir, "*.pt")),
                key=lambda f: os.path.getmtime(f))
            if not candidates:
                candidates = sorted(
                    _glob.glob(os.path.join(
                        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                        "lfs_checkpoints", "*.pt")),
                    key=lambda f: os.path.getmtime(f))
            if candidates:
                resume_path = candidates[-1]
                print(f"[train] auto-found latest checkpoint: {resume_path}", flush=True)
            else:
                print("[train] no checkpoint found, training from scratch", flush=True)
                resume_path = None
        if resume_path and Path(resume_path).exists():
            ckpt = torch.load(resume_path, map_location=device, weights_only=False)
            brain.load_state_dict(ckpt["model"], strict=False)
            if "optim" in ckpt:
                try:
                    optim.load_state_dict(ckpt["optim"])
                except Exception as e:
                    print(f"[train] could not restore optimizer state: {e}", flush=True)
            start_step = ckpt.get("step", 0)
            if "gene_pool" in ckpt and not cfg.baseline:
                from .genome import GenePool
                brain.gene_pool = GenePool.from_state(ckpt["gene_pool"])
            # Restore memory checkpoint if available
            if not cfg.baseline:
                mem_path = str(resume_path).replace(".pt", ".mem")
            if Path(mem_path).exists():
                try:
                    brain.load_memory_checkpoint(mem_path)
                    print(f"[train] restored memory from {mem_path}", flush=True)
                except Exception as e:
                    print(f"[train] could not restore memory: {e}", flush=True)
            print(f"[train] resumed from {resume_path} @ step {start_step}", flush=True)
    elif args.transfer and Path(args.transfer).exists():
        ckpt = torch.load(args.transfer, map_location=device, weights_only=False)
        brain.load_partial(ckpt["model"])
        if "gene_pool" in ckpt and not cfg.baseline:
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
        if not cfg.baseline:
            content = tok.decode(ids[0].tolist())
            content_vec = ids[0].float().cpu().numpy()  # Placeholder: use embedding for real system
            nt_state = brain.transmitters.vector()[0].cpu().numpy()
            emotion = None  # Placeholder: can be inferred from NT or model output
            tags = []
            context = {'self': True}  # Placeholder: can be set based on batch/task
            brain.record_episode(content, content_vec, nt_state, emotion, tags, context)

        # 2. Forward pass (full brain pipeline — no create_graph needed here)
        with amp_ctx():
            out = brain.forward_lm(ids, targets)
            loss = out["loss"]

        # If meta-training is enabled, perform a one-step differentiable unroll.
        # We do a SEPARATE language-only forward pass for the meta path to avoid
        # in-place modification issues from the full brain pipeline.
        if args.meta and not cfg.baseline:
            # FOMAML meta-training: first-order grads + meta-forward.
            # No need for math SDP or CuDNN disabling since we don't do
            # higher-order differentiation anymore.

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

            # FOMAML: compute first-order grads (no create_graph).
            # meta_loss.backward() still differentiates through learned_opt
            # (the meta-learnable part) without needing second-order grads
            # through the language model forward pass itself.
            with amp_ctx():
                inner_logits, _, _ = brain.language(ids)
                inner_loss = torch.nn.functional.cross_entropy(
                inner_logits.reshape(-1, inner_logits.size(-1)),
                targets.reshape(-1), ignore_index=-100)

            # Meta-learn language module parameters (including geometry adapters)
            model_named = list(brain.language.named_parameters())
            model_params = [p for _, p in model_named]
            grads = torch.autograd.grad(inner_loss, model_params,
                                        create_graph=False, allow_unused=True)
            # Detach grads (first-order approx) — learned_opt still gets gradients
            grads = tuple(g.detach() if g is not None else None for g in grads)

            # neuromodulatory vector (DA, NE, 5HT, ACh)
            nm = brain.transmitters.vector().detach().mean(dim=0)[:4].to(device)

            # form virtual updated parameters for the language module
            virtual_map = {}
            for (name, p), g in zip(model_named, grads):
                if g is None:
                    g = torch.zeros_like(p)
                mult = brain.learned_opt(g, p.detach(), nm,
                                         comprehension_delta=comp_delta,
                                         param_name=name)
                transformed = g * mult
                # Detach p so backward only flows through learned_opt (mult),
                # not back into the live language parameters.
                virtual = p.detach() - cfg.lr * transformed
                virtual_map[name] = virtual

            # Evaluate meta-loss under virtual language params
            # ── Comprehension-focused meta-objective ──
            # Instead of raw cross-entropy (which rewards fast memorization),
            # we optimize for deep understanding: calibrated predictions,
            # diverse semantic representations, and smooth reasoning.
            with amp_ctx():
                meta_out = functional_call(brain.language, virtual_map, (meta_ids,))
                logits_meta, sem_meta, _ = meta_out

            # (a) Base language modeling loss
            raw_lm_loss = torch.nn.functional.cross_entropy(
                logits_meta.reshape(-1, logits_meta.size(-1)),
                meta_targets.reshape(-1), ignore_index=-100)

            # (b) Calibration penalty — penalize overconfident wrong predictions
            #     Comprehension = knowing what you DON'T know
            meta_probs = torch.softmax(logits_meta.detach(), dim=-1)
            top_prob = meta_probs.max(dim=-1).values.mean()
            calibration = torch.relu(top_prob - 0.85) * 2.0

            # (c) Semantic diversity — rich internal representations
            #     Collapsed representations = rote memorization, not understanding
            if sem_meta is not None and sem_meta.size(1) > 1:
                sem_flat = sem_meta.reshape(-1, sem_meta.size(-1))
                sem_norm = torch.nn.functional.normalize(sem_flat, dim=-1)
                # Sample subset to avoid O(n²) cost
                n_sample = min(64, sem_norm.size(0))
                idx = torch.randperm(sem_norm.size(0), device=device)[:n_sample]
                sem_sub = sem_norm[idx]
                sim = torch.mm(sem_sub, sem_sub.T)
                # Off-diagonal mean: lower = more diverse
                mask = ~torch.eye(n_sample, device=device, dtype=torch.bool)
                diversity_loss = sim[mask].mean()
            else:
                diversity_loss = torch.zeros(1, device=device)

            # (d) Prediction smoothness — comprehension means coherent predictions
            #     Erratic logit jumps = pattern matching, not understanding
            if logits_meta.size(1) > 2:
                logit_diff = (logits_meta[:, 1:] - logits_meta[:, :-1]).pow(2).mean()
                smoothness_loss = logit_diff * 0.001
            else:
                smoothness_loss = torch.zeros(1, device=device)

            # Combined comprehension meta-loss
            meta_loss = (
                raw_lm_loss
                + 0.1 * calibration
                + 0.05 * diversity_loss
                + smoothness_loss
            )

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

        if not args.meta and not cfg.baseline:
            optim.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(optim)
            gnorm = torch.nn.utils.clip_grad_norm_(brain.parameters(), cfg.grad_clip)
            scaler.step(optim)
            scaler.update()

        if not args.meta and cfg.baseline:
            optim.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(optim)
            gnorm = torch.nn.utils.clip_grad_norm_(brain.parameters(), cfg.grad_clip)
            scaler.step(optim)
            scaler.update()

        # 3. Tag memory with reward/insight (mesolimbic)
        if not cfg.baseline:
            reward = float(out["learning_gain"][0].item()) if "learning_gain" in out else 0.0
            insight = None  # Placeholder: can be set if new pattern detected
            brain.tag_memory(len(brain.episodic.buffer)-1, reward, insight)

        # 4. Consolidate and update narratives every 500 steps
        if not cfg.baseline and (step + 1) % 500 == 0:
            brain.consolidate_memory()
            brain.update_narratives()

        # Slow homeostatic regulation of NT baselines & gains.
        if not cfg.baseline:
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
            if cfg.baseline:
                print(f"step {step+1:5d} | loss {avg:.4f} | lm {avg_lm:.4f} "
                      f"| ppl {ppl:.1f} | lr {optim.param_groups[0]['lr']:.2e} "
                      f"| {tok_per_s:.0f} tok/s | BASELINE", flush=True)
            else:
                nt_str = " ".join(f"{k}={v:.2f}" for k, v in (brain.last_nt or {}).items())
                lg = float(brain.last_learning_gain.mean()) if brain.last_learning_gain is not None else 0.0
                print(f"step {step+1:5d} | loss {avg:.4f} | lm {avg_lm:.4f} "
                      f"| ppl {ppl:.1f} | lr {optim.param_groups[0]['lr']:.2e} "
                      f"| {tok_per_s:.0f} tok/s | mesoLG {lg:.2f} | NT[{nt_str}]", flush=True)
                # Log oscillation spectrum (tick() now called in forward_lm)
                try:
                    osc = brain.oscillation_tracker.compute_spectrum()
                    print(f"         oscillations: {osc.format()}", flush=True)
                except Exception:
                    pass
            running_loss = running_lm = 0.0
            n_obs = 0
            t0 = time.time()

        if (step + 1) % args.save_every == 0 or (step + 1) == args.steps:
            tag = "" if args.mode == "text" else f"_{args.mode}"
            bflag = "_baseline" if cfg.baseline else ""
            path = Path(args.ckpt_dir) / f"neuroslm_{args.preset}{tag}{bflag}_{step+1}.pt"
            save_dict = {
                "model": brain.state_dict(),
                "optim": optim.state_dict(),
                "cfg": cfg.__dict__,
                "step": step + 1,
                "preset": args.preset,
            }
            if not cfg.baseline:
                save_dict["gene_pool"] = brain.gene_pool.state()
                save_dict["trophic_stats"] = brain.trophic.stats()
                if hasattr(brain, 'module_genomes'):
                    save_dict["module_genomes"] = brain.module_genomes.state()
                    save_dict["compiled_lisp"] = brain.get_all_module_lisp()
                if hasattr(brain, 'brain_dna'):
                    save_dict["brain_dna"] = brain.brain_dna.to_dict()
            torch.save(save_dict, path)
            if not cfg.baseline:
                # ── Save portable memory checkpoint (.mem) ──
                try:
                    mem_path = Path("lfs_checkpoints") / f"neuroslm_{args.preset}{tag}_{step+1}.mem"
                    mem_path.parent.mkdir(parents=True, exist_ok=True)
                    stats = brain.save_memory_checkpoint(mem_path)
                    print(f"[train] saved memory checkpoint {mem_path.name} | {stats}", flush=True)
                except Exception as e:
                    print(f"[train] memory checkpoint failed: {e}", flush=True)
                # ── Intelligence metrics snapshot ──
                try:
                    brain.metrics.observe_narrative(brain.narrative_system)
                    brain.metrics.observe_memory(brain.episodic, brain.consolidated, brain.causal)
                    m = brain.metrics.format()
                    print(f"[train] intelligence: {m}", flush=True)
                except Exception as e:
                    print(f"[train] metrics snapshot failed: {e}", flush=True)
                # ── Genome Compilation Report: Genome → Latent → Lisp → Execute ──
                try:
                    if hasattr(brain, 'genome_compiler'):
                        decomp_dir = Path(args.ckpt_dir) / f"compiled_step_{step+1}"
                        brain.genome_compiler.save_all_lisp(str(decomp_dir))
                        report = brain.genome_compilation_report()
                        print(f"[train] genome compilation:\n{report}", flush=True)
                        # Print first 3 modules' Lisp for quick inspection
                        for i, (name, src) in enumerate(brain.get_all_module_lisp().items()):
                            if i >= 3:
                                print(f"[train] ... and {len(brain.get_all_module_lisp()) - 3} more modules", flush=True)
                                break
                            print(f"[train] {name} compiled Lisp:\n{src}", flush=True)
                except Exception as e:
                    print(f"[train] genome compilation report failed: {e}", flush=True)
                print(f"[train] saved {path} | genome={brain.gene_pool.active().id} "
                      f"gen={brain.gene_pool.active().generation} | "
                      f"trophic={brain.trophic.stats()}", flush=True)
            else:
                print(f"[train] saved {path} (baseline)", flush=True)

            # ── Auto-push every checkpoint to Git LFS ──
            push_checkpoint_to_lfs(str(path))

    print("[train] done.", flush=True)


if __name__ == "__main__":
    main()
