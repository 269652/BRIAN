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
from .xla_utils import (
    get_device, is_xla, mark_step, optimizer_step,
    should_use_gradient_checkpointing, to_bfloat16, make_loader,
)
try:
    from torch.func import functional_call
except ImportError:
    try:
        from functorch import functional_call  # type: ignore[no-redef]
    except ImportError:
        functional_call = None  # type: ignore[assignment]


# ---------------------------------------------------------------------------
# Global bfloat16 safety patches.
#
# Two issues need fixing on this PyTorch/CUDA build:
#
# 1. LayerNorm: F.layer_norm requires float32 for both input and weight/bias
#    on some kernels.  The patch promotes everything to float32 internally
#    and casts the output back to the input dtype.
#
# 2. MultiheadAttention: the MHA fast path can leave the query in a different
#    dtype than the in_proj_weight (e.g. when the input was just normalised in
#    fp32 round-trip).  The patch coerces query/key/value to the weight dtype
#    before calling the original forward, so linear() always sees matching
#    dtypes.
#
# Both patches are no-ops in pure-fp32 training.
# ---------------------------------------------------------------------------
import torch.nn.functional as _F
_orig_ln_fwd = torch.nn.LayerNorm.forward

def _bf16_safe_ln_fwd(self, x: torch.Tensor) -> torch.Tensor:
    dtype = x.dtype
    if dtype == torch.float32 and (self.weight is None or self.weight.dtype == torch.float32):
        return _orig_ln_fwd(self, x)
    return _F.layer_norm(
        x.float(),
        self.normalized_shape,
        self.weight.float() if self.weight is not None else None,
        self.bias.float() if self.bias is not None else None,
        self.eps,
    ).to(dtype)

torch.nn.LayerNorm.forward = _bf16_safe_ln_fwd

_orig_mha_fwd = torch.nn.MultiheadAttention.forward

def _bf16_safe_mha_fwd(self, query, key, value, *args, **kwargs):
    if self.in_proj_weight is not None:
        w_dtype = self.in_proj_weight.dtype
    elif getattr(self, "q_proj_weight", None) is not None:
        w_dtype = self.q_proj_weight.dtype
    else:
        w_dtype = query.dtype
    if query.dtype != w_dtype:
        query = query.to(dtype=w_dtype)
    if key is not None and key.dtype != w_dtype:
        key = key.to(dtype=w_dtype)
    if value is not None and value.dtype != w_dtype:
        value = value.to(dtype=w_dtype)
    return _orig_mha_fwd(self, query, key, value, *args, **kwargs)

torch.nn.MultiheadAttention.forward = _bf16_safe_mha_fwd

_orig_linear_fwd = torch.nn.Linear.forward

def _bf16_safe_linear_fwd(self, x: torch.Tensor) -> torch.Tensor:
    if x.dtype != self.weight.dtype:
        x = x.to(dtype=self.weight.dtype)
    return _orig_linear_fwd(self, x)

torch.nn.Linear.forward = _bf16_safe_linear_fwd
# ---------------------------------------------------------------------------


def cosine_lr(step: int, warmup: int, total: int, peak: float) -> float:
    if step < warmup:
        return peak * step / max(1, warmup)
    progress = (step - warmup) / max(1, total - warmup)
    return peak * 0.5 * (1.0 + math.cos(math.pi * min(progress, 1.0)))


def push_checkpoint_to_lfs(ckpt_path: str, repo_root: str | None = None):
    """Copy checkpoint + memory to lfs_checkpoints/ and push via Git LFS.

    Auth: relies on ~/.git-credentials written by the Colab setup cell.
    Falls back to injecting the GITHUB env-var token into the remote URL.
    """
    import shutil, subprocess
    try:
        if repo_root is None:
            repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        lfs_dir = os.path.join(repo_root, "lfs_checkpoints")
        os.makedirs(lfs_dir, exist_ok=True)

        basename = os.path.basename(ckpt_path)

        # Copy .pt
        dest = os.path.join(lfs_dir, basename)
        if os.path.abspath(ckpt_path) != os.path.abspath(dest):
            shutil.copy2(ckpt_path, dest)

        # Copy .mem if present
        mem_src = ckpt_path.replace('.pt', '.mem')
        mem_dst = os.path.join(lfs_dir, basename.replace('.pt', '.mem'))
        if os.path.exists(mem_src) and os.path.abspath(mem_src) != os.path.abspath(mem_dst):
            shutil.copy2(mem_src, mem_dst)

        # Ensure git identity
        subprocess.run(["git", "config", "user.email", "train@neuroslm"],
                       cwd=repo_root, capture_output=True)
        subprocess.run(["git", "config", "user.name", "NeuroSLM Train"],
                       cwd=repo_root, capture_output=True)

        # Prefer credential store (written by Colab cell 2); fall back to URL injection
        creds_file = os.path.expanduser("~/.git-credentials")
        token = (os.environ.get('GITHUB') or os.environ.get('GITHUB_TOKEN', '')).strip()
        if token and not os.path.exists(creds_file):
            import re
            result = subprocess.run(["git", "remote", "get-url", "origin"],
                                    cwd=repo_root, capture_output=True, text=True)
            url = re.sub(r'https://[^@]+@', 'https://', result.stdout.strip())
            subprocess.run(["git", "remote", "set-url", "origin",
                            url.replace('https://', f'https://{token}@', 1)],
                           cwd=repo_root, capture_output=True)

        r_add = subprocess.run(["git", "add", "-f", "lfs_checkpoints/"],
                               cwd=repo_root, capture_output=True, timeout=30, text=True)
        if r_add.returncode != 0:
            print(f"[train] ⚠ git add failed: {r_add.stderr[:100]}", flush=True)
            return

        r_commit = subprocess.run(["git", "commit", "--allow-empty", "-m", f"chkpt: {basename}"],
                                  cwd=repo_root, capture_output=True, text=True, timeout=30)
        if r_commit.returncode != 0 and "nothing to commit" not in r_commit.stdout.lower():
            print(f"[train] ⚠ git commit failed: {r_commit.stderr[:100]}", flush=True)
            return

        r_push = subprocess.run(["git", "push", "origin", "HEAD"],
                                cwd=repo_root, capture_output=True, text=True, timeout=120)
        if r_push.returncode == 0:
            print(f"[train] ✓ pushed {basename} to Git LFS", flush=True)
        else:
            print(f"[train] ⚠ git push failed: {r_push.stderr.strip()[:200]}", flush=True)
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
    ap.add_argument("--overwrite_ckpt", action="store_true",
                    help="Save to a fixed '_latest.pt' filename (overwrite each save) instead "
                         "of accumulating step-numbered files. Keeps LFS storage constant. "
                         "If the saved architecture no longer matches, loading is skipped "
                         "automatically and training starts fresh.")
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
    ap.add_argument("--grad_accum", type=int, default=1,
                    help="gradient accumulation steps; effective_batch = batch_size * grad_accum")
    ap.add_argument("--curriculum", action="store_true",
                    help="Start with short sequences and grow to full ctx_len over training "
                         "(improves DNC memory pointer calibration)")
    ap.add_argument("--optimizer", default="adafactor",
                    choices=["adafactor", "adamw"],
                    help="adafactor (default, TPU-optimal) or adamw (CUDA/debug)")
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    # ── Device selection (XLA → CUDA → CPU) ────────────────────────────
    if args.device:
        import os as _os
        _os.environ["NEUROSLM_DEVICE"] = args.device
    device = get_device()
    backend = "XLA/TPU" if is_xla() else ("CUDA" if torch.cuda.is_available() else "CPU")
    print(f"[train] device={device}  backend={backend}", flush=True)

    # Meta-mode: no second-order grads are needed (FOMAML), works on XLA too.
    # Warn if trying to use meta on plain CPU (slower, not unsupported).
    if args.meta and backend == "CPU":
        print("[train] WARNING: meta-training on CPU is slow. Consider --device cuda or TPU.",
              flush=True)

    cfg = PRESETS[args.preset]()
    tok = Tokenizer()
    cfg.vocab_size = tok.vocab_size
    if args.baseline:
        cfg.baseline = True
    # Gradient checkpointing is beneficial on both XLA and CUDA
    if should_use_gradient_checkpointing():
        cfg.gradient_checkpointing = True
    ctx_len = args.ctx or cfg.lang_ctx
    assert ctx_len <= cfg.lang_ctx

    # Auxiliary loss weights start small and ramp up once the LM loss drops below 8.0.
    # pred_coding_loss and active_inference_fe start at 0.01× their target weight,
    # ramping up only after lm_loss < 8.0 for a sustained window (100 steps).
    _aux_w_init   = 0.01
    _aux_w_target = 1.0
    _loss_ramp_threshold = 8.0
    _loss_ramp_window = 100  # steps below threshold to trigger ramp
    _loss_below_threshold_count = 0
    _lm_ema = None   # exponential moving average of LM loss (for gating)
    _ramp_started = False  # flag tracking if we've passed the threshold

    # ── Build model and cast to native precision ────────────────────────
    # TPU native format: bfloat16 (same dynamic range as fp32, half memory).
    # CUDA: bfloat16 (Ampere+) or fp32 on older cards.
    # CPU: fp32 (bfloat16 is slow on CPU).
    # AMP (GradScaler) is NOT used — bfloat16 does not need loss scaling because
    # it has the same exponent range as fp32 (unlike float16 which underflows).
    print("[train] building Brain...", flush=True)
    try:
        brain = Brain(cfg).to(device)
    except Exception as _e:
        print(f"[train] FATAL: Brain() failed: {_e}", flush=True)
        import traceback; traceback.print_exc()
        raise
    brain = to_bfloat16(brain)
    amp_ctx = nullcontext   # no autocast needed; model is already in bfloat16
    n_params = sum(p.numel() for p in brain.parameters())
    mode_label = "BASELINE (vanilla transformer)" if cfg.baseline else "FULL (bio modules)"
    print(f"[train] {mode_label} | params: {n_params/1e6:.2f}M "
          f"(preset={args.preset}, dtype={next(brain.parameters()).dtype})", flush=True)

    # ── Optimizer ────────────────────────────────────────────────────────
    # Adafactor: factor-wise second moment — ~8× less optimizer memory than
    # AdamW, essential for 258M+ params on TPU HBM.  Falls back to AdamW
    # when --optimizer=adamw is passed (useful for debugging).
    named = list(brain.named_parameters())
    model_params = [p for n, p in named if not n.startswith('learned_opt.')]
    if args.optimizer == "adafactor":
        _Adafactor = None
        try:
            from transformers.optimization import Adafactor as _Adafactor  # type: ignore[attr-defined]
        except ImportError:
            pass
        if _Adafactor is None:
            try:
                from torch.optim import Adafactor as _Adafactor  # type: ignore[attr-defined]
            except ImportError:
                pass
        if _Adafactor is not None:
            optim = _Adafactor(
                model_params,
                lr=None,
                scale_parameter=True,
                relative_step=True,
                warmup_init=True,
                weight_decay=cfg.weight_decay,
            )
            print("[train] optimizer=Adafactor (TPU-native)", flush=True)
        else:
            print("[train] WARNING: Adafactor not found; falling back to AdamW. "
                  "Install transformers: pip install transformers", flush=True)
            args.optimizer = "adamw"
    if args.optimizer == "adamw":
        optim = AdamW(model_params, lr=cfg.lr,
                      weight_decay=cfg.weight_decay, betas=(0.9, 0.95))

    meta_opt = None
    if args.meta and functional_call is None:
        print("[train] WARNING: torch.func.functional_call unavailable; "
              "disabling meta-training", flush=True)
        args.meta = False
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
            import glob as _glob
            # With --overwrite_ckpt prefer the fixed _latest file
            _latest_fixed = os.path.join(
                args.ckpt_dir,
                f"neuroslm_{args.preset}_{'baseline_' if args.baseline else ''}latest.pt")
            if args.overwrite_ckpt and os.path.exists(_latest_fixed):
                resume_path = _latest_fixed
            else:
                candidates = sorted(
                    _glob.glob(os.path.join(args.ckpt_dir, "*.pt")),
                    key=lambda f: os.path.getmtime(f))
                if not candidates:
                    candidates = sorted(
                        _glob.glob(os.path.join(
                            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                            "lfs_checkpoints", "*.pt")),
                        key=lambda f: os.path.getmtime(f))
                resume_path = candidates[-1] if candidates else None
                if resume_path:
                    print(f"[train] auto-found latest checkpoint: {resume_path}", flush=True)
                else:
                    print("[train] no checkpoint found, training from scratch", flush=True)

        if resume_path and Path(resume_path).exists():
            try:
                ckpt = torch.load(resume_path, map_location=device, weights_only=False)
                missing, unexpected = brain.load_state_dict(ckpt["model"], strict=False)
                # With --overwrite_ckpt, treat large key-mismatch as architecture change
                # and start fresh rather than loading a partly-matching checkpoint.
                _mismatch = len(missing) + len(unexpected)
                if args.overwrite_ckpt and _mismatch > 20:
                    print(f"[train] ⚠ architecture mismatch ({_mismatch} keys differ) "
                          f"— starting fresh (--overwrite_ckpt)", flush=True)
                    start_step = 0
                else:
                    if "optim" in ckpt:
                        try:
                            optim.load_state_dict(ckpt["optim"])
                        except Exception as e:
                            print(f"[train] could not restore optimizer state: {e}", flush=True)
                    start_step = ckpt.get("step", 0)
                    if not cfg.baseline:
                        mem_path = str(resume_path).replace(".pt", ".mem")
                        if Path(mem_path).exists():
                            try:
                                brain.load_memory_checkpoint(mem_path)
                                print(f"[train] restored memory from {mem_path}", flush=True)
                            except Exception as e:
                                print(f"[train] could not restore memory: {e}", flush=True)
                    print(f"[train] resumed from {resume_path} @ step {start_step}", flush=True)
            except Exception as _load_err:
                if args.overwrite_ckpt:
                    print(f"[train] ⚠ could not load checkpoint ({_load_err}) "
                          f"— starting fresh (--overwrite_ckpt)", flush=True)
                    start_step = 0
                else:
                    raise
    elif args.transfer and Path(args.transfer).exists():
        ckpt = torch.load(args.transfer, map_location=device, weights_only=False)
        brain.load_partial(ckpt["model"])
        print(f"[train] transferred matching tensors from {args.transfer}", flush=True)

    Path(args.ckpt_dir).mkdir(parents=True, exist_ok=True)

    print(f"[train] starting; ctx={ctx_len}, batch={args.batch_size}, "
        f"steps={args.steps}, mode={args.mode}"
        + (f" (chat_ratio={args.chat_ratio})" if args.mode == "mix" else ""), flush=True)
    _curriculum_end = max(1, int(args.steps * 0.15))  # ramp ctx over first 15%
    _raw_it = batch_iterator(tok, ctx_len, args.batch_size, seed=args.seed,
                             mode=args.mode, chat_ratio=args.chat_ratio,
                             curriculum=args.curriculum,
                             curriculum_start=0.25,
                             curriculum_end_step=_curriculum_end,
                             current_step=start_step)
    it = make_loader(_raw_it, device)
    t0 = time.time()
    running_loss = 0.0
    running_lm = 0.0
    n_obs = 0

    total_steps = args.steps
    if start_step >= total_steps:
        print(f"[train] already at step {start_step} >= {total_steps}; nothing to do.", flush=True)
        return
    print(f"[train] >>> training loop starting: steps {start_step}..{total_steps} <<<", flush=True)

    for step in range(start_step, total_steps):
        try:
            batch = next(it)
        except StopIteration:
            print("[train] dataset exhausted; restarting iterator", flush=True)
            _raw_it = batch_iterator(tok, ctx_len, args.batch_size,
                                     seed=args.seed + step,
                                     mode=args.mode, chat_ratio=args.chat_ratio,
                                     curriculum=args.curriculum,
                                     curriculum_start=0.25,
                                     curriculum_end_step=_curriculum_end,
                                     current_step=step)
            it = make_loader(_raw_it, device)
            batch = next(it)

        # _IdentityLoader already called .to(device) on the batch.
        ids, targets = batch[:, :-1], batch[:, 1:].contiguous()

        # LR schedule — Adafactor manages its own LR when relative_step=True;
        # only apply the cosine schedule when using AdamW.
        if args.optimizer != "adafactor":
            for pg in optim.param_groups:
                pg["lr"] = cosine_lr(step, cfg.warmup_steps, total_steps, cfg.lr)

        # --- Memory system integration ---
        # 1. Record episodic memory for this batch (deferred to CPU; non-blocking)
        if not cfg.baseline:
            try:
                content = tok.decode(ids[0].cpu().tolist())
                content_vec = ids[0].float().cpu().numpy()
                nt_state = brain.transmitters.vector()[0].detach().cpu().numpy()
                brain.record_episode(content, content_vec, nt_state, None, [], {'self': True})
            except Exception:
                pass  # non-critical

        # 2. Forward pass (full brain pipeline)
        try:
            with amp_ctx():
                out = brain.forward_lm(ids, targets)
                loss = out["loss"]
                if torch.isnan(loss) or torch.isinf(loss):
                    loss = out["lm_loss"]
                    out["loss"] = loss
                    print(f"[train] ⚠ NaN/Inf in total loss at step {step+1}, "
                          f"using lm_loss={loss.item():.4f}", flush=True)
        except Exception as _fwd_err:
            import traceback
            print(f"[train] ✖ forward_lm failed at step {step+1}: {_fwd_err}", flush=True)
            traceback.print_exc()
            continue  # skip this step

        # ── Auxiliary loss gating ──────────────────────────────────────────
        # Ramp auxiliary weights from 0.01 → target only after lm_loss < 8.0
        # for a sustained window (100 steps), ensuring core LM competency first.
        _lm_now = float(out["lm_loss"].item())
        if _lm_ema is None:
            _lm_ema = _lm_now
        _lm_ema = 0.99 * _lm_ema + 0.01 * _lm_now

        # Track when we've sustained below-threshold loss
        if _lm_now < _loss_ramp_threshold:
            _loss_below_threshold_count += 1
        else:
            _loss_below_threshold_count = 0

        # Once sustained below threshold, ramp auxiliary weights linearly
        if _loss_below_threshold_count >= _loss_ramp_window:
            _ramp_started = True

        if _ramp_started:
            # Linear ramp from 0.01 to target over remaining training
            steps_ramped = _loss_below_threshold_count - _loss_ramp_window
            max_ramp_steps = args.steps - step
            _aux_ramp = min(1.0, steps_ramped / max(1, max_ramp_steps))
        else:
            _aux_ramp = 0.0

        _aux_w = _aux_w_init + (_aux_w_target - _aux_w_init) * _aux_ramp
        # Update config weights live so brain.forward_lm picks them up next step
        if not cfg.baseline:
            cfg.w_pred_coding = getattr(cfg, '_w_pred_coding_base',
                                        cfg.w_pred_coding) * _aux_w
            if not hasattr(cfg, '_w_pred_coding_base'):
                cfg._w_pred_coding_base = cfg.w_pred_coding

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
                _raw_meta = batch_iterator(tok, ctx_len, args.batch_size,
                                           seed=args.seed + step,
                                           mode=args.mode, chat_ratio=args.chat_ratio)
                it = make_loader(_raw_meta, device)
                meta_batch = next(it)
            meta_ids, meta_targets = meta_batch[:, :-1], meta_batch[:, 1:].contiguous()

            # FOMAML: compute first-order grads (no create_graph).
            # meta_loss.backward() still differentiates through learned_opt
            # (the meta-learnable part) without needing second-order grads
            # through the language model forward pass itself.
            with amp_ctx():
                inner_logits, _, _, _ = brain.language(ids)
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
                logits_meta, sem_meta, _, _ = meta_out

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

        if not args.meta:
            ga = max(args.grad_accum, 1)
            optim.zero_grad(set_to_none=True)
            # bfloat16 has fp32-equivalent exponent range → no loss scaling needed
            (loss / ga).backward()
            total_lm_loss = float(out["lm_loss"].item())
            # extra micro-batches for gradient accumulation
            for _ga in range(1, ga):
                try:
                    ga_batch = next(it)
                except StopIteration:
                    _raw_it2 = batch_iterator(tok, ctx_len, args.batch_size,
                                              seed=args.seed + step * ga + _ga,
                                              mode=args.mode, chat_ratio=args.chat_ratio)
                    it = make_loader(_raw_it2, device)
                    ga_batch = next(it)
                ga_ids = ga_batch[:, :-1]
                ga_tgt = ga_batch[:, 1:].contiguous()
                with amp_ctx():
                    ga_out = brain.forward_lm(ga_ids, ga_tgt)
                    ga_loss = ga_out["loss"]
                (ga_loss / ga).backward()
                total_lm_loss += float(ga_out["lm_loss"].item())
            gnorm = optimizer_step(optim, brain.parameters(), cfg.grad_clip)
            mark_step()   # XLA: flush graph; no-op on CUDA/CPU
            # update loss/lm_loss for logging to reflect full accumulation
            loss = torch.tensor(float(loss.item()))  # detach from graph
            out["lm_loss"] = torch.tensor(total_lm_loss / ga)

        # 3. Tag memory with reward/insight (mesolimbic)
        if not cfg.baseline:
            reward = float(out["learning_gain"][0].item()) if "learning_gain" in out else 0.0
            da_level = float(brain.transmitters.vector()[0, 0].item()) if hasattr(brain, 'transmitters') else 0.5
            brain.tag_memory(len(brain.episodic.buffer)-1, reward, da_level=da_level)

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
            tok_per_s = args.log_every * args.batch_size * max(args.grad_accum, 1) * ctx_len / max(dt, 1e-3)
            _raw_lr = optim.param_groups[0].get('lr')
            _lr_str = f"{_raw_lr:.2e}" if _raw_lr is not None else "auto"
            if cfg.baseline:
                print(f"step {step+1:5d} | loss {avg:.4f} | lm {avg_lm:.4f} "
                      f"| ppl {ppl:.1f} | lr {_lr_str} "
                      f"| {tok_per_s:.0f} tok/s | BASELINE", flush=True)
            else:
                nt_str = " ".join(f"{k}={v:.2f}" for k, v in (brain.last_nt or {}).items())
                lg = float(brain.last_learning_gain.mean()) if brain.last_learning_gain is not None else 0.0

                # IIT 4.0 / bowtie observability — Φ proxy, Fiedler λ₁, GWS
                # ignition, and trophic state. Pulled from `brain` attrs that
                # forward_lm sets on every pass (None-safe defaults so the log
                # never crashes a step).
                phi   = float(getattr(brain, "_last_phi",      0.0))
                fid   = float(getattr(brain, "_last_fiedler",  1.0))
                ign_t = getattr(getattr(brain, "gws", None), "_last_ignition", None)
                ign   = float(ign_t.mean()) if ign_t is not None else 0.0
                troph = brain.trophic.stats() if hasattr(brain, "trophic") else {}
                t_act = troph.get("n_active", 0)
                t_tot = troph.get("n_projections", 0)
                t_mu  = troph.get("trophic_mean", 0.0)

                # Oscillation spectrum (multi-band: gamma/theta/alpha + cross-frequency coupling)
                osc_str = ""
                if hasattr(brain, 'oscillation_tracker'):
                    try:
                        osc = brain.oscillation_tracker.compute_spectrum()
                        osc_str = f" | osc[γ={osc.gamma:.3f} θ={osc.theta:.3f} α={osc.alpha:.3f}]"
                    except Exception as e:
                        pass  # tracker exists but compute failed; skip

                print(f"step {step+1:5d} | loss {avg:.4f} | lm {avg_lm:.4f} "
                      f"| ppl {ppl:.1f} | lr {_lr_str} "
                      f"| {tok_per_s:.0f} tok/s "
                      f"| Φ {phi:.3f} | λ₁ {fid:.3f} | ign {ign:.2f} "
                      f"| mesoLG {lg:.2f} "
                      f"| troph {t_act}/{t_tot} μ{t_mu:.2f} "
                      f"| NT[{nt_str}]{osc_str}", flush=True)
            running_loss = running_lm = 0.0
            n_obs = 0
            t0 = time.time()

        if (step + 1) % args.save_every == 0 or (step + 1) == total_steps:
            tag = "" if args.mode == "text" else f"_{args.mode}"
            bflag = "_baseline" if cfg.baseline else ""
            if args.overwrite_ckpt:
                # Fixed filename — overwritten every save, keeps LFS storage constant.
                # Step number is stored inside the file so resume still works.
                fname = f"neuroslm_{args.preset}{bflag}_latest.pt"
            else:
                fname = f"neuroslm_{args.preset}{tag}{bflag}_{step+1}.pt"
            path = Path(os.path.abspath(args.ckpt_dir)) / fname
            path.parent.mkdir(parents=True, exist_ok=True)
            save_dict = {
                "model": brain.state_dict(),
                "optim": optim.state_dict(),
                "cfg": cfg.__dict__,
                "step": step + 1,
                "preset": args.preset,
            }
            if not cfg.baseline:
                save_dict["trophic_stats"] = brain.trophic.stats()
            torch.save(save_dict, path)
            if not cfg.baseline:
                # ── Save portable memory checkpoint (.mem) ──
                try:
                    mem_path = path.parent / f"neuroslm_{args.preset}{tag}_{step+1}.mem"
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
                print(f"[train] saved {path} | trophic={brain.trophic.stats()}", flush=True)
            else:
                print(f"[train] saved {path} (baseline)", flush=True)

            # ── Auto-push every checkpoint to Git LFS ──
            push_checkpoint_to_lfs(str(path))

    print("[train] done.", flush=True)


if __name__ == "__main__":
    main()
