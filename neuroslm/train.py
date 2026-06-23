"""Training loop for NeuroSLM.

Streams a Phi-style open dataset (Cosmopedia by default) and runs the brain's
multi-objective forward pass.

Usage:
    python -m neuroslm.train --preset small --steps 2000 --batch_size 4
"""
from __future__ import annotations
import argparse
import json
import math
import os
import time
from contextlib import nullcontext
from pathlib import Path
from typing import Optional
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


def cosine_lr(step: int, warmup: int, total: int, peak: float,
              min_ratio: float = 0.0) -> float:
    """Linear warmup → cosine decay to `peak * min_ratio` by step `total`.

    `total` is the DECAY HORIZON, not necessarily the run length. If you train
    a model you intend to stop and evaluate at 10k, pass total=10000 so the LR
    actually anneals by then — the single biggest lever on final (and OOD)
    perplexity. With the old behavior (total = --steps = 100000) the LR is
    still ~98% of peak at step 10k, so an early-stopped model never sees the
    annealing phase.
    """
    if step < warmup:
        return peak * step / max(1, warmup)
    progress = (step - warmup) / max(1, total - warmup)
    cos = 0.5 * (1.0 + math.cos(math.pi * min(progress, 1.0)))
    return peak * (min_ratio + (1.0 - min_ratio) * cos)


def maturity_aux_gate(maturity: float, lo: float, hi: float) -> float:
    """Maturity-gated auxiliary-loss weight in [0, 1] (fix A, revised).

    aux scale = 0 below `lo`, ramps linearly to 1 at `hi`. Ties auxiliary-
    objective strength to the LM maturity index (MAT = 1 − lm_loss/L_random),
    so aux losses only gain weight once the LM is genuinely good (e.g. MAT
    0.50≈PPL220 → 0.65≈PPL50). This is self-correcting: if the LM regresses,
    MAT falls and the gate closes, letting the model refocus on LM.

    Replaces step-schedule ramps, which either blew up late (denominator → 0)
    or, when fixed-length, slammed aux to full by ~step 2.5k and overwhelmed
    the still-immature LM (observed early divergence).
    """
    if hi <= lo:
        return 1.0 if maturity >= hi else 0.0
    return min(1.0, max(0.0, (maturity - lo) / (hi - lo)))


def build_param_groups(named_params, weight_decay: float, decoupled: bool):
    """Split named params into optimizer groups for (decoupled) weight decay.

    `named_params`: iterable of (name, Parameter) — typically
        brain.named_parameters(); params under 'learned_opt.' are excluded
        (they belong to the meta-optimizer).

    decoupled=True  → two groups: 2-D matrices (Linear/embedding weights) get
                      `weight_decay`; 1-D params (RMSNorm gains, biases, NT
                      levels/vesicles) get 0.0. Decaying norms/biases is a
                      known generalization regression.
    decoupled=False → a single flat list of params (legacy behavior; the
                      optimizer applies its own uniform weight_decay).

    Returns either a list of param-group dicts (decoupled) or a flat list of
    Parameters (legacy).
    """
    params = [(n, p) for n, p in named_params
              if not n.startswith('learned_opt.') and p.requires_grad]
    if not decoupled:
        return [p for _, p in params]
    decay   = [p for _, p in params if p.dim() >= 2]
    nodecay = [p for _, p in params if p.dim() < 2]
    return [
        {"params": decay,   "weight_decay": weight_decay},
        {"params": nodecay, "weight_decay": 0.0},
    ]


def _resize_grown_adapters(brain, state: dict):
    """Resize NeuralGeometryAdapter kern_a/kern_b to match a checkpoint.

    BDNF structural growth increases the adapter's low-rank kernel during
    training (rank 96 → 100 → … as Φ accumulates), so a checkpoint saved
    after any growth has larger kern_a/kern_b than a freshly-built model.
    `load_state_dict(strict=False)` ignores missing/unexpected keys but
    STILL raises on a shape mismatch for keys present in both — so without
    this pre-resize, resuming a grown checkpoint always crashes.

    We re-create the model's kern Parameters at the checkpoint's shape
    (zero-init; load_state_dict fills the real values next).
    """
    import torch.nn as nn
    lang = getattr(brain, "language", None)
    adapters = getattr(lang, "adapters", None)
    if adapters is None:
        return
    for i, adapter in enumerate(adapters):
        ka = f"language.adapters.{i}.kern_a"
        if ka not in state or not hasattr(adapter, "kern_a"):
            continue
        ck_a = state[ka].shape          # (d_hyper, rank_ckpt)
        if tuple(adapter.kern_a.shape) == tuple(ck_a):
            continue
        d_hyper, new_rank = int(ck_a[0]), int(ck_a[1])
        dev = adapter.kern_a.device
        dt  = adapter.kern_a.dtype
        adapter.kern_a = nn.Parameter(torch.zeros(d_hyper, new_rank, device=dev, dtype=dt))
        adapter.kern_b = nn.Parameter(torch.zeros(new_rank, d_hyper, device=dev, dtype=dt))
        if hasattr(adapter, "rank"):
            adapter.rank = new_rank


def _load_compatible(brain, state: dict, label: str = "checkpoint"):
    """Robustly load a checkpoint state-dict into `brain`.

    1. Resize BDNF-grown adapters to the checkpoint's shapes.
    2. Drop any *still* shape-mismatched tensors (genuine architecture
       drift) so load_state_dict(strict=False) cannot raise.
    Returns (missing, unexpected, dropped) for logging.
    """
    _resize_grown_adapters(brain, state)
    model_sd = brain.state_dict()
    filtered, dropped = {}, []
    for k, v in state.items():
        if k in model_sd and hasattr(v, "shape") \
                and tuple(model_sd[k].shape) != tuple(v.shape):
            dropped.append(k)
            continue
        filtered[k] = v
    missing, unexpected = brain.load_state_dict(filtered, strict=False)
    if dropped:
        print(f"[train] ⚠ {len(dropped)} shape-mismatched tensor(s) skipped "
              f"while loading {label} (e.g. {dropped[0]})", flush=True)
    return missing, unexpected, dropped


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
        token = (os.environ.get('GH_TOKEN') or os.environ.get('GITHUB') or os.environ.get('GITHUB_TOKEN', '')).strip()
        if token and not os.path.exists(creds_file):
            import re
            result = subprocess.run(["git", "remote", "get-url", "origin"],
                                    cwd=repo_root, capture_output=True, text=True)
            url = re.sub(r'https://[^@]+@', 'https://', result.stdout.strip())
            subprocess.run(["git", "remote", "set-url", "origin",
                            url.replace('https://', f'https://{token}@', 1)],
                           cwd=repo_root, capture_output=True)

        # `git add` for a 430 MB LFS .pt routinely takes 60-120s — the old
        # 30s timeout silently truncated the stage and left the index empty,
        # while `--allow-empty` then produced a chkpt: commit with no content.
        # Bumped to 600s (10 min) which is plenty even on slow disks.
        r_add = subprocess.run(["git", "add", "-f", "lfs_checkpoints/"],
                               cwd=repo_root, capture_output=True, timeout=600, text=True)
        if r_add.returncode != 0:
            print(f"[train] ⚠ git add failed: {r_add.stderr[:200]}", flush=True)
            return

        # No --allow-empty — if the add produced nothing we want the commit
        # to fail loudly so the user knows the upload didn't actually happen.
        r_commit = subprocess.run(["git", "commit", "-m", f"chkpt: {basename}"],
                                  cwd=repo_root, capture_output=True, text=True, timeout=60)
        if r_commit.returncode != 0:
            stdout_low = r_commit.stdout.lower()
            if ("nothing to commit" in stdout_low
                    or "no changes added" in stdout_low):
                print(f"[train] ⚠ nothing to commit for {basename} — file may "
                      f"already be tracked, or git add timed out silently",
                      flush=True)
            else:
                print(f"[train] ⚠ git commit failed: "
                      f"{(r_commit.stderr or r_commit.stdout)[:200]}", flush=True)
            return

        # `git push` of an LFS object is bandwidth-bound — bumped to 600s.
        # Concurrency: when several instances (e.g. a full run + a baseline
        # run launched by scripts/vast_deploy.sh) push to the same branch,
        # the non-first push is rejected (non-fast-forward). Because each run
        # writes a DIFFERENT checkpoint stream (full vs *_baseline_*), a
        # `git pull --rebase` always merges cleanly — so we retry with a
        # rebase up to 5×.
        _branch = subprocess.run(["git", "rev-parse", "--abbrev-ref", "HEAD"],
                                 cwd=repo_root, capture_output=True, text=True,
                                 timeout=30).stdout.strip() or "HEAD"
        pushed = False
        for attempt in range(5):
            r_push = subprocess.run(["git", "push", "origin", "HEAD"],
                                    cwd=repo_root, capture_output=True,
                                    text=True, timeout=600)
            if r_push.returncode == 0:
                print(f"[train] ✓ pushed {basename} to Git LFS", flush=True)
                pushed = True
                break
            # Rejected (likely a concurrent push) — rebase onto remote & retry.
            print(f"[train] push attempt {attempt+1} rejected; "
                  f"rebasing on origin/{_branch} and retrying ...", flush=True)
            subprocess.run(["git", "pull", "--rebase", "origin", _branch],
                           cwd=repo_root, capture_output=True, text=True,
                           timeout=300)
        if not pushed:
            print(f"[train] ⚠ git push failed after retries: "
                  f"{r_push.stderr.strip()[:300]}", flush=True)
    except subprocess.TimeoutExpired as e:
        print(f"[train] ⚠ LFS push timed out at: {e.cmd[:3]}... "
              f"(timeout={e.timeout}s)", flush=True)
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
    ap.add_argument("--ckpt_backend", default="gitlfs",
                    choices=["gitlfs", "drive", "local"],
                    help="Where checkpoints live. 'gitlfs' (default) pushes "
                         "each save to GitHub Git LFS. 'drive'/'local' just "
                         "keep them in --ckpt_dir (point that at a mounted "
                         "Google Drive folder, e.g. /content/drive/MyDrive/"
                         "neuroslm_ckpts, for Drive-backed storage) and skip "
                         "the GitHub push.")
    ap.add_argument("--keep_last_n_ckpt", type=int, default=3,
                    help="After each checkpoint save, delete obsolete checkpoints "
                         "from disk so only the N newest per stream remain. "
                         "Pass 0 to disable rotation. Default: 3.")
    ap.add_argument("--cross_stream_resume", action="store_true", default=True,
                    help="When `--resume latest` finds no checkpoint in the "
                         "current run's own stream, warm-start from the most "
                         "recent SAME-ARCHITECTURE checkpoint (same preset + "
                         "param count, any optimizer/mode). Lets a --meta "
                         "'full' run pick up the ablation run's checkpoints. "
                         "Model weights + memory are loaded; optimizer state "
                         "is skipped on a stream mismatch. Default ON.")
    ap.add_argument("--no_cross_stream_resume", dest="cross_stream_resume",
                    action="store_false",
                    help="Disable cross-stream warm-start (strict same-stream "
                         "resume only).")
    ap.add_argument("--prune_git", action="store_true", default=True,
                    help="When rotating, route deletions through `git rm` + "
                         "commit (local only — caller pushes). Default ON.")
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

    # ── Generalization / OOD knobs (override preset defaults) ────────────
    ap.add_argument("--dropout", type=float, default=None,
                    help="Override cfg.dropout (embedding + standard-block + "
                         "pre-head dropout). 0.1 narrows the train-vs-OOD gap; "
                         "0 restores legacy no-dropout. Default: preset value.")
    ap.add_argument("--weight_decay", type=float, default=None,
                    help="Override cfg.weight_decay (applied to 2-D matrices "
                         "only when decoupled WD is on). Default: preset value.")
    ap.add_argument("--lr_decay_steps", type=int, default=None,
                    help="Cosine LR-decay horizon (AdamW). The LR anneals to "
                         "lr*min_lr_ratio by this step. Set to the step you "
                         "plan to STOP at (e.g. 10000) so an early-stopped run "
                         "actually gets the annealing phase. Default: --steps.")
    ap.add_argument("--min_lr_ratio", type=float, default=None,
                    help="Cosine floor as a fraction of peak LR (e.g. 0.1). "
                         "Default: preset value (0.1).")
    ap.add_argument("--label_smoothing", type=float, default=None,
                    help="LM cross-entropy label smoothing. Default 0.0 (off; "
                         "can inflate raw perplexity).")
    ap.add_argument("--no_decoupled_wd", dest="decoupled_wd",
                    action="store_false", default=None,
                    help="Disable decoupled weight decay (decay ALL params, "
                         "legacy behavior).")

    # ── Fresh start ──────────────────────────────────────────────────────
    ap.add_argument("--fresh", action="store_true",
                    help="Start training from step 0 with a freshly-initialized "
                         "model: ignore any --resume / 'latest' checkpoint and "
                         "do NOT warm-start. Use after changing regularization / "
                         "LR schedule so the new, more efficient config trains "
                         "from scratch rather than inheriting the old run's "
                         "weights and optimizer state.")
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

    # ── Apply generalization-knob overrides onto the preset cfg ──────────
    if args.dropout is not None:
        cfg.dropout = args.dropout
    if args.weight_decay is not None:
        cfg.weight_decay = args.weight_decay
    if args.lr_decay_steps is not None:
        cfg.lr_decay_steps = args.lr_decay_steps
    if args.min_lr_ratio is not None:
        cfg.min_lr_ratio = args.min_lr_ratio
    if args.label_smoothing is not None:
        cfg.label_smoothing = args.label_smoothing
    if args.decoupled_wd is not None:   # only set when --no_decoupled_wd passed
        cfg.decoupled_wd = args.decoupled_wd
    print(f"[train] regularization: dropout={cfg.dropout} "
          f"weight_decay={cfg.weight_decay} decoupled_wd={cfg.decoupled_wd} "
          f"lr_decay_steps={cfg.lr_decay_steps or args.steps} "
          f"min_lr_ratio={cfg.min_lr_ratio} "
          f"label_smoothing={cfg.label_smoothing}", flush=True)

    # ── Fresh start: ignore any resume so the new config trains from 0 ───
    if args.fresh:
        if args.resume:
            print(f"[train] --fresh: ignoring --resume {args.resume!r}; "
                  f"starting from step 0 with freshly-initialized weights.",
                  flush=True)
        args.resume = None
        args.transfer = None
        args.cross_stream_resume = False
    # Gradient checkpointing is beneficial on both XLA and CUDA
    if should_use_gradient_checkpointing():
        cfg.gradient_checkpointing = True
    ctx_len = args.ctx or cfg.lang_ctx
    assert ctx_len <= cfg.lang_ctx

    # Topological Maturation Schedule (neural infancy → awakening)
    # ── Infancy stage (first 5,000 steps or until lm_loss < 7.5) ────────
    # - Re-entrant loops are physically present but chemically suppressed
    # - DA/NE forced to baseline 0.1 (no natural fluctuation)
    # - Auxiliary weights clamped at 0.001 (world/self models silent)
    # - Forces "Linguistic First" convergence (LM before global integration)
    #
    # ── Awakening stage (after infancy && lm_loss < 7.5) ────────────────
    # - DA/NE allowed to fluctuate naturally (homeostasis adjusts)
    # - Auxiliary weights ramp from 0.001 → config target
    _maturation_infancy_steps = 5000
    _maturation_lm_threshold = 7.5
    _maturation_awakened = False
    _aux_w_init = 0.001  # infancy weight (reduced from 0.01)
    _aux_w_target = 1.0
    _loss_ramp_threshold = 7.5  # trigger awakening at this threshold
    _loss_ramp_window = 100  # steps below threshold to trigger full ramping
    _loss_below_threshold_count = 0
    _lm_ema = None   # exponential moving average of LM loss (for gating)
    _ramp_started = False  # flag tracking if we've passed the threshold
    _ramp_start_step = None  # step at which the aux-loss ramp began (fix A)

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
    # One-time topology breakdown so you can verify the full bio stack is
    # actually constructed (not silently degenerating into baseline).
    try:
        print("[train] " + brain.topology_summary().replace("\n", "\n[train] "),
              flush=True)
    except Exception as _e:
        print(f"[train] topology_summary failed: {_e}", flush=True)

    # ── Optimizer ────────────────────────────────────────────────────────
    # Adafactor: factor-wise second moment — ~8× less optimizer memory than
    # AdamW, essential for 258M+ params on TPU HBM.  Falls back to AdamW
    # when --optimizer=adamw is passed (useful for debugging).
    named = list(brain.named_parameters())
    model_params = [p for n, p in named if not n.startswith('learned_opt.')]

    # RCC Bowtie Phase 3: parameter closure isolation.
    # When cfg.rcc_isolate_optimizer is True, partition params into trunk
    # (Adam-tracked) and bio (not Adam-tracked). Bio modules can then
    # mutate their own params in-place without poisoning Adam's momentum
    # state for trunk params. See brain.partition_trunk_bio_params() and
    # docs/architecture.md §5.6 (P3).
    if bool(getattr(cfg, 'rcc_isolate_optimizer', False)):
        _trunk_named, _bio_named = brain.partition_trunk_bio_params()
        _trunk_param_set = {id(p) for _, p in _trunk_named}
        # Restrict `named` to trunk-only BEFORE build_param_groups runs.
        _filtered = [(n, p) for n, p in named if id(p) in _trunk_param_set]
        _n_dropped = len(named) - len(_filtered)
        named = _filtered
        model_params = [p for n, p in named if not n.startswith('learned_opt.')]
        # Freeze the bio params so PyTorch doesn't waste memory tracking
        # their grads either (we never call backward on them via the LM
        # loss path, but explicit freeze prevents accidental grad
        # accumulation if some aux path leaks).
        for n, p in _bio_named:
            p.requires_grad_(False)
        print(f"[train] RCC P3 — parameter closure isolation ON: "
              f"trunk={len(_trunk_named)} params Adam-tracked, "
              f"bio={len(_bio_named)} params frozen (bio-side machinery "
              f"can still mutate them in-place via no_grad ops). "
              f"Dropped {_n_dropped} from Adam.", flush=True)

    # ── Decoupled weight decay ───────────────────────────────────────────
    # Apply weight decay only to 2-D weight matrices (Linear / embedding) and
    # exempt 1-D params (RMSNorm gains, biases, NT levels/vesicles). Decaying
    # those is a known generalization regression. Falls back to a single
    # undifferentiated group when cfg.decoupled_wd is False (legacy behavior).
    _decoupled = bool(getattr(cfg, "decoupled_wd", False))
    param_groups = build_param_groups(named, cfg.weight_decay, _decoupled)
    if _decoupled:
        print(f"[train] decoupled WD: {len(param_groups[0]['params'])} decayed "
              f"matrices, {len(param_groups[1]['params'])} exempt 1-D params "
              f"(wd={cfg.weight_decay})", flush=True)

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
                param_groups,
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
        optim = AdamW(param_groups, lr=cfg.lr,
                      weight_decay=cfg.weight_decay, betas=(0.9, 0.95))

    # Surprise-Gated Branching EMA — meta-optimizer that protects the
    # trunk from late-training divergence. See §7.5 in
    # docs/architecture.md and neuroslm/intelligence/branching_ema.py.
    branching_ema = None
    if getattr(cfg, 'use_branching_ema', False):
        from .intelligence.branching_ema import BranchingEMA
        branching_ema = BranchingEMA(
            brain,
            history_len=int(getattr(cfg, 'bema_history_len', 10)),
            gamma=float(getattr(cfg, 'bema_gamma', 0.5)),
            base_alpha_cap=float(getattr(cfg, 'bema_alpha_cap', 0.01)),
            update_every=int(getattr(cfg, 'bema_update_every', 1)),
            best_ema_alpha=float(getattr(cfg, 'bema_best_ema_alpha', 0.1)),
        )
        print(f"[train] Branching EMA enabled "
              f"(history={branching_ema.history.maxlen}, "
              f"gamma={branching_ema.gamma}, "
              f"alpha_cap={branching_ema.base_alpha_cap})", flush=True)

    meta_opt = None
    if args.meta and functional_call is None:
        print("[train] WARNING: torch.func.functional_call unavailable; "
              "disabling meta-training", flush=True)
        args.meta = False
    if args.meta and not cfg.baseline:
        # meta optimizer trains ONLY the learned optimizer's own params.
        # Geometry adapters are part of brain.language and are trained by the
        # MAIN optimizer in the standard update path below — including them
        # here too would double-step the same tensors with conflicting grads.
        meta_params = list(brain.learned_opt.parameters())
        meta_opt = AdamW(meta_params, lr=args.meta_lr)
        print("[train] --meta: the standard full-model update runs every "
              "step (all params train normally); meta-training of the "
              "learned optimizer is an additional non-destructive side "
              "channel. Drop --meta for ~same quality at higher throughput.",
              flush=True)

    start_step = 0
    resume_cross_stream = False   # set True by the same-arch warm-start path
    if args.resume:
        resume_path = args.resume
        if resume_path == "latest":
            import glob as _glob
            # Checkpoint streams are partitioned by (preset, optimizer,
            # baseline-or-not). `--resume latest` only matches files in the
            # current run's stream — an Adafactor checkpoint can never be
            # picked up by an AdamW resume, so they coexist on disk.
            _otag  = f"_{args.optimizer}"
            _bflag = "_baseline" if args.baseline else ""
            # Legacy prefix (pre-param-tag).  Kept so checkpoints written
            # before the `_<N>M` tag was added still resume cleanly.
            _prefix_legacy = f"neuroslm_{args.preset}{_otag}{_bflag}"
            # New-style pattern: neuroslm_<preset>_<N>M_<optimizer>[_baseline]_...
            import re as _re
            _ptag_re = _re.compile(
                rf"^neuroslm_{_re.escape(args.preset)}_(\d+)M{_re.escape(_otag)}"
                rf"{_re.escape(_bflag)}(?:_|$)")

            def _matches_stream(path: str) -> bool:
                name = os.path.basename(path)
                if name.startswith(_prefix_legacy):
                    # Non-baseline runs must not accidentally pick up baseline
                    # files (whose names extend the prefix with `_baseline`).
                    if not args.baseline and name.startswith(
                            f"neuroslm_{args.preset}{_otag}_baseline"):
                        return False
                    return True
                if _ptag_re.match(name):
                    return True
                return False

            # With --overwrite_ckpt prefer the fixed _latest file (either
            # naming convention; new takes precedence if both exist).
            import glob as _g2
            _latest_glob = _g2.glob(os.path.join(
                args.ckpt_dir, f"neuroslm_{args.preset}_*{_otag}{_bflag}_latest.pt"))
            _latest_legacy = os.path.join(args.ckpt_dir, f"{_prefix_legacy}_latest.pt")
            if args.overwrite_ckpt and _latest_glob:
                resume_path = max(_latest_glob, key=os.path.getmtime)
            elif args.overwrite_ckpt and os.path.exists(_latest_legacy):
                resume_path = _latest_legacy
            else:
                # Rank by the step number in the filename. Step-numbered files
                # always beat a `_latest.pt` of the same stream — otherwise an
                # older `_latest.pt` (overwritten at a low step) can shadow a
                # later step-numbered checkpoint just because its mtime is fresher.
                import re as _re
                _step_re = _re.compile(r"_(\d+)\.pt$")

                def _step_of(path: str) -> tuple:
                    name = os.path.basename(path)
                    if name.endswith("_latest.pt"):
                        # _latest files rank below any step-numbered file; among
                        # themselves, mtime breaks ties
                        return (0, os.path.getmtime(path))
                    m = _step_re.search(name)
                    step = int(m.group(1)) if m else 0
                    return (1, step)

                _lfs = os.path.join(
                    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                    "lfs_checkpoints")
                candidates = [f for f in _glob.glob(
                                os.path.join(args.ckpt_dir, "*.pt"))
                              if _matches_stream(f)]
                candidates.sort(key=_step_of)
                if not candidates:
                    candidates = [f for f in _glob.glob(os.path.join(_lfs, "*.pt"))
                                  if _matches_stream(f)]
                    candidates.sort(key=_step_of)
                resume_path = candidates[-1] if candidates else None

                # ── Cross-stream warm-start fallback ──────────────────────
                # No checkpoint in our OWN stream (e.g. a fresh --meta 'full'
                # run while only the ablation stream has checkpoints). Fall
                # back to the newest SAME-ARCHITECTURE checkpoint: same preset
                # + same param-count tag, any optimizer/mode, but NOT a
                # `_baseline` vanilla checkpoint (its lang-layer count differs
                # → shape-incompatible). We flag this so the loader skips the
                # optimizer state (likely a different optimizer) and keeps
                # only the model weights + memory.
                # Cross-stream is DISABLED for baseline runs: the vanilla
                # baseline has a unique architecture (N stacked layers, no
                # bio modules) and — since the param-parity fix made it the
                # same param-count as the full model — the _<N>M tag can no
                # longer distinguish them. A baseline run must only ever
                # resume a `_baseline` checkpoint of its own stream, never a
                # full-model checkpoint. So we only warm-start full runs,
                # matching full (non-`_baseline`) checkpoints.
                resume_cross_stream = False
                if (resume_path is None
                        and getattr(args, 'cross_stream_resume', True)
                        and not args.baseline):
                    _ptag = f"_{round(n_params / 1e6)}M"

                    def _arch_match(path: str) -> bool:
                        name = os.path.basename(path)
                        return (name.startswith(f"neuroslm_{args.preset}{_ptag}_")
                                and "_baseline" not in name
                                and not name.endswith("_latest.pt"))

                    for _dir in (args.ckpt_dir, _lfs):
                        arch_cands = [f for f in _glob.glob(os.path.join(_dir, "*.pt"))
                                      if _arch_match(f)]
                        # also allow _latest.pt of same arch as a last resort
                        arch_latest = [f for f in _glob.glob(os.path.join(
                            _dir, f"neuroslm_{args.preset}{_ptag}_*_latest.pt"))
                            if "_baseline" not in os.path.basename(f)]
                        arch_cands = arch_cands + arch_latest
                        arch_cands.sort(key=_step_of)
                        if arch_cands:
                            resume_path = arch_cands[-1]
                            resume_cross_stream = True
                            break

                if resume_path and resume_cross_stream:
                    print(f"[train] no same-stream checkpoint — cross-stream "
                          f"warm-start from {os.path.basename(resume_path)} "
                          f"(model weights + memory only; optimizer state "
                          f"skipped)", flush=True)
                elif resume_path:
                    print(f"[train] auto-found latest checkpoint: {resume_path}", flush=True)
                else:
                    print(f"[train] no checkpoint found for stream {_prefix_legacy}* "
                          f"— training from scratch", flush=True)

        # A Git LFS pointer (when `git lfs pull` didn't materialise the blob)
        # is a tiny text file starting with "version https://git-lfs…".
        # torch.load on it crashes with "invalid load key, 'v'" and, under
        # the restart loop, burns money crash-looping. Detect + skip it.
        def _is_lfs_pointer(p):
            try:
                with open(p, "rb") as fh:
                    return fh.read(48).startswith(b"version https://git-lfs")
            except Exception:
                return False

        if resume_path and Path(resume_path).exists() and _is_lfs_pointer(resume_path):
            print(f"[train] ⚠ {os.path.basename(str(resume_path))} is an unfetched "
                  f"Git LFS pointer (run `git lfs pull`). Training from scratch.",
                  flush=True)
            resume_path = None

        if resume_path and Path(resume_path).exists():
            try:
                ckpt = torch.load(resume_path, map_location=device, weights_only=False)
                # Adapter-rank-aware load: resizes BDNF-grown kernels and
                # drops any genuinely incompatible tensors so a grown
                # checkpoint resumes instead of crashing on shape mismatch.
                missing, unexpected, _dropped = _load_compatible(
                    brain, ckpt["model"], label=os.path.basename(str(resume_path)))
                # With --overwrite_ckpt, treat large key-mismatch as architecture change
                # and start fresh rather than loading a partly-matching checkpoint.
                _mismatch = len(missing) + len(unexpected) + len(_dropped)
                if args.overwrite_ckpt and _mismatch > 20:
                    print(f"[train] ⚠ architecture mismatch ({_mismatch} keys differ) "
                          f"— starting fresh (--overwrite_ckpt)", flush=True)
                    start_step = 0
                else:
                    if "optim" in ckpt and not resume_cross_stream:
                        # Checkpoints are partitioned by optimizer in the
                        # filename (see save block + resume glob), so a class
                        # mismatch here only happens when --resume points at
                        # an explicit path the user chose to override.
                        try:
                            optim.load_state_dict(ckpt["optim"])
                        except Exception as e:
                            print(f"[train] could not restore optimizer state: {e}",
                                  flush=True)
                    elif resume_cross_stream:
                        # Cross-stream warm-start: the source checkpoint likely
                        # used a different optimizer (e.g. ablation=AdamW vs
                        # full=Adafactor). Keep the freshly-built optimizer and
                        # only inherit the model weights + maturity + memory.
                        print("[train] cross-stream: optimizer state NOT loaded "
                              "(fresh optimizer, inherited model weights)",
                              flush=True)
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
                # NEVER hard-fail on a bad/corrupt/partial checkpoint — under
                # the vast restart loop that's an infinite crash-loop burning
                # GPU time. Warn loudly and train from scratch instead.
                print(f"[train] ⚠ could not load checkpoint ({_load_err}) "
                      f"— training from scratch", flush=True)
                start_step = 0
    elif args.transfer and Path(args.transfer).exists():
        ckpt = torch.load(args.transfer, map_location=device, weights_only=False)
        brain.load_partial(ckpt["model"])
        print(f"[train] transferred matching tensors from {args.transfer}", flush=True)

    # Reset MAT to cold start when training fresh. Brain defaults
    # `maturity=1.0` so tests/inference see a mature graph; training must
    # begin from the L_random plateau. If we resumed from a checkpoint
    # (start_step > 0), the saved `maturity` buffer is already loaded —
    # leave it intact so the maturation schedule continues seamlessly.
    if not cfg.baseline and start_step == 0 and hasattr(brain, "maturity"):
        brain.maturity.zero_()
        brain._infancy = True

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
    running_gnorm = 0.0
    n_obs = 0
    gnorm = 0.0  # last optimizer step's grad norm (set in non-meta branch)
    # Gradient-spike rejection state (fix D): EMA of gnorm + skip counter.
    _gnorm_ema = None
    _n_spikes_skipped = 0
    _spike_factor = float(getattr(cfg, 'grad_spike_factor', 0.0))
    _spike_warmup = int(getattr(cfg, 'grad_spike_warmup', 100))

    # ── Best-checkpoint tracking ─────────────────────────────────────────
    # Maintain a single `*_best.pt` holding the lowest-loss weights seen so
    # far. It is overwritten ONLY when a strictly lower smoothed LM loss is
    # reached, and is never pruned (the `_best` stem has no trailing-digit
    # step, so prune_old_checkpoints' regex skips it). We compare on an EMA
    # of the LM loss so one noisy step can't trigger a spurious "best".
    # The best loss is persisted in a tiny `*_best.json` sidecar so we can
    # read it across restarts without loading the 400+ MB checkpoint.
    best_lm_ema: Optional[float] = None
    _best_tag   = "" if args.mode == "text" else f"_{args.mode}"
    _best_bflag = "_baseline" if cfg.baseline else ""
    _best_otag  = f"_{args.optimizer}"
    _best_ptag  = f"_{round(n_params / 1e6)}M"
    best_fname  = (f"neuroslm_{args.preset}{_best_ptag}{_best_otag}"
                   f"{_best_bflag}{_best_tag}_best.pt")
    best_loss = float("inf")
    try:
        import glob as _bglob
        _meta_candidates = [os.path.join(os.path.abspath(args.ckpt_dir),
                                          best_fname.replace(".pt", ".json"))]
        _lfs_best = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            "lfs_checkpoints", best_fname.replace(".pt", ".json"))
        _meta_candidates.append(_lfs_best)
        for _bm in _meta_candidates:
            if os.path.exists(_bm):
                with open(_bm) as _bf:
                    _bj = json.load(_bf)
                best_loss = float(_bj.get("best_loss", float("inf")))
                print(f"[train] best-ckpt: prior best_loss={best_loss:.4f} "
                      f"(step {_bj.get('step','?')}) from {os.path.basename(_bm)}",
                      flush=True)
                break
    except Exception as _be:
        print(f"[train] best-ckpt: could not read prior best ({_be})", flush=True)

    total_steps = args.steps
    if start_step >= total_steps:
        print(f"[train] already at step {start_step} >= {total_steps}; nothing to do.", flush=True)
        return
    print(f"[train] >>> training loop starting: steps {start_step}..{total_steps} <<<", flush=True)

    for step in range(start_step, total_steps):
        # Drive Smooth-Gated-Bus temporal schedule. No-op when SGB is
        # disabled (`cfg.use_smooth_gated_bus=False`). See §5.6 in
        # docs/architecture.md and neuroslm/modules/smooth_gated_bus.py.
        if not cfg.baseline and hasattr(brain, "set_training_step"):
            brain.set_training_step(step)
        # Keep the NT system's internal step aligned with the train loop so
        # the 5-HT hard cap (warmup window < 20k steps) phases correctly
        # across resumes / restarts. No-op for the baseline path.
        if not cfg.baseline and hasattr(brain, "transmitters"):
            brain.transmitters.set_train_step(step)
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
            _decay_total = getattr(cfg, "lr_decay_steps", 0) or total_steps
            _min_ratio = getattr(cfg, "min_lr_ratio", 0.0)
            _lr_now = cosine_lr(step, cfg.warmup_steps, _decay_total, cfg.lr,
                                min_ratio=_min_ratio)
            for pg in optim.param_groups:
                # Per-group LR scaling preserved (e.g. a no-decay group can
                # carry its own base via pg.get("lr_scale", 1.0)).
                pg["lr"] = _lr_now * pg.get("lr_scale", 1.0)

        # --- Memory system integration ---
        # 1. Record episodic memory for this batch (deferred to CPU; non-blocking).
        # Skipped pre-awakening — the content vector is meaningful but the
        # nt_state and downstream consolidation depend on a learning network.
        # The tokenizer.decode + .cpu().numpy() round-trip also costs wall time.
        # Gate is now the continuous MAT (pre-update reads the previous step).
        _prev_maturity = (brain.maturity_scalar()
                          if (not cfg.baseline and hasattr(brain, "maturity_scalar"))
                          else 0.0)
        _in_infancy_step = _prev_maturity < 0.3
        if not cfg.baseline and not _in_infancy_step:
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

        # ── Maturity-Driven Topological Maturation ─────────────────────────
        # We have replaced the hard "step < 5000" infancy gate with a continuous
        # MAT (Maturity Index): M_t = clamp(1 - L_lm / L_random, 0, 1).
        # Smoothed via EMA on `brain.maturity`. Modules consume this directly
        # to fade in (expert cortices, MoD compute, ε-routing strength).
        #
        # Awakening criterion (auxiliary-loss ramp activation):
        #   brain.maturity > 0.3  AND  lm_loss < 7.5
        _lm_now = float(out["lm_loss"].item())
        if _lm_ema is None:
            _lm_ema = _lm_now
        _lm_ema = 0.99 * _lm_ema + 0.01 * _lm_now

        # Update MAT (continuous maturation signal). Baseline runs skip this
        # because the bio brain isn't built.
        if not cfg.baseline and hasattr(brain, "update_maturity"):
            _maturity = brain.update_maturity(_lm_now)
        else:
            _maturity = 0.0

        # Awakening transition: M > 0.3 AND lm_loss < 7.5 (sustained drop)
        if (not _maturation_awakened
                and _maturity > 0.3
                and _lm_now < _maturation_lm_threshold):
            _maturation_awakened = True
            if not cfg.baseline:
                if hasattr(brain, "set_awakened"):
                    try:
                        brain.set_awakened(True)
                    except Exception as _e:
                        print(f"[train] ⚠ brain.set_awakened failed: {_e}", flush=True)
                print(f"[train] 🧠 Neural awakening at step {step+1}: "
                      f"maturity={_maturity:.3f}, lm_loss={_lm_now:.4f}, "
                      f"entering growth phase", flush=True)

        # Pre-awakening: damp GWS ignition by raising the threshold so the
        # broadcast doesn't pin at 1.0 while modules are still random. We
        # interpolate the threshold linearly with maturity: at M=0 → 1.2
        # (hard to ignite); at M=0.3 → 0.8 (config default).
        if not cfg.baseline and not _maturation_awakened and hasattr(brain, 'gws'):
            _thresh = 1.2 - (1.2 - 0.8) * min(1.0, _maturity / 0.3)
            brain.gws.slot_thresholds.data = torch.full_like(
                brain.gws.slot_thresholds, _thresh)

        # Track when we've sustained below-threshold loss (only after awakening)
        if _maturation_awakened:
            if _lm_now < _loss_ramp_threshold:
                _loss_below_threshold_count += 1
            else:
                _loss_below_threshold_count = 0

            # (fix A, revised) Maturity-GATED aux weight: aux strengthens only
            # as the LM genuinely matures (MAT lo→hi), and backs off
            # automatically if the LM regresses (MAT falls). Self-correcting,
            # so it can't slam aux to full on an immature LM the way a step
            # schedule did.
            _mat_now = (brain.maturity_scalar()
                        if hasattr(brain, 'maturity_scalar') else 0.0)
            _aux_ramp = maturity_aux_gate(
                _mat_now,
                float(getattr(cfg, 'aux_gate_mat_lo', 0.50)),
                float(getattr(cfg, 'aux_gate_mat_hi', 0.65)))
        else:
            # Pre-awakening: auxiliary weights pinned at 0.001
            _aux_ramp = 0.0

        _aux_w = _aux_w_init + (_aux_w_target - _aux_w_init) * _aux_ramp
        # Single per-step gate that scales ALL auxiliary losses uniformly
        # (world / motor / cpc / phi / kl / novel / pred_coding / id_drift).
        # Pre-awakening this is ~0.001, ramping to 1.0 after lm_loss stabilises.
        # Trophic/BDNF structural updates are suppressed while _infancy=True
        # (now derived from MAT < 0.3 inside Brain.update_maturity).
        if not cfg.baseline:
            brain._aux_w_scale = float(_aux_w)
            # _infancy is now set by Brain.update_maturity (MAT < 0.3).

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

            # Update ONLY the learned optimizer (non-destructive side channel).
            # The real model update happens in the standard full-model path
            # below.  The meta path NO LONGER hijacks the model update — the
            # old code applied learned-optimizer-transformed grads to
            # brain.language ONLY and called optim.step() on just those
            # params, so a --meta run trained 25.7M of 107M params on 1/16
            # the data through a random-init optimizer.  That was the entire
            # full-vs-ablation performance gap.
            meta_opt.zero_grad(set_to_none=True)
            meta_loss.backward()
            meta_opt.step()
            brain.learned_opt.reset_state()

        # ── Standard full-model update — ALWAYS runs ──────────────────────
        # Trains EVERY parameter (LM trunk + full bio/bowtie stack) through
        # forward_lm with proper gradient accumulation and the real
        # optimizer.  Previously gated behind `if not args.meta`, so a
        # --meta run skipped it and only the language module trained.
        if True:
            ga = max(args.grad_accum, 1)
            optim.zero_grad(set_to_none=True)
            # bfloat16 has fp32-equivalent exponent range → no loss scaling needed
            (loss / ga).backward()
            total_lm_loss = float(out["lm_loss"].item())
            total_loss    = float(loss.item())            # ← bug fix: track total
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
                total_loss    += float(ga_loss.item())     # ← bug fix
            # (fix D) Gradient-spike rejection: skip the step when the pre-clip
            # grad norm blows past `_spike_factor × EMA(gnorm)`. Active only
            # after warmup so the EMA is established. Prevents a single spiked
            # batch from kicking the model into the post-awakening divergence.
            _spike_thr = None
            if (_spike_factor > 0 and _gnorm_ema is not None
                    and step >= _spike_warmup):
                _spike_thr = _spike_factor * _gnorm_ema
            gnorm = optimizer_step(optim, brain.parameters(), cfg.grad_clip,
                                   skip_threshold=_spike_thr)
            _spiked = _spike_thr is not None and gnorm > _spike_thr
            if _spiked:
                _n_spikes_skipped += 1
                if _n_spikes_skipped <= 20 or _n_spikes_skipped % 50 == 0:
                    print(f"[train] ⚡ grad spike at step {step+1}: gnorm "
                          f"{gnorm:.2f} > {_spike_thr:.2f} "
                          f"({_spike_factor}×EMA) — step skipped "
                          f"(#{_n_spikes_skipped})", flush=True)
            else:
                # Update the EMA only from accepted (non-spike) steps so the
                # threshold tracks the normal gradient regime, not the spikes.
                _gnorm_ema = (gnorm if _gnorm_ema is None
                              else 0.9 * _gnorm_ema + 0.1 * gnorm)
            mark_step()   # XLA: flush graph; no-op on CUDA/CPU
            # Display values must reflect the FULL grad-accum window — the old
            # code averaged lm_loss across micro-batches but used only the
            # first micro-batch's `loss`, which made the printed "loss" column
            # swing wildly while "lm" stayed smooth (false "surge" pattern
            # observed in the awakening logs).
            loss = torch.tensor(total_loss / ga)
            out["loss"]    = loss
            out["lm_loss"] = torch.tensor(total_lm_loss / ga)

        # 3. Tag memory with reward/insight (mesolimbic) — paired with the
        # episodic record above, so gated identically.
        if not cfg.baseline and not _in_infancy_step:
            reward = float(out["learning_gain"][0].item()) if "learning_gain" in out else 0.0
            da_level = float(brain.transmitters.vector()[0, 0].item()) if hasattr(brain, 'transmitters') else 0.5
            brain.tag_memory(len(brain.episodic.buffer)-1, reward, da_level=da_level)

        # 4. Consolidate and update narratives every 500 steps (post-infancy only)
        if not cfg.baseline and not _in_infancy_step and (step + 1) % 500 == 0:
            brain.consolidate_memory()
            brain.update_narratives()

        # Slow homeostatic regulation of NT baselines & gains.
        # Suppressed during infancy — the controller's targets (NT mean/std,
        # grad-norm bands) are calibrated for a learning network, not a
        # random-init one. Letting it run during infancy drives NT to ceiling.
        if not cfg.baseline and _maturation_awakened:
            brain.homeostasis.observe(brain.transmitters,
                                      float(out["lm_loss"].item()), float(gnorm))

        running_loss += float(loss.item())
        _lm_now = float(out["lm_loss"].item())
        running_lm += _lm_now
        running_gnorm += float(gnorm)
        n_obs += 1

        # EMA of LM loss for the best-checkpoint comparison (noise-robust).
        if math.isfinite(_lm_now):
            best_lm_ema = (_lm_now if best_lm_ema is None
                           else 0.98 * best_lm_ema + 0.02 * _lm_now)

        # ── Branching EMA per-step hook ──────────────────────────────
        # Track current PPL, gate-update the stable shadow, collapse to
        # params_best on catastrophic spike. Fixes the synth-v1 pathology
        # where training hit ppl 59 at step 3800 then drifted to ppl
        # 200-400 from step 4800 onwards and never recovered the basin.
        # No-op when cfg.use_branching_ema=False. See docs §7.5.
        if branching_ema is not None and math.isfinite(_lm_now):
            _bema_ppl_now = math.exp(min(20.0, _lm_now))
            branching_ema.maybe_update(brain, ppl=_bema_ppl_now, step=step)
            _did_collapse = branching_ema.maybe_collapse_to_best(
                brain, current_ppl=_bema_ppl_now,
                trigger_ratio=float(getattr(cfg, 'bema_collapse_ratio', 3.0)),
            )
            if _did_collapse:
                print(f"[train] BEMA COLLAPSE at step {step+1}: "
                      f"ppl {_bema_ppl_now:.1f} > "
                      f"{getattr(cfg, 'bema_collapse_ratio', 3.0)}x "
                      f"best_ema_ppl {branching_ema._best_ema_ppl:.1f}; "
                      f"trunk reverted to best snapshot "
                      f"(#{branching_ema._n_collapses})", flush=True)

        if (step + 1) % args.log_every == 0:
            dt = time.time() - t0
            avg = running_loss / n_obs
            avg_lm = running_lm / n_obs
            avg_gnorm = running_gnorm / n_obs
            ppl = math.exp(min(avg_lm, 20))
            tok_per_s = args.log_every * args.batch_size * max(args.grad_accum, 1) * ctx_len / max(dt, 1e-3)
            _raw_lr = optim.param_groups[0].get('lr')
            _lr_str = f"{_raw_lr:.2e}" if _raw_lr is not None else "auto"

            # Oscillation spectrum (multi-band: delta/theta/gamma) — available in all modes
            osc_str = ""
            if hasattr(brain, 'oscillation_tracker'):
                try:
                    osc = brain.oscillation_tracker.compute_spectrum()
                    osc_str = f" | osc[δ={osc.delta:.3f} θ={osc.theta:.3f} γ={osc.gamma:.3f}]"
                except Exception as e:
                    # Log oscillation compute errors for debugging
                    import traceback
                    print(f"[train] ⚠ oscillation compute failed: {e}", flush=True)
                    if not cfg.baseline:
                        traceback.print_exc()

            if cfg.baseline:
                print(f"step {step+1:5d} | loss {avg:.4f} | lm {avg_lm:.4f} "
                      f"| ppl {ppl:.1f} | gnorm {avg_gnorm:.3f} | lr {_lr_str} "
                      f"| {tok_per_s:.0f} tok/s | BASELINE{osc_str}", flush=True)
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

                print(f"step {step+1:5d} | loss {avg:.4f} | lm {avg_lm:.4f} "
                      f"| ppl {ppl:.1f} | gnorm {avg_gnorm:.3f} | lr {_lr_str} "
                      f"| {tok_per_s:.0f} tok/s "
                      f"| Φ {phi:.3f} | λ₁ {fid:.3f} | ign {ign:.2f} "
                      f"| mesoLG {lg:.2f} "
                      f"| troph {t_act}/{t_tot} μ{t_mu:.2f} "
                      f"| NT[{nt_str}]{osc_str}", flush=True)
            running_loss = running_lm = running_gnorm = 0.0
            n_obs = 0
            t0 = time.time()

        if (step + 1) % args.save_every == 0 or (step + 1) == total_steps:
            tag = "" if args.mode == "text" else f"_{args.mode}"
            bflag = "_baseline" if cfg.baseline else ""
            otag = f"_{args.optimizer}"  # e.g. _adafactor / _adamw — partitions
                                          # checkpoint streams so an Adafactor
                                          # save and an AdamW save coexist on
                                          # disk without ever clashing.
            # Encode param count so checkpoints across config edits
            # (e.g. SRC-TEH on/off, d_hidden bumps, expert resizes) are
            # disambiguated on disk and the resume path can refuse to
            # cross-load a file whose param shape no longer matches.
            ptag = f"_{round(n_params / 1e6)}M"
            if args.overwrite_ckpt:
                # Fixed filename — overwritten every save, keeps LFS storage constant.
                # Step number is stored inside the file so resume still works.
                fname = f"neuroslm_{args.preset}{ptag}{otag}{bflag}_latest.pt"
            else:
                fname = f"neuroslm_{args.preset}{ptag}{otag}{bflag}{tag}_{step+1}.pt"
            path = Path(os.path.abspath(args.ckpt_dir)) / fname
            path.parent.mkdir(parents=True, exist_ok=True)
            save_dict = {
                "model": brain.state_dict(),
                "optim": optim.state_dict(),
                "optim_class": type(optim).__name__,
                "cfg": cfg.__dict__,
                "step": step + 1,
                "preset": args.preset,
            }
            if not cfg.baseline:
                save_dict["trophic_stats"] = brain.trophic.stats()
            torch.save(save_dict, path)
            if not cfg.baseline:
                # ── Save portable memory checkpoint (.mem) ──
                # Mirror the .pt stem (including optimizer + baseline + mode
                # + step tags) so the resume logic's `.replace('.pt', '.mem')`
                # always finds the matching memory file.
                try:
                    mem_path = path.with_suffix(".mem")
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

            # ── Checkpoint upload ──
            # Default: push to Git LFS. With --ckpt_backend drive/local the
            # checkpoint just stays in --ckpt_dir (point that at a mounted
            # Google Drive folder for Drive-backed storage) and we skip the
            # GitHub push entirely.
            if getattr(args, "ckpt_backend", "gitlfs") == "gitlfs":
                push_checkpoint_to_lfs(str(path))
            else:
                print(f"[train] checkpoint kept in {path.parent} "
                      f"(backend={args.ckpt_backend}, no GitHub push)", flush=True)

            # ── Best checkpoint (lowest smoothed LM loss; never pruned) ───
            # Overwrite `*_best.pt` only when the EMA LM loss improves. The
            # best loss lives in a tiny `*_best.json` sidecar so resume can
            # read it without loading the full checkpoint. The `_best` stem
            # has no trailing-digit step, so rotation never deletes it.
            if (not args.overwrite_ckpt) and best_lm_ema is not None \
                    and best_lm_ema < best_loss:
                prev_best = best_loss
                best_loss = float(best_lm_ema)
                best_path = path.parent / best_fname
                try:
                    best_dict = dict(save_dict)
                    best_dict["best_loss"] = best_loss
                    torch.save(best_dict, best_path)
                    with open(str(best_path).replace(".pt", ".json"), "w") as _bjf:
                        json.dump({"best_loss": best_loss, "step": step + 1,
                                   "lm_ema": best_lm_ema}, _bjf)
                    if not cfg.baseline:
                        try:
                            brain.save_memory_checkpoint(best_path.with_suffix(".mem"))
                        except Exception as _bme:
                            print(f"[train] best-ckpt memory save failed: {_bme}",
                                  flush=True)
                    _prev_str = (f"{prev_best:.4f}" if math.isfinite(prev_best)
                                 else "inf")
                    print(f"[train] ★ new BEST checkpoint @ step {step+1}: "
                          f"lm_ema {best_loss:.4f} (prev {_prev_str}) "
                          f"→ {best_path.name}", flush=True)
                    if getattr(args, "ckpt_backend", "gitlfs") == "gitlfs":
                        push_checkpoint_to_lfs(str(best_path))
                except Exception as _bse:
                    print(f"[train] best-ckpt save failed: {_bse}", flush=True)
                    best_loss = prev_best   # don't lose the prior best on error

            # ── Checkpoint rotation (keep last N per stream) ──────────────
            # Without rotation the repo grows by ~430 MiB per save.  We
            # delete obsolete .pt + .mem + .mem.json + .dna.json companions
            # AFTER the new save has been pushed so we never sit in a state
            # with fewer than (keep) good checkpoints on disk.  Rotation is
            # best-effort; a failure here must not crash the training loop.
            if (not args.overwrite_ckpt) and (args.keep_last_n_ckpt or 0) > 0:
                try:
                    from .tools.prune_ckpts import prune_old_checkpoints
                    _gitlfs = (getattr(args, "ckpt_backend", "gitlfs") == "gitlfs")
                    _scan_dirs = [Path(os.path.abspath(args.ckpt_dir))]
                    if _gitlfs:
                        # Also scan the lfs_checkpoints mirror that the GitHub
                        # push writes to; git rm the obsolete tracked blobs.
                        _lfs_dir = Path(os.path.dirname(os.path.dirname(
                            os.path.abspath(__file__)))) / "lfs_checkpoints"
                        if _lfs_dir != _scan_dirs[0] and _lfs_dir.exists():
                            _scan_dirs.append(_lfs_dir)
                    # Drive/local backend: ckpt_dir isn't a git repo, so prune
                    # via plain unlink (prune_old_checkpoints auto-detects the
                    # absence of a git work tree and falls back to unlink).
                    prune_old_checkpoints(
                        _scan_dirs,
                        keep=int(args.keep_last_n_ckpt),
                        use_git=bool(args.prune_git) and _gitlfs,
                        verbose=True,
                    )
                except Exception as _prune_err:
                    print(f"[train] checkpoint rotation skipped: {_prune_err}",
                          flush=True)

    print("[train] done.", flush=True)


if __name__ == "__main__":
    main()
