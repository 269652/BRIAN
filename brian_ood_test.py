"""BRIAN — Quick OOD Perplexity Test (Colab T4-ready).

Runs WikiText-103 (test) PPL on a BRIAN/NeuroSLM checkpoint and compares
it to an in-distribution PPL sample from the training stream. Designed for
Google Colab T4 (≈15 GB VRAM): no_grad / inference_mode, empty_cache on
each batch, cap on OOD windows + train batches.

Colab prep cell (uncomment if needed):
    # !pip install datasets tqdm

Usage:
    python brian_ood_test.py                     # auto-pick newest checkpoint
    python brian_ood_test.py --step 6580         # require a specific step
    python brian_ood_test.py --checkpoint /path/to.pt
    python brian_ood_test.py --max_ood_windows 400 --max_train_batches 200
"""
from __future__ import annotations

import argparse
import glob
import json
import math
import os
import re
import sys
import time
from pathlib import Path
from typing import List, Optional, Tuple

import torch


# ── Step 1: locate the checkpoint ──────────────────────────────────────────

_CKPT_DIRS = [
    "./checkpoints", "./ckpt", "./runs", "./output",
    "./lfs_checkpoints", "./checkpoints_baseline",
    ".",
]
_STEP_RE = re.compile(r"_(\d+)\.pt$")


def find_checkpoints(want_step: Optional[int] = None) -> List[Tuple[Optional[int], str]]:
    """Search common dirs for .pt files. Returns list of (step, path) ranked
    by step ascending; if `want_step` given, restricts to that step."""
    hits: List[Tuple[Optional[int], str]] = []
    for d in _CKPT_DIRS:
        if not os.path.isdir(d):
            continue
        for p in glob.glob(os.path.join(d, "*.pt")):
            m = _STEP_RE.search(os.path.basename(p))
            step = int(m.group(1)) if m else None
            if want_step is not None and step != want_step:
                continue
            hits.append((step, p))
    hits.sort(key=lambda x: (x[0] is None, x[0] or -1))
    return hits


def is_lfs_pointer(path: str) -> bool:
    try:
        with open(path, "rb") as fh:
            return fh.read(48).startswith(b"version https://git-lfs")
    except Exception:
        return False


# ── Step 2: load the model ─────────────────────────────────────────────────

def load_brain(ckpt_path: str, device: torch.device):
    """Rebuild Brain from the cfg saved IN the checkpoint, then load weights."""
    sys.path.insert(0, os.getcwd())
    from neuroslm.brain import Brain
    from neuroslm.config import BrainConfig, PRESETS
    from neuroslm.tokenizer import Tokenizer

    tok = Tokenizer()
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)

    cfg = BrainConfig()
    if isinstance(ckpt, dict) and "cfg" in ckpt and isinstance(ckpt["cfg"], dict):
        valid = set(cfg.__dict__.keys())
        for k, v in ckpt["cfg"].items():
            if k in valid:
                setattr(cfg, k, v)
    else:
        cfg = PRESETS.get("large", lambda: cfg)()
    cfg.vocab_size = tok.vocab_size

    brain = Brain(cfg).to(device)

    # ── Patch per-batch neurotransmitter state ───────────────────────────
    # transmitters.level / transmitters.vesicles are saved at the TRAIN batch
    # size (e.g. [32, 7]) but a freshly-built eval Brain has [1, 7]. Without
    # this, the rank-aware loader shape-skips them and the model runs on
    # random-init NT state, which modulates attention temperature/gain and
    # badly inflates perplexity. Collapse the batch dim by averaging so the
    # eval model inherits the trained resting NT levels.
    sd = ckpt["model"] if isinstance(ckpt, dict) and "model" in ckpt else ckpt
    model_sd = brain.state_dict()
    for key in ("transmitters.level", "transmitters.vesicles"):
        if key in sd and key in model_sd and sd[key].shape != model_sd[key].shape:
            ck_t, m_t = sd[key], model_sd[key]
            if (ck_t.dim() == m_t.dim() and ck_t.dim() >= 1
                    and m_t.shape[0] == 1 and ck_t.shape[0] > 1
                    and ck_t.shape[1:] == m_t.shape[1:]):
                sd[key] = ck_t.mean(dim=0, keepdim=True)
                print(f"[patch] {key}: {tuple(ck_t.shape)} -> {tuple(sd[key].shape)} "
                      f"(mean over batch dim)")
            else:
                print(f"[patch] {key}: shape {tuple(ck_t.shape)} vs model "
                      f"{tuple(m_t.shape)} — left as-is (unexpected layout)")

    # Reuse the train-time adapter-rank-aware loader for BDNF-grown ckpts.
    try:
        from neuroslm.train import _load_compatible
        _load_compatible(brain, sd, label=os.path.basename(ckpt_path))
    except Exception:
        brain.load_state_dict(sd, strict=False)
    brain.eval()
    return brain, tok, cfg, ckpt.get("step", "?") if isinstance(ckpt, dict) else "?"


# ── Step 4: sliding-window OOD PPL ─────────────────────────────────────────

@torch.inference_mode()
def compute_ppl_sliding(brain, tok, texts: List[str], ctx_len: int,
                        stride: int, batch_size: int,
                        max_windows: Optional[int] = None) -> Tuple[float, int]:
    """Mean cross-entropy → PPL over a list of texts via a strided sliding
    window (the canonical HF fixed-length-model recipe).

    The corpus is encoded into one long token stream. Each window of up to
    `ctx_len` tokens advances by `stride`. To avoid double-counting the
    overlap, each window scores ONLY its genuinely-new tokens — those past
    the previous window's end (`trg_len = end - prev_end`). The very first
    window scores all of its predictable tokens.

    Correctness details handled here:
      • `trg_len` is tracked via `prev_end` (NOT a fixed `stride`), so the
        first and last windows are accounted exactly.
      • The scored region is placed relative to each row's REAL length, so
        right-padding a short final window never leaks loss onto pad tokens.
      • Targets are the next-token shift; the last `n_new` target positions
        of the real sequence are the new tokens' predictions.
    """
    from tqdm.auto import tqdm

    device = next(brain.parameters()).device
    # Encode the whole corpus into one stream.
    all_ids: List[int] = []
    for t in texts:
        if not isinstance(t, str) or not t.strip():
            continue
        all_ids.extend(tok.encode(t))
    if len(all_ids) < 2:
        return float("nan"), 0

    # Build (start, end, trg_len) windows. trg_len = new tokens vs prev window.
    windows: List[Tuple[int, int, int]] = []
    prev_end = 0
    begin = 0
    n = len(all_ids)
    while begin < n:
        end = min(begin + ctx_len, n)
        trg_len = end - prev_end
        windows.append((begin, end, trg_len))
        prev_end = end
        if end == n:
            break
        begin += stride
    if max_windows:
        windows = windows[:max_windows]

    total_loss, total_tokens = 0.0, 0

    for b_start in tqdm(range(0, len(windows), batch_size), desc="OOD PPL"):
        batch = windows[b_start:b_start + batch_size]
        max_len = max(e - s for s, e, _ in batch)
        bsz = len(batch)
        ids_batch = torch.zeros(bsz, max_len, dtype=torch.long)
        # Target-valid mask aligned to the shifted targets (length max_len-1).
        tgt_valid = torch.zeros(bsz, max_len - 1, dtype=torch.bool)
        for i, (s, e, trg_len) in enumerate(batch):
            seq = all_ids[s:e]
            L = len(seq)                          # real (unpadded) length
            ids_batch[i, :L] = torch.tensor(seq, dtype=torch.long)
            # Predictable targets for this row: indices [0 .. L-2] (length L-1).
            # Score only the last n_new of them — the new tokens. Clamp so the
            # first window (trg_len == L) scores all L-1 predictions.
            n_new = min(trg_len, L - 1)
            if n_new > 0:
                tgt_valid[i, (L - 1 - n_new):(L - 1)] = True

        ids_batch = ids_batch.to(device)
        tgt_valid = tgt_valid.to(device)

        ids_in = ids_batch[:, :-1]
        tgt    = ids_batch[:, 1:].contiguous()

        out = brain.forward_lm(ids_in, tgt)
        logits = out["logits"]
        loss = torch.nn.functional.cross_entropy(
            logits.reshape(-1, logits.size(-1)),
            tgt.reshape(-1),
            reduction="none",
        ).reshape(tgt.shape)
        total_loss   += float((loss * tgt_valid.to(loss.dtype)).sum().item())
        total_tokens += int(tgt_valid.sum().item())

        if device.type == "cuda":
            torch.cuda.empty_cache()

    if total_tokens == 0:
        return float("nan"), 0
    mean_loss = total_loss / total_tokens
    return math.exp(min(mean_loss, 20.0)), total_tokens


# ── In-distribution sample PPL (training stream) ───────────────────────────

@torch.inference_mode()
def compute_train_ppl(brain, tok, cfg, batch_size: int, n_batches: int,
                      mode: str = "mix", chat_ratio: float = 0.6) -> Tuple[float, int]:
    """In-distribution PPL on a fresh sample of the TRAINING stream.

    Defaults to mode='mix' / chat_ratio=0.6 to match how the model was
    trained, so the gap vs OOD reflects true distribution shift rather than
    a text-vs-chat mismatch. seed=12345 (≠ training seed 42) draws an
    unseen sample from the same distribution.
    """
    from neuroslm.data import batch_iterator
    from tqdm.auto import tqdm

    device = next(brain.parameters()).device
    ctx_len = cfg.lang_ctx
    it = batch_iterator(tok, ctx_len, batch_size, seed=12345,
                        mode=mode, chat_ratio=chat_ratio)

    total_loss, total_tok = 0.0, 0
    for _ in tqdm(range(n_batches), desc="train PPL"):
        batch = next(it).to(device)
        ids = batch[:, :-1]
        tgt = batch[:, 1:].contiguous()
        out = brain.forward_lm(ids, tgt)
        logits = out["logits"]
        loss = torch.nn.functional.cross_entropy(
            logits.reshape(-1, logits.size(-1)),
            tgt.reshape(-1),
            reduction="mean",
        )
        total_loss += float(loss.item()) * tgt.numel()
        total_tok  += tgt.numel()
        if device.type == "cuda":
            torch.cuda.empty_cache()

    if total_tok == 0:
        return float("nan"), 0
    return math.exp(min(total_loss / total_tok, 20.0)), total_tok


# ── Main ───────────────────────────────────────────────────────────────────

def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--checkpoint", default=None, help="Explicit .pt path")
    ap.add_argument("--step", type=int, default=None,
                    help="Find checkpoint by exact step number")
    ap.add_argument("--max_ood_windows", type=int, default=200,
                    help="Cap on sliding windows (for speed on T4)")
    ap.add_argument("--max_train_batches", type=int, default=100)
    ap.add_argument("--batch_size", type=int, default=4)
    ap.add_argument("--stride", type=int, default=512)
    ap.add_argument("--train_mode", default="mix", choices=["mix", "text", "chat"],
                    help="Data mode for the in-distribution sample. Default "
                         "'mix' to match training. Use 'text' for a lighter, "
                         "download-free comparison.")
    ap.add_argument("--chat_ratio", type=float, default=0.6,
                    help="chat fraction when --train_mode mix (match training)")
    ap.add_argument("--output", default="ood_test_results.json")
    args = ap.parse_args()

    # Step 1
    if args.checkpoint:
        ckpt_path = args.checkpoint
    else:
        hits = find_checkpoints(args.step)
        if not hits:
            tag = f"step {args.step}" if args.step is not None else "any step"
            print(f"✗ no checkpoint found ({tag}) in {_CKPT_DIRS}", file=sys.stderr)
            return 1
        # Newest by step (last after sort)
        ckpt_path = hits[-1][1]
        print(f"[ood] candidates found: {len(hits)}; picked newest: {ckpt_path}")

    if not Path(ckpt_path).exists():
        print(f"✗ checkpoint does not exist: {ckpt_path}", file=sys.stderr)
        return 1
    if is_lfs_pointer(ckpt_path):
        print(f"✗ {ckpt_path} is an unfetched Git LFS pointer. "
              f"Run `git lfs pull --include={ckpt_path}` first.", file=sys.stderr)
        return 1

    # Step 2
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[ood] device: {device}")
    brain, tok, cfg, step = load_brain(ckpt_path, device)
    n_params = sum(p.numel() for p in brain.parameters())
    print(f"[ood] loaded BRIAN: {n_params/1e6:.1f}M params, step={step}, "
          f"ctx={cfg.lang_ctx}, vocab={cfg.vocab_size}")

    # Step 3
    from datasets import load_dataset
    try:
        ds = load_dataset("wikitext", "wikitext-103-v1", split="test")
        ds_name = "wikitext-103-v1"
    except Exception as e:
        print(f"[ood] wikitext-103 failed ({e}); falling back to wikitext-2")
        ds = load_dataset("wikitext", "wikitext-2-v1", split="test")
        ds_name = "wikitext-2-v1"
    ood_texts = [r["text"] for r in ds]
    print(f"[ood] OOD dataset: {ds_name} ({len(ood_texts)} rows)")

    # Step 4
    ood_ppl, ood_tokens = compute_ppl_sliding(
        brain, tok, ood_texts, cfg.lang_ctx,
        stride=args.stride, batch_size=args.batch_size,
        max_windows=args.max_ood_windows,
    )
    print(f"[ood] OOD eval: {ood_tokens} tokens scored")

    try:
        train_ppl, train_tokens = compute_train_ppl(
            brain, tok, cfg, batch_size=args.batch_size,
            n_batches=args.max_train_batches,
            mode=args.train_mode, chat_ratio=args.chat_ratio,
        )
    except Exception as e:
        print(f"[ood] train-PPL ({args.train_mode}) failed ({e}); "
              f"retrying with mode=text", file=sys.stderr)
        train_ppl, train_tokens = compute_train_ppl(
            brain, tok, cfg, batch_size=args.batch_size,
            n_batches=args.max_train_batches, mode="text",
        )

    # Step 5: report
    gap   = ood_ppl - train_ppl
    ratio = ood_ppl / max(train_ppl, 1e-6)
    if ratio < 1.5:
        verdict = "GOOD GENERALIZATION"
    elif ratio < 2.0:
        verdict = "POSSIBLE OVERFITTING"
    else:
        verdict = "STRONG OVERFITTING"

    print()
    print("════════════════════════════════════════")
    print(" BRIAN OOD Generalization Test")
    print(f" Checkpoint: step_{step}")
    print("════════════════════════════════════════")
    print(f" Train PPL (in-distribution):   {train_ppl:.1f}")
    print(f" {ds_name} PPL (OOD):    {ood_ppl:.1f}")
    print(f" Generalization Gap:             {gap:.1f}  (lower = better)")
    print(f" Gap Ratio (OOD/Train):          {ratio:.2f}  (< 2.0 = good, < 1.5 = excellent)")
    print("════════════════════════════════════════")
    print(f" Verdict: {verdict}")
    print("════════════════════════════════════════")

    # Step 6: save
    results = {
        "checkpoint_path": ckpt_path,
        "step": step,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "n_params": int(n_params),
        "tokenizer": "neuroslm.Tokenizer (gpt2 BPE, vocab=%d)" % cfg.vocab_size,
        "ctx_len": int(cfg.lang_ctx),
        "ood_dataset": ds_name,
        "ood_tokens_evaluated": int(ood_tokens),
        "train_mode": args.train_mode,
        "train_batches": int(args.max_train_batches),
        "train_tokens_evaluated": int(train_tokens),
        "train_ppl": float(train_ppl),
        "ood_ppl": float(ood_ppl),
        "gap": float(gap),
        "gap_ratio": float(ratio),
        "verdict": verdict,
    }
    with open(args.output, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n[ood] results saved to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
