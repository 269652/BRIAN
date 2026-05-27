# -*- coding: utf-8 -*-
"""DSL-driven training entrypoint.

Parallel to `neuroslm.train` (which trains the hand-written `Brain`):
this module loads an architecture from `architectures/<name>/`,
compiles it via the DSL pipeline, wraps it in a `BRIANHarness`, and
runs a language-model training loop.

The harness reads `training { ... }` from the architecture's
`arch.neuro` for loss clipping, optimizer choice, label smoothing,
grad accumulation, and grad clipping. Per-step model behavior is
otherwise determined entirely by the .neuro files — no Python
architecture code path involved.

Usage:
    python -m neuroslm.train_dsl --arch architectures/rcc_bowtie \\
        --steps 10000 --batch 4 --seq_len 256 --d_sem 256
"""
from __future__ import annotations
import argparse
import os
import time
from pathlib import Path
from typing import Optional

import torch

from neuroslm.dsl.codegen import CodeGenerator
from neuroslm.dsl.multifile import compile_folder
from neuroslm.dsl.training_config import load_training_config_from_arch
from neuroslm.dsl.param_scopes import load_param_scopes_from_arch
from neuroslm.harness import BRIANHarness


# ── Tokenizer ────────────────────────────────────────────────────────

def _load_tokenizer():
    """Reuse the project's tokenizer if available; fall back to a
    synthetic vocab for smoke tests when the tokenizer module is missing
    or its data files aren't present.
    """
    try:
        from neuroslm.tokenizer import Tokenizer
        return Tokenizer()
    except Exception as e:
        print(f"[train_dsl] tokenizer unavailable ({e!r}); using synthetic vocab")
        class _SynthTok:
            vocab_size = 1024
            def encode(self, _: str): return [0]
            def decode(self, _): return ""
        return _SynthTok()


# ── Data — synthetic random batches for Phase A smoke ───────────────

class SyntheticBatchSource:
    """Generates random (ids, targets) batches.

    Phase A ships synthetic data so we can prove the harness trains end-
    to-end without dragging in the real data pipeline. Phase E plugs in
    `neuroslm.data` properly.
    """
    def __init__(self, vocab_size: int, batch: int, seq_len: int,
                 device: str = "cpu", seed: int = 0):
        self.vocab_size = vocab_size
        self.batch = batch
        self.seq_len = seq_len
        self.device = device
        self.gen = torch.Generator(device=device).manual_seed(seed)

    def next(self):
        ids = torch.randint(0, self.vocab_size, (self.batch, self.seq_len),
                            device=self.device, generator=self.gen)
        targets = torch.randint(0, self.vocab_size,
                                (self.batch, self.seq_len),
                                device=self.device, generator=self.gen)
        return ids, targets


# ── Build the harness ────────────────────────────────────────────────

def build_harness(arch_root: Path, vocab_size: int, d_sem: int,
                  device: str = "cpu",
                  sink_population: str = "motor") -> BRIANHarness:
    print(f"[train_dsl] compiling architecture from {arch_root}")
    ir = compile_folder(arch_root)
    print(f"[train_dsl]   populations:  {len(ir.populations)}")
    print(f"[train_dsl]   synapses:     {len(ir.synapses)}")
    print(f"[train_dsl]   modulations:  {len(ir.modulations)}")

    cfg = load_training_config_from_arch(arch_root)
    print(f"[train_dsl] training config: "
          f"loss_clip={cfg.loss_clipping.enabled}(f={cfg.loss_clipping.factor}), "
          f"opt={cfg.optimizer}, lr={cfg.learning_rate}, "
          f"grad_accum={cfg.grad_accum}, label_smooth={cfg.label_smoothing}")

    Cls = CodeGenerator(ir, module_name="DSLCircuit").compile_to_module()
    circuit = Cls(d_sem=d_sem).to(device)

    harness = BRIANHarness(
        circuit=circuit, vocab_size=vocab_size, d_sem=d_sem,
        training_config=cfg, sink_population=sink_population,
    ).to(device)

    # Apply declarative gradient isolation (p3 fix) from param_scope blocks.
    scopes = load_param_scopes_from_arch(arch_root)
    if scopes:
        harness.apply_param_scopes(scopes)
        detached = [s.name for s in scopes
                    if s.gradient == "detached_from_main_loss"]
        print(f"[train_dsl] param_scopes: {len(scopes)} declared, "
              f"detached-from-main-loss: {detached}")

    return harness


def _maybe_resume(harness: BRIANHarness, ckpt_dir: Path) -> int:
    """Load the most recent dsl_arch_step*.pt from ckpt_dir if present.

    Returns the resumed step (0 if no checkpoint found).
    """
    if not ckpt_dir.is_dir():
        return 0
    ckpts = sorted(
        ckpt_dir.glob("dsl_arch_step*.pt"),
        key=lambda p: int(p.stem.replace("dsl_arch_step", "")),
    )
    if not ckpts:
        return 0
    latest = ckpts[-1]
    step = harness.load_checkpoint(str(latest))
    print(f"[train_dsl] resumed from {latest} @ step {step}")
    return step


# ── Train loop ───────────────────────────────────────────────────────

def train(harness: BRIANHarness, source: SyntheticBatchSource,
          steps: int, log_every: int = 20, save_every: int = 1000,
          ckpt_dir: Optional[Path] = None, start_step: int = 0) -> None:
    """Run train_steps from `start_step+1` to `steps`. Logs + checkpoints."""
    if ckpt_dir is not None:
        ckpt_dir.mkdir(parents=True, exist_ok=True)

    t0 = time.time()
    last_log = t0
    log_buf = []

    for step in range(start_step + 1, steps + 1):
        ids, targets = source.next()
        loss = harness.train_step(ids, targets)
        log_buf.append(loss)

        if step % log_every == 0:
            now = time.time()
            avg = sum(log_buf) / len(log_buf)
            log_buf.clear()
            steps_per_sec = log_every / max(now - last_log, 1e-6)
            print(f"[train_dsl] step={step:>6d} loss={avg:.4f} "
                  f"steps/s={steps_per_sec:.2f} "
                  f"elapsed={int(now - t0)}s", flush=True)
            last_log = now

        if ckpt_dir is not None and step % save_every == 0:
            path = ckpt_dir / f"dsl_arch_step{step}.pt"
            harness.save_checkpoint(str(path), step=step)
            print(f"[train_dsl] saved checkpoint {path}", flush=True)


# ── CLI ──────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arch", required=True,
                        help="path to architecture folder (containing arch.neuro)")
    parser.add_argument("--steps", type=int, default=1000)
    parser.add_argument("--batch", type=int, default=4)
    parser.add_argument("--seq_len", type=int, default=128)
    parser.add_argument("--d_sem", type=int, default=256)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--log_every", type=int, default=20)
    parser.add_argument("--save_every", type=int, default=1000)
    parser.add_argument("--ckpt_dir", default="lfs_checkpoints")
    parser.add_argument("--sink", default="motor",
                        help="population whose output feeds the LM head")
    parser.add_argument("--vocab_size", type=int, default=0,
                        help="0 → take from tokenizer")
    parser.add_argument("--amp", default="bf16", choices=["bf16", "fp16"],
                        help="mixed-precision dtype (cuda only)")
    parser.add_argument("--resume", action="store_true",
                        help="resume from the latest dsl_arch_step*.pt in ckpt_dir")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    arch_root = Path(args.arch).resolve()
    if not (arch_root / "arch.neuro").is_file():
        parser.error(f"missing {arch_root}/arch.neuro")

    vocab_size = args.vocab_size or _load_tokenizer().vocab_size

    harness = build_harness(
        arch_root=arch_root, vocab_size=vocab_size, d_sem=args.d_sem,
        device=args.device, sink_population=args.sink,
    )

    # LR schedule over the full step budget (10% warmup) and mixed precision.
    warmup = max(1, args.steps // 10)
    harness.set_schedule(warmup=warmup, total=args.steps, min_lr_ratio=0.1)
    if args.device == "cuda":
        harness.enable_mixed_precision(dtype=args.amp)
        print(f"[train_dsl] mixed precision: {args.amp}")

    print(harness.topology_summary())
    print(f"[train_dsl] device={args.device}, warmup={warmup}/{args.steps}")

    ckpt_dir = Path(args.ckpt_dir) if args.ckpt_dir else None
    start_step = 0
    if args.resume and ckpt_dir is not None:
        start_step = _maybe_resume(harness, ckpt_dir)
        harness._global_step = start_step

    source = SyntheticBatchSource(
        vocab_size=vocab_size, batch=args.batch, seq_len=args.seq_len,
        device=args.device, seed=args.seed,
    )

    train(
        harness=harness, source=source,
        steps=args.steps, log_every=args.log_every,
        save_every=args.save_every, ckpt_dir=ckpt_dir,
        start_step=start_step,
    )

    print("[train_dsl] done.")


if __name__ == "__main__":
    main()
