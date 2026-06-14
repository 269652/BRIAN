#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Tiny CPU training for Colab — minimal settings to fit in 12GB RAM.

Run in Colab cell:
  %run colab_train_tiny_cpu.py

No GPU required — runs on CPU-only Colab (12GB RAM).
Trains for 100 steps to verify the pipeline.
"""
import os
import subprocess
import sys
import time

def run_training():
    """Run minimal training on CPU."""
    os.chdir("/content/brian")

    # Set environment
    os.environ["PYTHONUNBUFFERED"] = "1"
    os.environ["SCALE"] = "tiny"  # Custom tiny scale

    # Minimal training config for CPU
    # d_model=128 (vs 512), depth=2 (vs 8), batch=1, seq_len=256, steps=100
    cmd = (
        "python -u -m neuroslm.train_dsl "
        "--arch architectures/current "
        "--model dsl_lm "
        "--preset rcc_bowtie_30m_p4 "  # Use existing preset but override with flags
        "--data real "
        "--mode mix "
        "--chat_ratio 0.6 "
        "--steps 100 "          # Tiny: 100 steps
        "--batch 1 "            # Tiny: batch size 1
        "--seq_len 256 "        # Tiny: 256 tokens (vs 2048)
        "--d_sem 128 "          # Tiny: 128 dim (vs 512)
        "--device cpu "
        "--log_every 10 "
        "--save_every 50 "
        "--ood_every 0 "        # Skip OOD eval
        "--ckpt_dir /tmp/ckpts"
    )

    print("=" * 70)
    print("NeuroSLM Tiny CPU Training — Colab")
    print("=" * 70)
    print(f"\nConfig:")
    print(f"  Device: CPU")
    print(f"  Steps: 100")
    print(f"  Batch: 1")
    print(f"  Seq Len: 256")
    print(f"  D_SEM: 128 (vs 512)")
    print(f"  Preset: rcc_bowtie_30m_p4 (dims overridden)")
    print(f"\nExpected memory: ~2GB")
    print(f"Expected runtime: ~5-10 minutes on CPU")
    print("\n" + "=" * 70)
    print(f"Command:\n  {cmd}\n")

    # Run with streaming output
    proc = subprocess.Popen(
        cmd,
        shell=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )

    for line in proc.stdout:
        print(line, end="", flush=True)

    rc = proc.wait()

    print("\n" + "=" * 70)
    if rc == 0:
        print("TRAINING COMPLETE - Steps finished successfully")
    else:
        print(f"Training exited with code {rc}")
    print("=" * 70)

    return rc


if __name__ == "__main__":
    exit_code = run_training()
    sys.exit(exit_code)
