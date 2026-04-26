"""Helper script to run training on Colab or other GPU hosts.

This script detects CUDA availability, selects an appropriate preset and device,
and forwards arguments to the project's training entrypoint.
"""
from __future__ import annotations
import argparse
import subprocess
import sys
import shlex

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--preset", default="medium")
    ap.add_argument("--steps", type=int, default=2000)
    ap.add_argument("--batch_size", type=int, default=8)
    ap.add_argument("--device", default=None, help="override device (cuda or cpu)")
    args = ap.parse_args()

    device = args.device or ("cuda" if _cuda_available() else "cpu")
    print(f"[train_colab] device={device}")

    cmd = f"python -m neuroslm.train --preset {args.preset} --steps {args.steps} --batch_size {args.batch_size} --device {device}"
    print(f"Running: {cmd}")
    subprocess.check_call(shlex.split(cmd))

def _cuda_available() -> bool:
    try:
        import torch
        return torch.cuda.is_available()
    except Exception:
        return False

if __name__ == '__main__':
    main()
