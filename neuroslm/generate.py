"""Sample from a trained NeuroSLM checkpoint with full per-tick brain logs.

Usage:
  python -m neuroslm.generate --ckpt checkpoints/neuroslm_small_500.pt \
         --prompt "Once upon a time" --max_new 80 --log_ticks --wander_first 4
"""
from __future__ import annotations
import argparse
import json
import sys
import torch

from .config import BrainConfig
from .tokenizer import Tokenizer
from .brain import Brain
from .dna import GenePool


def make_logger(verbose: bool):
    def _log(step, info):
        if not verbose:
            return
        nt = info["nt"]
        nt_str = " ".join(f"{k}={v:.2f}" for k, v in nt.items())
        rou = info["routing"][0].tolist()
        rou_str = " ".join(f"{x:.2f}" for x in rou)
        line = (
            f"tick {step:3d} | act={info.get('action_name','?'):<13s} "
            f"| emit={'Y' if info.get('emitted') else '.'} "
            f"| threat={float(info['threat'][0]):.2f}"
            f"{' SURVIVAL!' if bool(info['survival'][0]) else ''} "
            f"| novelty={float(info['novelty'][0]):.2f} "
            f"| value={float(info['value'][0]):+.2f} "
            f"| route=[{rou_str}] "
            f"| NT[{nt_str}]"
        )
        print(line, file=sys.stderr)
    return _log


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--prompt", default="Once upon a time")
    ap.add_argument("--max_new", type=int, default=80)
    ap.add_argument("--temperature", type=float, default=0.9)
    ap.add_argument("--top_k", type=int, default=40)
    ap.add_argument("--device", default=None)
    ap.add_argument("--log_ticks", action="store_true")
    ap.add_argument("--wander_first", type=int, default=0)
    ap.add_argument("--show_genome", action="store_true")
    ap.add_argument("--show_trophic", action="store_true")
    args = ap.parse_args()

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = torch.load(args.ckpt, map_location=device)
    cfg = BrainConfig(**ckpt["cfg"])
    tok = Tokenizer()
    brain = Brain(cfg).to(device).eval()
    brain.load_partial(ckpt["model"], verbose=False)
    if "gene_pool" in ckpt:
        brain.gene_pool = GenePool.from_state(ckpt["gene_pool"])

    if args.show_genome:
        print("=== Active genome ===", file=sys.stderr)
        print(json.dumps(brain.gene_pool.active().to_dict(), indent=2), file=sys.stderr)
    if args.show_trophic:
        print("=== Trophic stats ===", file=sys.stderr)
        print(json.dumps(brain.trophic.stats(), indent=2), file=sys.stderr)

    ids = torch.tensor([tok.encode(args.prompt)], dtype=torch.long, device=device)

    if args.wander_first > 0:
        print(f"=== Mind wandering for {args.wander_first} ticks ===",
              file=sys.stderr)
        state = brain.init_latents(ids.size(0), device)
        brain.wander(ids[:, -cfg.lang_ctx:], state, max_steps=args.wander_first,
                     on_step=make_logger(args.log_ticks))

    print("=== Generation ===", file=sys.stderr)
    out = brain.generate(ids, max_new=args.max_new,
                         temperature=args.temperature, top_k=args.top_k,
                         on_tick=make_logger(args.log_ticks))
    text = tok.decode(out[0].tolist())
    try:
        print(text)
    except UnicodeEncodeError:
        sys.stdout.buffer.write(text.encode("utf-8", errors="replace"))
        sys.stdout.buffer.write(b"\n")


if __name__ == "__main__":
    main()
