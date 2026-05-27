"""Faster inspection of the step-1500 hard batch.

Approach: re-run the exact data pipeline with seed=0 and PRINT each
batch's pathology heuristics as we go, not just at the target. That way
we see immediate output (sanity-check the script is alive) and can
identify whether step 1500 contains an obvious pattern or whether the
pathology is something else.

Run:
    .venv/Scripts/python.exe scripts/inspect_step1500_v2.py
"""
from __future__ import annotations
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import time
import itertools

from neuroslm.tokenizer import Tokenizer
from neuroslm.data import batch_iterator


SEED = 0
MODE = "mix"
CHAT_RATIO = 0.6
CTX_LEN = 1024
BATCH = 4
GRAD_ACCUM = 4
TARGET = 1500


def _score(text: str) -> dict:
    n = max(1, len(text))
    longest = max((len(w) for w in text.split()), default=0)
    nonascii = sum(1 for c in text if ord(c) > 127) / n
    ws = sum(1 for c in text if c.isspace()) / n
    # 4-gram repetition
    chars = text[:8000]
    ng = {}
    for i in range(len(chars) - 3):
        g = chars[i:i+4]
        ng[g] = ng.get(g, 0) + 1
    rep = sum(v for v in ng.values() if v > 1) / max(1, len(ng))
    return {
        "chars": len(text),
        "longest_token": longest,
        "nonascii_pct": int(nonascii * 100),
        "ws_pct": int(ws * 100),
        "rep_4gram": round(rep, 2),
    }


def main():
    print(f"[init] seed={SEED} mode={MODE} chat_ratio={CHAT_RATIO} "
          f"ctx={CTX_LEN} batch={BATCH} grad_accum={GRAD_ACCUM}", flush=True)
    t0 = time.time()
    tok = Tokenizer()
    print(f"[init] tokenizer ready ({time.time()-t0:.1f}s)", flush=True)

    print("[init] opening data iterator (this initializes HF streams)...", flush=True)
    t1 = time.time()
    it = batch_iterator(tok, CTX_LEN, BATCH, seed=SEED,
                        mode=MODE, chat_ratio=CHAT_RATIO)

    # PEEK at the first batch right away so we know the stream is alive.
    first = next(it)
    print(f"[init] first batch shape {tuple(first.shape)} "
          f"({time.time()-t1:.1f}s since iterator open)", flush=True)

    # Now fast-forward. Print every 200 micro-batches so we see progress.
    n_skip = (TARGET - 1) * GRAD_ACCUM   # skip up to just before TARGET-1
    print(f"[ff] skipping {n_skip} micro-batches "
          f"(already consumed 1, need {n_skip-1} more)...", flush=True)
    t2 = time.time()
    last_print = t2
    for i in range(1, n_skip):
        next(it)
        if (i + 1) % 200 == 0 and time.time() - last_print > 5.0:
            print(f"[ff] micro-batch {i+1}/{n_skip} "
                  f"({time.time()-t2:.0f}s elapsed)", flush=True)
            last_print = time.time()
    print(f"[ff] done ({time.time()-t2:.0f}s)", flush=True)

    # Now dump steps TARGET-1, TARGET, TARGET+1.
    print()
    print(f"=== inspecting steps {TARGET-1}, {TARGET}, {TARGET+1} ===")
    for micro_offset in range(3 * GRAD_ACCUM):
        step = (TARGET - 1) + micro_offset // GRAD_ACCUM
        micro = micro_offset % GRAD_ACCUM
        batch = next(it)
        marker = " *** TARGET ***" if step == TARGET else ""
        for b in range(batch.size(0)):
            ids = batch[b].tolist()
            text = tok.decode(ids)
            stats = _score(text)
            preview = text.replace("\n", "\\n")[:160]
            print(f"step {step:4d} micro {micro} seq {b}{marker}")
            print(f"  stats: {stats}")
            print(f"  text[:160]: {preview!r}")
            print()


if __name__ == "__main__":
    main()
