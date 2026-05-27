"""Extract and inspect the exact data batch hitting step ~1500 in our training.

THREE independent runs (RCC P1, P2, P3) all spike from ppl 125 -> 493 at
exactly step 1500 with the SAME seed. That's deterministic, not stochastic
-- the same data lands at the same step every time, and that data is hard.

This script reproduces the data iterator with the SAME settings as the
training runs (seed=0, mode='mix', chat_ratio=0.6, batch=4, grad_accum=4)
and dumps the sequences that comprise step 1500 so we can see what
breaks the model.

Usage:
    .venv/Scripts/python.exe scripts/inspect_step1500_batch.py
"""
from __future__ import annotations
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import itertools
from neuroslm.tokenizer import Tokenizer
from neuroslm.data import batch_iterator


# Match the training-run config exactly
SEED = 0
MODE = "mix"
CHAT_RATIO = 0.6        # from rcc_bowtie_30m_p2 launches
CTX_LEN = 1024
BATCH = 4
GRAD_ACCUM = 4

TARGET_STEP = 1500       # the spike step
WINDOW = 2               # also dump step 1499 + 1501 for context


def _heuristic_score(text: str) -> dict:
    """Per-sequence pathology heuristics that often correlate with hard
    batches: low text-entropy, long no-space runs (URLs/base64/code),
    high non-ASCII fraction, high repetition."""
    n = max(1, len(text))
    # Longest run without whitespace
    longest_token = max((len(w) for w in text.split()), default=0)
    # Non-ASCII fraction
    nonascii = sum(1 for c in text if ord(c) > 127) / n
    # Repetition: 4-gram self-overlap fraction
    chars = text[:8000]   # cap for speed
    ngrams = {}
    for i in range(len(chars) - 3):
        g = chars[i:i+4]
        ngrams[g] = ngrams.get(g, 0) + 1
    repeated = sum(c for c in ngrams.values() if c > 1) / max(1, len(ngrams))
    # Whitespace-density (low → blob)
    ws_density = sum(1 for c in text if c.isspace()) / n
    return {
        "len_chars": len(text),
        "longest_token": longest_token,
        "nonascii_frac": round(nonascii, 3),
        "repeated_4gram_frac": round(repeated, 3),
        "whitespace_frac": round(ws_density, 3),
    }


def main():
    print(f"Reproducing data stream: seed={SEED} mode={MODE} chat_ratio={CHAT_RATIO}")
    print(f"  ctx_len={CTX_LEN} batch={BATCH} grad_accum={GRAD_ACCUM}")
    print(f"  Target: train-loop step {TARGET_STEP} "
          f"(= micro-batches {TARGET_STEP*GRAD_ACCUM} .. "
          f"{(TARGET_STEP+1)*GRAD_ACCUM - 1})")
    print()

    tok = Tokenizer()
    it = batch_iterator(tok, CTX_LEN, BATCH, seed=SEED,
                        mode=MODE, chat_ratio=CHAT_RATIO)

    # Fast-forward to just before target step.
    # train.py treats each "step" as `grad_accum` micro-batches.
    target_micro_start = TARGET_STEP * GRAD_ACCUM
    target_micro_end   = (TARGET_STEP + WINDOW) * GRAD_ACCUM

    # Skip forward (sip from the iterator without doing anything)
    skip_n = (TARGET_STEP - WINDOW) * GRAD_ACCUM
    print(f"Skipping {skip_n} micro-batches...")
    for _ in range(skip_n):
        next(it)
    print("Skipped.")
    print()

    # Now decode + analyze the next 3*GRAD_ACCUM = 12 micro-batches
    # (steps WINDOW=1499, target=1500, 1501)
    for micro_i in range(skip_n, target_micro_end):
        step = micro_i // GRAD_ACCUM
        micro = micro_i % GRAD_ACCUM
        batch = next(it)   # (B, ctx_len+1)
        for b in range(batch.size(0)):
            ids = batch[b].tolist()
            text = tok.decode(ids)
            stats = _heuristic_score(text)
            preview = text.replace("\n", "\\n")[:240]
            marker = "  <<< TARGET STEP" if step == TARGET_STEP else ""
            print(f"step {step:4d}  micro {micro}  seq {b}  {marker}")
            print(f"  stats: {stats}")
            print(f"  text[:240]: {preview!r}")
            print()


if __name__ == "__main__":
    main()
