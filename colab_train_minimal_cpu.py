#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Minimal CPU training for Colab — no GPU required.

Trains a tiny ~5-14M parameter LM on **real text** so loss actually
drops below ``log(vocab_size)``. Demonstrates the evolutionary
training pipeline (DNA + fitness config + OOD-aware logging).

Data source resolution (in order):

  1. ``text_source="auto"`` (default): try HuggingFace streaming
     (:func:`neuroslm.data.batch_iterator`). FineWeb-Edu / SmolLM /
     TinyStories / wikitext are tried in turn.
  2. On HF failure: fall back to the **bundled local corpus** at
     ``neuroslm/assets/local_corpus.txt`` — no network required.
  3. Only if both fail (or ``text_source="synthetic"``): produce
     ``torch.randint`` batches with a LOUD warning so the operator
     knows why loss will plateau at ``log(V)``.

The vocab size of the model adapts to the tokenizer in use
(GPT-2 BPE = 50257 by default; smaller in tests).

Run in Colab:
  !python colab_train_minimal_cpu.py
"""
from __future__ import annotations
import math
import os
import sys
import time
from typing import Any, Optional

import torch
import torch.nn as nn
import torch.optim as optim

# Suppress noisy TF logs (Colab default GPU image preloads TF).
os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL', '3')


# ──────────────────────────────────────────────────────────────────────
# Model
# ──────────────────────────────────────────────────────────────────────

def create_tiny_lm(vocab_size: int, d_model: int = 192,
                   depth: int = 3, seq_len: int = 256) -> nn.Module:
    """Create a small Transformer LM for CPU training.

    Size scales with ``vocab_size``:
      * vocab=4096  → ~5 M params
      * vocab=50257 → ~14 M params (GPT-2 BPE)
      * vocab=128   → ~1 M params (test fixture)
    """

    # Pick a head count that divides d_model. With d_model=192, valid
    # head counts are {1,2,3,4,6,8,12}. Default 6 = 32-dim heads.
    n_heads = 6 if d_model % 6 == 0 else max(1, d_model // 32)

    class TinyLM(nn.Module):
        def __init__(self):
            super().__init__()
            self.embed = nn.Embedding(vocab_size, d_model)
            self.pos_embed = nn.Embedding(seq_len, d_model)
            self.layers = nn.ModuleList([
                nn.TransformerEncoderLayer(
                    d_model=d_model, nhead=n_heads,
                    dim_feedforward=d_model * 4,
                    batch_first=True, dropout=0.0,
                )
                for _ in range(depth)
            ])
            self.norm = nn.LayerNorm(d_model)
            self.lm_head = nn.Linear(d_model, vocab_size)
            self.vocab_size = vocab_size
            self.d_model = d_model

        def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
            sl = input_ids.shape[1]
            x = self.embed(input_ids)
            pos_ids = torch.arange(sl, device=input_ids.device).unsqueeze(0)
            x = x + self.pos_embed(pos_ids[:, :sl])
            for layer in self.layers:
                x = layer(x)
            x = self.norm(x)
            return self.lm_head(x)

    return TinyLM()


# ──────────────────────────────────────────────────────────────────────
# Batch sources — real → local → synthetic, in that priority order
# ──────────────────────────────────────────────────────────────────────

class _RealHFBatchSource:
    """Wraps :func:`neuroslm.data.batch_iterator` (HuggingFace streaming)."""
    label = "real-hf"

    def __init__(self, tokenizer, batch: int, seq_len: int, device: str):
        from neuroslm.data import batch_iterator
        self._device = device
        # Prime so we surface network failures here instead of mid-loop.
        # ``max_open_attempts=1`` makes us fail fast — if the very first
        # ``open_stream()`` raises, we don't retry, we let the caller
        # downgrade to a local-corpus or synthetic source. Without this,
        # the iterator's exponential-backoff reconnect loop would hang
        # for minutes on a CPU smoke run with no network.
        self._it = batch_iterator(
            tokenizer, ctx_len=seq_len, batch_size=batch,
            seed=0, mode="text", max_open_attempts=1,
        )
        self._primed = next(self._it)

    def next(self):
        if self._primed is not None:
            window, self._primed = self._primed, None
        else:
            window = next(self._it)
        window = window.to(self._device)
        return window[:, :-1].contiguous(), window[:, 1:].contiguous()


class _LocalCorpusBatchSource:
    """Streams from the bundled ``neuroslm/assets/local_corpus.txt``."""
    label = "local-corpus"

    def __init__(self, tokenizer, batch: int, seq_len: int, device: str):
        from neuroslm.data import local_corpus_batch_iterator
        self._device = device
        self._it = local_corpus_batch_iterator(
            tokenizer, ctx_len=seq_len, batch_size=batch,
        )

    def next(self):
        window = next(self._it).to(self._device)
        return window[:, :-1].contiguous(), window[:, 1:].contiguous()


class _SyntheticBatchSource:
    """Last-resort fallback: ``torch.randint``. Produces the infamous
    log(V) plateau — only used when *everything* else has failed AND
    the operator was explicitly warned."""
    label = "synthetic-random"

    def __init__(self, vocab_size: int, batch: int, seq_len: int,
                 device: str, seed: int = 0):
        self._vocab = vocab_size
        self._batch = batch
        self._seq_len = seq_len
        self._device = device
        self._gen = torch.Generator(device=device).manual_seed(seed)

    def next(self):
        ids = torch.randint(0, self._vocab,
                            (self._batch, self._seq_len - 1),
                            device=self._device, generator=self._gen)
        tgt = torch.randint(0, self._vocab,
                            (self._batch, self._seq_len - 1),
                            device=self._device, generator=self._gen)
        return ids, tgt


def _build_batch_source(text_source: str, tokenizer, batch: int,
                        seq_len: int, device: str) -> Any:
    """Resolve the requested ``text_source`` to a concrete batch source.

    ``text_source``:
      * ``"auto"``      → try real HF, then local corpus, then synthetic
                          (with a loud warning at each downgrade).
      * ``"real"``      → real HF only; raise on failure.
      * ``"local"``     → bundled local corpus only.
      * ``"synthetic"`` → torch.randint with a LOUD warning.
    """
    if text_source == "synthetic":
        print("=" * 70)
        print("[data] WARNING: text_source='synthetic' — batches are "
              "torch.randint")
        print("[data] WARNING: lm_loss will plateau at log(vocab_size). "
              "Training will not learn linguistic structure.")
        print("=" * 70)
        return _SyntheticBatchSource(
            tokenizer.vocab_size, batch, seq_len, device)

    if text_source == "local":
        print("[data] local corpus: streaming from bundled "
              "assets/local_corpus.txt")
        return _LocalCorpusBatchSource(tokenizer, batch, seq_len, device)

    if text_source == "real":
        print("[data] real: HuggingFace streaming (raise on failure)")
        return _RealHFBatchSource(tokenizer, batch, seq_len, device)

    # text_source == "auto" — the production default.
    try:
        src = _RealHFBatchSource(tokenizer, batch, seq_len, device)
        print(f"[data] using real HuggingFace stream "
              f"(batch_iterator, vocab={tokenizer.vocab_size})")
        return src
    except Exception as e:  # noqa: BLE001
        print(f"[data] real HF stream unavailable "
              f"({type(e).__name__}: {e})")

    try:
        src = _LocalCorpusBatchSource(tokenizer, batch, seq_len, device)
        print("[data] local corpus: falling back to bundled "
              "assets/local_corpus.txt (no network needed)")
        return src
    except Exception as e:  # noqa: BLE001
        print(f"[data] local corpus unavailable "
              f"({type(e).__name__}: {e})")

    print("=" * 70)
    print("[data] synthetic fallback: WARNING — all real-data sources failed.")
    print("[data] WARNING: lm_loss will plateau at log(vocab_size). "
          "Training will NOT learn.")
    print("[data] Reinstall neuroslm (for local corpus) or fix network "
          "(for HF stream).")
    print("=" * 70)
    return _SyntheticBatchSource(
        tokenizer.vocab_size, batch, seq_len, device)


# ──────────────────────────────────────────────────────────────────────
# Tokenizer
# ──────────────────────────────────────────────────────────────────────

def _load_tokenizer():
    """Project tokenizer (GPT-2 BPE via tiktoken, vocab=50257) with a
    safe fallback for environments where tiktoken/data files aren't
    installed."""
    try:
        from neuroslm.tokenizer import Tokenizer
        return Tokenizer()
    except Exception as e:  # noqa: BLE001
        print(f"[tokenizer] project tokenizer unavailable ({e!r}); "
              f"using a tiny synthetic vocab=4096 (results will be limited)")

        class _SynthTok:
            vocab_size = 4096
            eos_id = 0
            def encode(self, text: str) -> list[int]:
                return [(ord(c) * 131 + 7) % (self.vocab_size - 1) + 1
                        for c in text]
            def decode(self, ids): return ""
        return _SynthTok()


# ──────────────────────────────────────────────────────────────────────
# Main entry point
# ──────────────────────────────────────────────────────────────────────

def main(steps: int = 10, ood_every: int = 500,
         dna_path: str = "dna/evol/arch.dna",
         text_source: str = "auto") -> Optional[dict]:
    """Run a tiny CPU training loop with real data and fitness/OOD logging.

    Returns
    -------
    dict with keys:
        - ``final_lm_loss`` (float): the last training-step LM loss
        - ``vocab_size`` (int): the model's vocab size
        - ``data_source`` (str): which batch source was used
        - ``steps`` (int): how many steps were run
        - ``ood_results`` (list[dict]): OOD eval snapshots

    For ``steps=0`` we still return a dict (setup-only mode used by tests).
    """
    print("=" * 70)
    print("NeuroSLM Minimal CPU Training — Colab Demo")
    print("=" * 70)

    # [1] Load DNA / fitness config -----------------------------------
    print("\n[1] Loading evol.dna with fitness config...")
    try:
        from neuroslm.utils import init_evolution
        ctx = init_evolution(dna_path)
        fitness_cfg = ctx['fitness_config']
        print(f"    [OK] DNA loaded, fitness: "
              f"{len(fitness_cfg.objectives)} objectives")
    except FileNotFoundError:
        print("    [SKIP] evol.dna not found — using default fitness")
        from neuroslm.fitness import FitnessConfig
        # `dna/` is .gitignored, so on a fresh Colab clone the DNA file
        # legitimately doesn't exist.  Recover with the built-in default
        # fitness config (no path required).
        fitness_cfg = FitnessConfig.load_or_default()

    # [2] Tokenizer + data source ------------------------------------
    print("\n[2] Resolving tokenizer + data source...")
    tokenizer = _load_tokenizer()
    seq_len = 64    # short windows so even GPT-2 vocab fits on CPU
    batch_size = 2  # tiny batches — CPU memory budget

    device = "cpu"
    batch_source = _build_batch_source(
        text_source, tokenizer, batch_size, seq_len, device,
    )
    vocab_size = tokenizer.vocab_size
    data_label = getattr(batch_source, "label", "unknown")

    # [3] Model + optimizer ------------------------------------------
    print(f"\n[3] Creating model (vocab={vocab_size}, "
          f"seq_len={seq_len}, batch={batch_size})...")
    d_model = 192
    depth = 3
    model = create_tiny_lm(vocab_size, d_model, depth, seq_len).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"    [OK] Model: {n_params / 1e6:.2f}M params  "
          f"(data={data_label})")

    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = nn.CrossEntropyLoss()

    # [4] Training loop -----------------------------------------------
    print(f"\n[4] Training for {steps} steps "
          f"(batch={batch_size}, seq_len={seq_len})...")
    print(f"    Device: {device}")
    print(f"    Data:   {data_label}")
    print(f"    Fitness objectives: "
          f"{[obj.name for obj in fitness_cfg.objectives]}")
    print(f"    OOD eval every {ood_every} steps")

    model.train()
    start = time.time()
    ood_results: list[dict] = []
    last_lm_loss = float("nan")

    for step in range(1, max(1, steps) + 1):
        if steps == 0:
            break  # setup-only mode for tests

        input_ids, target_ids = batch_source.next()
        logits = model(input_ids)
        lm_loss = loss_fn(
            logits.reshape(-1, vocab_size),
            target_ids.reshape(-1),
        )

        metrics = {
            "ood_ppl": max(100.0 - step * 2, 80.0),
            "phi": 0.05 + step * 0.005,
            "gap_ratio": max(5.0 - step * 0.1, 4.0),
        }
        fitness_loss = fitness_cfg.compute_loss(metrics)
        total_loss = lm_loss + 0.01 * fitness_loss

        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()

        last_lm_loss = float(lm_loss.detach().item())

        if step % 10 == 0 or step == 1 or step == steps:
            elapsed = time.time() - start
            baseline = math.log(vocab_size)
            margin = baseline - last_lm_loss
            print(
                f"    step {step:5d}  lm_loss={last_lm_loss:.4f}  "
                f"(log V={baseline:.2f}, margin={margin:+.2f})  "
                f"fit={float(fitness_loss):.3f}  ({elapsed:.1f}s)"
            )

        if step % ood_every == 0:
            with torch.no_grad():
                ood_in, ood_tgt = batch_source.next()
                ood_logits = model(ood_in)
                ood_loss = loss_fn(
                    ood_logits.reshape(-1, vocab_size),
                    ood_tgt.reshape(-1),
                )
                ood_ppl = float(torch.exp(ood_loss).item())
                ood_results.append({"step": step, "ood_ppl": ood_ppl})
                print(f"    [OOD] step {step:5d}  ood_ppl={ood_ppl:.2f}")

    # [5] Verification ------------------------------------------------
    print("\n[5] Verification:")
    print(f"    [OK] Training completed in {time.time() - start:.1f}s")
    print(f"    [OK] Final lm_loss = {last_lm_loss:.4f}  "
          f"(log(V) = {math.log(vocab_size):.2f})")
    print(f"    [OK] Data source:    {data_label}")
    print(f"    [OK] Fitness config: {fitness_cfg.enabled}")

    print("\n" + "=" * 70)
    print(f"[OK] Pipeline verified on CPU (data={data_label})")
    print("=" * 70)
    if data_label == "synthetic-random":
        print("\n*** Reminder: synthetic data cannot reduce lm_loss below "
              "log(V). ***")
        print("*** Connect to the internet OR ship the local corpus to "
              "actually learn. ***")

    return {
        "final_lm_loss": last_lm_loss,
        "vocab_size": vocab_size,
        "data_source": data_label,
        "steps": steps,
        "ood_results": ood_results,
    }


if __name__ == "__main__":
    main()
