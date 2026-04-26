"""Streaming data loader.

Two modes:
  - mode="text" : narrative / textbook text (TinyStories, Cosmopedia, wikitext).
  - mode="chat" : multi-turn dialogue, formatted as
        User: ...
        Assistant: ...
    so the model learns the alternation pattern.
  - mode="mix"  : interleaves text and chat samples (3:1 chat-favoured).

We stream (no full download) and produce fixed-length token windows.
The iterator is infinite & self-healing: any HF stream error is caught
and the stream is reopened with a backoff. Token buffer is preserved
across reconnects so no data is lost mid-batch.
"""
from __future__ import annotations
import random
import itertools
from typing import Iterator, Callable
import torch
from .tokenizer import Tokenizer


# ----------------------------------------------------------------------
# Per-dataset formatters: take a raw HF example dict, return plain text.
# ----------------------------------------------------------------------
def _fmt_plain(field: str) -> Callable[[dict], str]:
    def f(ex: dict) -> str:
        return ex.get(field) or ""
    return f


def _fmt_daily_dialog(ex: dict) -> str:
    # `dialog` is a list of utterances; speakers alternate starting User.
    turns = ex.get("dialog") or []
    out = []
    for i, t in enumerate(turns):
        role = "User" if i % 2 == 0 else "Assistant"
        out.append(f"{role}: {t.strip()}")
    return "\n".join(out)


def _fmt_oasst1(ex: dict) -> str:
    # Single message per row; we'd normally need to walk the tree.
    # Approx: just emit role + text. Buffer concatenation across rows
    # naturally produces multi-turn-looking sequences.
    role = ex.get("role") or ""
    text = ex.get("text") or ""
    if role == "prompter":
        return f"User: {text.strip()}"
    elif role == "assistant":
        return f"Assistant: {text.strip()}"
    return text.strip()


def _fmt_slimorca(ex: dict) -> str:
    # SlimOrca: "conversations" = list of {from: human|gpt|system, value: ...}
    conv = ex.get("conversations") or []
    out = []
    for turn in conv:
        src = turn.get("from", "")
        val = (turn.get("value") or "").strip()
        if src == "system":
            out.append(f"System: {val}")
        elif src == "human":
            out.append(f"User: {val}")
        elif src == "gpt":
            out.append(f"Assistant: {val}")
    return "\n".join(out)


def _fmt_hh(ex: dict) -> str:
    # Anthropic/hh-rlhf: "chosen" already contains "\n\nHuman: ... \n\nAssistant: ..."
    # Convert to our convention.
    text = (ex.get("chosen") or "").strip()
    return (text.replace("\n\nHuman:", "\nUser:")
                .replace("\n\nAssistant:", "\nAssistant:")
                .strip())


# ----------------------------------------------------------------------
# Dataset chains. Each entry: (path, config, split, formatter, label).
# ----------------------------------------------------------------------
TEXT_CHAIN = [
    ("roneneldan/TinyStories",      None,                "train", _fmt_plain("text"), "TinyStories"),
    ("HuggingFaceTB/smollm-corpus", "cosmopedia-v2",     "train", _fmt_plain("text"), "Cosmopedia-v2"),
    ("wikitext",                    "wikitext-2-raw-v1", "train", _fmt_plain("text"), "wikitext"),
]

CHAT_CHAIN = [
    ("Open-Orca/SlimOrca",     None, "train", _fmt_slimorca,     "SlimOrca"),
    ("daily_dialog",           None, "train", _fmt_daily_dialog, "DailyDialog"),
    ("Anthropic/hh-rlhf",      None, "train", _fmt_hh,           "hh-rlhf"),
    ("OpenAssistant/oasst1",   None, "train", _fmt_oasst1,       "oasst1"),
]


def _try_load_streaming(path, config, split):
    from datasets import load_dataset
    if config:
        return load_dataset(path, config, split=split, streaming=True)
    return load_dataset(path, split=split, streaming=True)


def open_stream(mode: str = "text"):
    """Return (iterable_dataset, formatter, label).

    Tries each dataset in the chain until one succeeds.
    """
    chain = TEXT_CHAIN if mode == "text" else CHAT_CHAIN
    last_err = None
    for path, cfg, split, formatter, label in chain:
        try:
            ds = _try_load_streaming(path, cfg, split)
            print(f"[data:{mode}] using {label} ({path}"
                  + (f":{cfg}" if cfg else "") + ")")
            return ds, formatter, label
        except Exception as e:  # noqa: BLE001
            last_err = e
            print(f"[data:{mode}] could not load {path}: "
                  f"{type(e).__name__}: {e}")
    raise RuntimeError(
        f"No dataset could be opened for mode={mode}. Last error: {last_err}")


# ----------------------------------------------------------------------
# Single-stream tokenized window iterator (infinite + self-healing).
# ----------------------------------------------------------------------
def _stream_iterator(tokenizer: Tokenizer, ctx_len: int, mode: str,
                     buffer_size: int = 8192) -> Iterator[list[int]]:
    buf: list[int] = []
    eos = tokenizer.eos_id
    reconnects = 0
    while True:
        try:
            ds, formatter, _ = open_stream(mode=mode)
            for ex in ds:
                text = formatter(ex)
                if not text:
                    continue
                ids = tokenizer.encode(text)
                buf.extend(ids)
                buf.append(eos)
                while len(buf) >= ctx_len + 1:
                    window = buf[: ctx_len + 1]
                    buf = buf[ctx_len:]
                    yield window
                if len(buf) > buffer_size:
                    buf = buf[-buffer_size:]
            reconnects += 1
            print(f"[data:{mode}] stream exhausted, reopening "
                  f"(reconnect #{reconnects})")
        except Exception as e:  # noqa: BLE001
            reconnects += 1
            wait = min(60, 5 * reconnects)
            print(f"[data:{mode}] stream error: {type(e).__name__}: {e} — "
                  f"reconnecting in {wait}s (reconnect #{reconnects})")
            import time
            time.sleep(wait)


def token_window_iterator(tokenizer: Tokenizer, ctx_len: int,
                          buffer_size: int = 8192, seed: int = 0,
                          mode: str = "text",
                          chat_ratio: float = 0.75
                          ) -> Iterator[list[int]]:
    """Yield fixed-length token windows of size ctx_len + 1.

    mode:
      "text" - text-only chain (default, backwards-compatible)
      "chat" - chat-only chain
      "mix"  - probabilistically interleave; `chat_ratio` of windows from chat.
    """
    if mode in ("text", "chat"):
        yield from _stream_iterator(tokenizer, ctx_len, mode, buffer_size)
        return
    if mode != "mix":
        raise ValueError(f"unknown mode: {mode}")
    # Mix mode: pull from both streams, choose per-window by ratio.
    rng = random.Random(seed)
    text_it = _stream_iterator(tokenizer, ctx_len, "text", buffer_size)
    chat_it = _stream_iterator(tokenizer, ctx_len, "chat", buffer_size)
    while True:
        if rng.random() < chat_ratio:
            yield next(chat_it)
        else:
            yield next(text_it)


def batch_iterator(tokenizer: Tokenizer, ctx_len: int, batch_size: int,
                   seed: int = 0, mode: str = "text",
                   chat_ratio: float = 0.75) -> Iterator[torch.Tensor]:
    it = token_window_iterator(tokenizer, ctx_len, seed=seed,
                               mode=mode, chat_ratio=chat_ratio)
    while True:
        batch = list(itertools.islice(it, batch_size))
        if len(batch) < batch_size:
            return
        t = torch.tensor(batch, dtype=torch.long)  # (B, ctx_len+1)
        yield t

