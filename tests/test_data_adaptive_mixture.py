"""Tests for adaptive mixture + DAR domain labels in data pipeline.

PR2 wiring: data-side support for
  (a) `ratio_ref: Callable[[], float]` overriding `chat_ratio` per-window
  (b) `with_labels: bool` yielding (window, source_id) tuples for DAR

We monkeypatch `_stream_iterator` to avoid needing network/tokenizers.
text stream emits tokens [0,0,0,...]; chat stream emits tokens [1,1,1,...].
That gives a trivial way to detect which stream a window came from.
"""
from __future__ import annotations

import itertools

import pytest


@pytest.fixture
def patched_streams(monkeypatch):
    """Replace _stream_iterator with deterministic 0/1 token streams."""
    from neuroslm import data as data_mod

    def fake_stream(_tok, ctx_len, mode, _buf=8192):
        tok = 0 if mode == "text" else 1
        while True:
            yield [tok] * (ctx_len + 1)

    monkeypatch.setattr(data_mod, "_stream_iterator", fake_stream)
    return data_mod


def _window_label(w):
    """Recover source label from a fake window: all-zero=text(0), all-one=chat(1)."""
    return 1 if w[0] == 1 else 0


def test_token_window_iterator_ratio_zero_all_text(patched_streams):
    it = patched_streams.token_window_iterator(
        tokenizer=None, ctx_len=8, mode="mix", chat_ratio=0.0
    )
    wins = list(itertools.islice(it, 20))
    assert all(_window_label(w) == 0 for w in wins)


def test_token_window_iterator_ratio_one_all_chat(patched_streams):
    it = patched_streams.token_window_iterator(
        tokenizer=None, ctx_len=8, mode="mix", chat_ratio=1.0
    )
    wins = list(itertools.islice(it, 20))
    assert all(_window_label(w) == 1 for w in wins)


def test_token_window_iterator_ratio_ref_overrides_static(patched_streams):
    """When ratio_ref is provided, it overrides static chat_ratio per-window."""
    box = {"r": 0.0}
    it = patched_streams.token_window_iterator(
        tokenizer=None, ctx_len=8, mode="mix",
        chat_ratio=0.5, ratio_ref=lambda: box["r"],
    )
    # Drain a few — should all be text (ratio_ref=0).
    pre = list(itertools.islice(it, 30))
    assert all(_window_label(w) == 0 for w in pre)

    # Flip ratio_ref → 1.0 mid-stream; subsequent windows must be chat.
    box["r"] = 1.0
    post = list(itertools.islice(it, 30))
    assert all(_window_label(w) == 1 for w in post)


def test_token_window_iterator_with_labels_yields_tuples(patched_streams):
    it = patched_streams.token_window_iterator(
        tokenizer=None, ctx_len=8, mode="mix",
        chat_ratio=0.5, with_labels=True, seed=7,
    )
    pairs = list(itertools.islice(it, 50))
    assert all(isinstance(p, tuple) and len(p) == 2 for p in pairs)
    # Labels must match recovered window source
    for win, lab in pairs:
        assert lab == _window_label(win)
    # And both classes appear
    labs = {lab for _, lab in pairs}
    assert labs == {0, 1}


def test_token_window_iterator_text_mode_ignores_ratio_ref(patched_streams):
    """Non-mix modes must not call ratio_ref (back-compat guarantee)."""
    calls = {"n": 0}
    def ref():
        calls["n"] += 1
        return 0.5

    it = patched_streams.token_window_iterator(
        tokenizer=None, ctx_len=8, mode="text", ratio_ref=ref,
    )
    _ = list(itertools.islice(it, 5))
    assert calls["n"] == 0


# ── batch_iterator: end-to-end with labels ──────────────────────────────────

def test_batch_iterator_with_labels_returns_batch_and_label_tensor(patched_streams):
    import torch
    it = patched_streams.batch_iterator(
        tokenizer=None, ctx_len=8, batch_size=4,
        mode="mix", chat_ratio=0.5, with_labels=True, seed=11,
    )
    batch, labels = next(it)
    assert isinstance(batch, torch.Tensor)
    assert isinstance(labels, torch.Tensor)
    assert batch.shape == (4, 9)
    assert labels.shape == (4,)
    assert labels.dtype == torch.long
    # Each label must agree with its window's content
    for i in range(4):
        expected = 1 if batch[i, 0].item() == 1 else 0
        assert labels[i].item() == expected


def test_batch_iterator_ratio_ref_drives_chat_fraction(patched_streams):
    import torch
    box = {"r": 0.0}
    it = patched_streams.batch_iterator(
        tokenizer=None, ctx_len=8, batch_size=8, mode="mix",
        chat_ratio=0.5, ratio_ref=lambda: box["r"],
        with_labels=True, seed=3,
    )
    # ratio=0 → expect ~0 chat labels
    b0, l0 = next(it)
    assert l0.sum().item() == 0

    # Bump to 1.0; iterator must observe the change.
    box["r"] = 1.0
    b1, l1 = next(it)
    assert l1.sum().item() == 8
