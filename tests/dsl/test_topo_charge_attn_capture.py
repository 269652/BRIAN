# -*- coding: utf-8 -*-
"""LanguageCortex attention-output capture contract.

Phase 1.3a: opt-in forward hooks on each block's self.attn that stash
the per-(B, H, T, head_dim) attention output into
LanguageCortex._last_attn_per_layer for downstream consumption by
TopoChargeDiagnostic. Hooks are inactive by default (no memory or
gradient overhead). Per CLAUDE.md sec 1b, the hook approach avoids
intrusive modification of CausalSelfAttention / DiffTransformerBlock
which are in the hot path.

Contracts:
  1. Default off -> _last_attn_per_layer == [] at all times.
  2. Enable -> after forward, _last_attn_per_layer has L tensors of
     shape (B, n_heads, T, head_dim).
  3. Each forward CLEARS the list at start (no leak across calls).
  4. Hooks are removable (cleanup contract for tests that build many
     LanguageCortex instances).
"""
from __future__ import annotations

import pytest
import torch

from neuroslm.modules.language import LanguageCortex


def _build(enable_capture: bool, n_layers: int = 3) -> LanguageCortex:
    return LanguageCortex(
        vocab_size=64,
        d_hidden=32,
        d_sem=16,
        n_layers=n_layers,
        n_heads=4,
        max_ctx=16,
        baseline=True,   # only standard TransformerBlocks (homogeneous .attn)
        enable_attn_capture=enable_capture,
    )


class TestAttnCaptureFlag:
    def test_default_off_no_capture(self):
        m = _build(enable_capture=False)
        ids = torch.randint(0, 64, (2, 8))
        with torch.no_grad():
            m(ids)
        assert m._last_attn_per_layer == []

    def test_enabled_captures_one_tensor_per_block(self):
        m = _build(enable_capture=True, n_layers=3)
        ids = torch.randint(0, 64, (2, 8))
        with torch.no_grad():
            m(ids)
        assert len(m._last_attn_per_layer) == 3

    def test_captured_shape_is_B_H_T_head_dim(self):
        m = _build(enable_capture=True, n_layers=2)
        B, T = 3, 5
        ids = torch.randint(0, 64, (B, T))
        with torch.no_grad():
            m(ids)
        for layer_t in m._last_attn_per_layer:
            assert layer_t.shape == (B, 4, T, 32 // 4)  # (B, H, T, head_dim)

    def test_repeated_forward_resets_list(self):
        m = _build(enable_capture=True, n_layers=2)
        ids = torch.randint(0, 64, (1, 4))
        with torch.no_grad():
            m(ids)
        first = len(m._last_attn_per_layer)
        with torch.no_grad():
            m(ids)
        second = len(m._last_attn_per_layer)
        assert first == second == 2, (
            "every forward must clear and re-populate the list; "
            "leaking across calls would grow it unboundedly"
        )

    def test_off_means_no_memory_or_grad_overhead(self):
        """When the flag is off, no hook is registered. We can't easily
        measure memory, but we CAN assert the captured tensor list is
        always empty -- a stub that registers a hook but skips the
        append would still pass test_default_off_no_capture, but
        would not pass this stricter contract."""
        m = _build(enable_capture=False)
        # Inspect the modules: no block.attn should have a registered
        # forward hook from us.
        for blk in m.blocks:
            attn = getattr(blk, "attn", None)
            if attn is None:
                continue
            # _forward_hooks is the underlying registry; should be empty
            # OR not contain any hook tagged with our capture flag.
            hooks = list(attn._forward_hooks.values())
            assert all(
                getattr(h, "_topo_charge_capture", False) is False
                for h in hooks
            ), "no topo-charge capture hook should be installed when off"

    def test_on_installs_tagged_hook(self):
        """Companion to the previous test: when on, EXACTLY one
        tagged hook per block.attn (no duplicates across rebuilds)."""
        m = _build(enable_capture=True, n_layers=2)
        for blk in m.blocks:
            attn = getattr(blk, "attn", None)
            if attn is None:
                continue
            tagged = [
                h for h in attn._forward_hooks.values()
                if getattr(h, "_topo_charge_capture", False) is True
            ]
            assert len(tagged) == 1, (
                f"expected exactly one capture hook per block.attn; "
                f"got {len(tagged)}"
            )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
