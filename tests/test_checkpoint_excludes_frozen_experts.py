# -*- coding: utf-8 -*-
"""Regression tests for the 2 GiB LFS checkpoint bloat (instance 41049651).

Production failure trace
========================

::

    [ckpt_push] push attempt 1 rejected; rebasing on origin/master and retrying ...
    [ckpt_push] push attempt 2 rejected; rebasing on origin/master and retrying ...
    [ckpt_push] push attempt 3 rejected; rebasing on origin/master and retrying ...
    [ckpt_push] push attempt 4 rejected; rebasing on origin/master and retrying ...
    [ckpt_push] push attempt 5 rejected; rebasing on origin/master and retrying ...
    remote: error: GH001: Large files detected. You may want to try Git Large
        File Storage - https://git-lfs.github.com.
    remote: error: File lfs_checkpoints/.../step1000.pt is N MB; this exceeds
        GitHub's file size limit of 100.00 MB
    Size must be less than or equal to 2147483648: [422]

The user diagnosed this as "rebase doesn't work". It is not — the trace
shows rebase ran 5 times, all rejected by GitHub LFS's hard 2 GiB
per-file limit (``2147483648`` bytes exactly).

The actual cause is that ``BRIANHarness.save_checkpoint`` calls
``self.state_dict()`` blindly, which serialises the three frozen
HuggingFace experts (``multi_cortex.experts.<i>.lm.*``):

  * ``smollm2_360m``               ~720 MB fp32
  * ``microsoft/CodeGPT-small-py`` ~150 MB fp32
  * ``Qwen/Qwen2.5-0.5B``          ~2 GB fp32

Combined with the AdamW optimizer state (2× trainable param count for
momentum + variance) and the genetics/transmitter buffers, every save
exceeds the LFS 2 GiB single-file ceiling — and contributes zero
information, because every frozen weight is byte-for-byte identical
to the HuggingFace cache that ``_build_multi_cortex`` will re-load
on resume anyway.

What the fix is
===============

``save_checkpoint`` must filter the model state-dict to exclude any
subtree whose source-of-truth lives outside the checkpoint. Concretely:

  * keys under ``multi_cortex.experts.`` are dropped on save
    (HF models, re-loaded by ``_build_multi_cortex`` from the local
    HF cache or downloaded once per machine)

  * ``load_checkpoint`` switches to ``strict=False`` so checkpoints
    saved by the new code (without expert keys) load cleanly into a
    harness that already has those experts attached from init

  * any other missing key (i.e. one that is NOT under the allow-listed
    expert subtree) still raises — accidental losses of trainable
    weight must remain loud

These tests pin the exact contract so the bloat cannot regress.

Implementation notes
====================

We do NOT instantiate real HF experts in this file — that would burn
seconds on download/load and add a transformers dependency to a
correctness test. Instead we attach a fake ``nn.Module`` tree to
``h.multi_cortex`` that mimics the production layout
(``experts: nn.ModuleList`` + ``router: nn.Module``) so the prefix
filter is exercised end-to-end through the real ``save_checkpoint``
code path.
"""
from __future__ import annotations

from pathlib import Path

import pytest
import torch
import torch.nn as nn

from neuroslm.dsl.codegen import CodeGenerator
from neuroslm.dsl.multifile import compile_folder
from neuroslm.dsl.training_config import TrainingConfig
from neuroslm.harness import BRIANHarness


_ARCH_ROOT = Path(__file__).resolve().parent.parent / "architectures" / "master"


# ──────────────────────────────────────────────────────────────────────
# Fixtures
# ──────────────────────────────────────────────────────────────────────


@pytest.fixture(scope="module")
def _compiled_circuit_factory():
    """Module-scoped circuit IR (compile is slow); per-test circuits are
    constructed from it so each test gets a fresh submodule tree.
    Mirrors the helper in ``test_checkpoint_path_layout.py``."""
    ir = compile_folder(_ARCH_ROOT)
    Cls = CodeGenerator(ir, module_name="CkptFilterTestCircuit").compile_to_module()
    return lambda: Cls(d_sem=64)


def _fresh_harness(circuit_factory) -> BRIANHarness:
    """Build a minimal harness — no multi_cortex, no genetics. Tests
    attach a fake ``multi_cortex`` post-hoc to exercise the filter."""
    cfg = TrainingConfig()
    cfg.genetics.enabled = False
    cfg.grad_accum_steps = 1
    return BRIANHarness(
        circuit=circuit_factory(),
        vocab_size=512,
        d_sem=64,
        training_config=cfg,
    )


class _FakeExpert(nn.Module):
    """Mimics a frozen ``LMExpert`` — an ``nn.Module`` named ``lm`` whose
    parameters all have ``requires_grad=False``. Big enough that we can
    see the byte-savings in the size-sanity test, small enough to keep
    the test under a second."""

    def __init__(self, hidden: int = 256, vocab: int = 1024):
        super().__init__()
        self.lm = nn.Sequential(
            nn.Linear(hidden, hidden),
            nn.Linear(hidden, hidden),
            nn.Linear(hidden, vocab),
        )
        for p in self.lm.parameters():
            p.requires_grad = False


class _FakeMultiCortex(nn.Module):
    """Mimics ``LMExpertEnsemble`` layout: ``experts`` (frozen, big) +
    ``router`` (trainable, small). The router stands in for both the
    real ``ThalamicRouter`` and any other trainable head — the filter
    must preserve it intact while dropping all of ``experts``."""

    def __init__(self, n_experts: int = 3, hidden: int = 256, vocab: int = 1024):
        super().__init__()
        self.experts = nn.ModuleList(
            [_FakeExpert(hidden=hidden, vocab=vocab) for _ in range(n_experts)]
        )
        self.router = nn.Linear(hidden, n_experts)


def _attach_fake_multi_cortex(h: BRIANHarness, **kw) -> _FakeMultiCortex:
    """Install a fake ``multi_cortex`` on the harness. ``nn.Module.__setattr__``
    registers it as a real submodule so it appears in ``h.state_dict()``."""
    fake = _FakeMultiCortex(**kw)
    h.multi_cortex = fake
    return fake


# ──────────────────────────────────────────────────────────────────────
# Contract 1 — frozen expert subtree is excluded from saved checkpoint
# ──────────────────────────────────────────────────────────────────────


class TestSaveCheckpointExcludesFrozenExperts:
    """``save_checkpoint`` must drop the entire ``multi_cortex.experts.*``
    subtree from the persisted state-dict. These weights are
    re-loaded from HuggingFace on resume; saving them is the root
    cause of the 2 GiB LFS limit blow-up."""

    def test_expert_keys_absent_from_saved_payload(
        self, _compiled_circuit_factory, tmp_path
    ):
        h = _fresh_harness(_compiled_circuit_factory)
        _attach_fake_multi_cortex(h)

        # Sanity: before save, the experts ARE in state_dict (so this
        # test actually proves the filter does work, not that the
        # experts were already absent).
        live_keys = list(h.state_dict().keys())
        assert any(k.startswith("multi_cortex.experts.") for k in live_keys), (
            "precondition: fake experts must appear in live state_dict "
            "before save (otherwise the test is vacuous)"
        )

        ckpt = tmp_path / "filtered.pt"
        h.save_checkpoint(str(ckpt), step=42)

        payload = torch.load(str(ckpt), weights_only=False, map_location="cpu")
        saved_keys = list(payload["model"].keys())
        leaked = [k for k in saved_keys if k.startswith("multi_cortex.experts.")]
        assert leaked == [], (
            f"frozen experts leaked into the checkpoint payload — "
            f"this re-creates the 2 GiB LFS bloat. Offending keys "
            f"(first 5): {leaked[:5]}"
        )

    def test_step_and_metadata_preserved(
        self, _compiled_circuit_factory, tmp_path
    ):
        """Filtering ``experts`` must not break the rest of the payload —
        step, vocab_size, d_sem, sink_population, optimizer all stay."""
        h = _fresh_harness(_compiled_circuit_factory)
        _attach_fake_multi_cortex(h)

        ckpt = tmp_path / "meta.pt"
        h.save_checkpoint(str(ckpt), step=99)

        payload = torch.load(str(ckpt), weights_only=False, map_location="cpu")
        assert payload["step"] == 99
        assert payload["vocab_size"] == 512
        assert payload["d_sem"] == 64
        assert "sink_population" in payload
        assert "model" in payload

    def test_router_and_other_trainables_are_preserved(
        self, _compiled_circuit_factory, tmp_path
    ):
        """The filter must be surgical: only ``multi_cortex.experts.*``
        is dropped. The router (trainable) and every non-multi_cortex
        weight must survive."""
        h = _fresh_harness(_compiled_circuit_factory)
        _attach_fake_multi_cortex(h)

        ckpt = tmp_path / "router_kept.pt"
        h.save_checkpoint(str(ckpt), step=0)

        payload = torch.load(str(ckpt), weights_only=False, map_location="cpu")
        saved_keys = set(payload["model"].keys())

        # Router survives
        assert any(k.startswith("multi_cortex.router.") for k in saved_keys), (
            f"multi_cortex.router was dropped — the filter is too broad. "
            f"Saved multi_cortex.* keys: "
            f"{[k for k in saved_keys if k.startswith('multi_cortex.')][:10]}"
        )

        # Circuit (the rest of the brain) survives
        live = set(h.state_dict().keys())
        non_expert_live = {
            k for k in live if not k.startswith("multi_cortex.experts.")
        }
        # Every non-expert key in the live model must be in the saved
        # payload — no over-eager filtering.
        missing = non_expert_live - saved_keys
        assert missing == set(), (
            f"non-expert weights were silently dropped: {sorted(missing)[:10]}"
        )


# ──────────────────────────────────────────────────────────────────────
# Contract 2 — load round-trip with the new (filtered) format
# ──────────────────────────────────────────────────────────────────────


class TestLoadCheckpointHandlesMissingExpertKeys:
    """A harness with multi_cortex attached must load a checkpoint
    that has NO expert keys (i.e. one saved by the new code) without
    error. The experts are already attached from ``_build_multi_cortex``,
    so the load just needs to be non-strict about the expert subtree."""

    def test_load_after_filtered_save_does_not_raise(
        self, _compiled_circuit_factory, tmp_path
    ):
        h1 = _fresh_harness(_compiled_circuit_factory)
        _attach_fake_multi_cortex(h1)

        ckpt = tmp_path / "rt.pt"
        h1.save_checkpoint(str(ckpt), step=11)

        h2 = _fresh_harness(_compiled_circuit_factory)
        _attach_fake_multi_cortex(h2)  # simulates _build_multi_cortex at init

        # MUST NOT raise — saved payload has no experts, harness has experts
        step = h2.load_checkpoint(str(ckpt))
        assert step == 11

    def test_experts_remain_functional_after_load(
        self, _compiled_circuit_factory, tmp_path
    ):
        """The expert weights on h2 must be the ones from its own
        ``_build_multi_cortex`` (i.e. the fake attached at init), NOT
        re-initialised to zeros by load_state_dict. Confirms that the
        load is non-destructive."""
        h1 = _fresh_harness(_compiled_circuit_factory)
        _attach_fake_multi_cortex(h1)

        ckpt = tmp_path / "rt2.pt"
        h1.save_checkpoint(str(ckpt), step=5)

        h2 = _fresh_harness(_compiled_circuit_factory)
        fake2 = _attach_fake_multi_cortex(h2)

        # Snapshot expert weight BEFORE load
        snap_before = fake2.experts[0].lm[0].weight.detach().clone()

        h2.load_checkpoint(str(ckpt))

        snap_after = fake2.experts[0].lm[0].weight.detach().clone()
        # The load must NOT have touched the expert weights at all —
        # they came from h2's own init (the HF cache in production).
        assert torch.equal(snap_before, snap_after), (
            "expert weights were modified by load_checkpoint; the load "
            "must leave the externally-sourced subtree alone"
        )


# ──────────────────────────────────────────────────────────────────────
# Contract 3 — backward compatibility with old (bloated) checkpoints
# ──────────────────────────────────────────────────────────────────────


class TestLoadCheckpointBackwardCompat:
    """Checkpoints saved by the OLD code DO contain expert keys.
    The new load path must accept them — strict=False must allow
    'unexpected' keys (i.e. keys in the payload that the harness
    is going to overwrite anyway from its own init)."""

    def test_load_old_format_with_expert_keys_succeeds(
        self, _compiled_circuit_factory, tmp_path
    ):
        # Manually construct an "old" payload by saving the full
        # state_dict (no filter). This is bit-for-bit what pre-fix
        # save_checkpoint produced.
        h1 = _fresh_harness(_compiled_circuit_factory)
        _attach_fake_multi_cortex(h1)

        ckpt = tmp_path / "old.pt"
        old_payload = {
            "step": 7,
            "model": h1.state_dict(),   # NO FILTER — old format
            "vocab_size": h1.vocab_size,
            "d_sem": h1.d_sem,
            "sink_population": h1.sink_population,
        }
        torch.save(old_payload, str(ckpt))

        # Confirm the old format does have expert keys (otherwise
        # the back-compat scenario is fake).
        check = torch.load(str(ckpt), weights_only=False, map_location="cpu")
        assert any(
            k.startswith("multi_cortex.experts.") for k in check["model"]
        ), "test setup: old payload must contain expert keys"

        # Load it with a fresh harness — must succeed.
        h2 = _fresh_harness(_compiled_circuit_factory)
        _attach_fake_multi_cortex(h2)
        step = h2.load_checkpoint(str(ckpt))
        assert step == 7


# ──────────────────────────────────────────────────────────────────────
# Contract 4 — accidental loss of trainable weight is still loud
# ──────────────────────────────────────────────────────────────────────


class TestLoadCheckpointStillRejectsUnexpectedMissingKeys:
    """The fix's ``strict=False`` is narrow: missing keys are only
    tolerated when they live under the allow-listed expert subtree.
    If a real trainable weight goes missing (e.g. because the user
    pointed at a wrong-arch checkpoint), the load MUST still raise
    so the operator notices instead of silently training a zero-init
    half of the model."""

    def test_missing_non_expert_key_raises(
        self, _compiled_circuit_factory, tmp_path
    ):
        h1 = _fresh_harness(_compiled_circuit_factory)
        _attach_fake_multi_cortex(h1)

        # Build a payload that omits a circuit weight (not under the
        # allow-listed prefix). Simulates a cross-architecture mistake.
        bad_state = dict(h1.state_dict())
        # Find a non-expert, non-multi_cortex key and drop it.
        droppable = [
            k for k in bad_state
            if not k.startswith("multi_cortex.")
            and not k.startswith("_genetics_orch.")
            and not k.startswith("_transmitter_sys.")
        ]
        assert droppable, "test setup: expected at least one circuit weight"
        victim = droppable[0]
        del bad_state[victim]

        ckpt = tmp_path / "bad.pt"
        torch.save({
            "step": 0,
            "model": bad_state,
            "vocab_size": h1.vocab_size,
            "d_sem": h1.d_sem,
            "sink_population": h1.sink_population,
        }, str(ckpt))

        h2 = _fresh_harness(_compiled_circuit_factory)
        _attach_fake_multi_cortex(h2)
        with pytest.raises((RuntimeError, KeyError)) as excinfo:
            h2.load_checkpoint(str(ckpt))
        # The error message should be informative about WHAT was missing.
        assert victim in str(excinfo.value) or "missing" in str(excinfo.value).lower(), (
            f"load error should name the missing key {victim!r}, got: "
            f"{excinfo.value!r}"
        )


# ──────────────────────────────────────────────────────────────────────
# Contract 5 — size sanity (filter actually saves bytes)
# ──────────────────────────────────────────────────────────────────────


class TestCheckpointSizeSanity:
    """Bytes are what matter. With the filter on, a checkpoint
    containing 3 fake experts (each ~1 MB of frozen weight) must be
    materially smaller than without the filter."""

    def test_filtered_checkpoint_is_smaller_than_unfiltered(
        self, _compiled_circuit_factory, tmp_path
    ):
        h = _fresh_harness(_compiled_circuit_factory)
        # Big-ish fakes so the difference is unambiguous: 3 experts ×
        # 3 linears × (256² + 256×1024) ≈ 3 × 0.6 MB ≈ 2 MB total.
        _attach_fake_multi_cortex(h, hidden=256, vocab=1024)

        # Filtered (new code)
        ckpt_new = tmp_path / "new.pt"
        h.save_checkpoint(str(ckpt_new), step=0)

        # Unfiltered (old code) — save the raw state_dict ourselves
        ckpt_old = tmp_path / "old.pt"
        torch.save({
            "step": 0,
            "model": h.state_dict(),
            "vocab_size": h.vocab_size,
            "d_sem": h.d_sem,
            "sink_population": h.sink_population,
        }, str(ckpt_old))

        new_size = ckpt_new.stat().st_size
        old_size = ckpt_old.stat().st_size

        # The filter must remove at least 1 MB (the bulk of 3×~600 KB
        # frozen experts). In production with real HF weights this is
        # ~1 GB; here we just need a clean signal.
        savings = old_size - new_size
        assert savings >= 1_000_000, (
            f"filter did not save bytes: old={old_size:,}, "
            f"new={new_size:,}, savings={savings:,}"
        )
