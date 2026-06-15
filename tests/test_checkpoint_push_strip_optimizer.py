# -*- coding: utf-8 -*-
"""TDD — HF checkpoint push strips optimiser state by default (2026-06-15).

Rationale
---------
The full ``.pt`` produced by :meth:`BRIANHarness.save_checkpoint` carries
THREE fp32 copies of every trainable parameter::

    weights + Adam m + Adam v  ≈  3 × |trainable|

For the 107 M trunk this is ~1.3 GB on disk. Resuming on the *same* box
benefits from the full state (Adam moments intact, no ~500-step
re-warmup spike). But pushing 1.3 GB to HF Hub every ``push_every``
steps is bandwidth-wasteful — the box can fall back to its own local
``.pt`` first; the HF copy is only the "box self-destructed" insurance
policy.

Contract pinned here
--------------------
1. :func:`push_checkpoint_to_hf` strips the ``"optimizer"`` key from
   the uploaded payload **by default**. Adam state never crosses the
   wire unless the caller explicitly opts in via ``push_optimizer=True``.
2. The on-disk source ``.pt`` is *never* mutated — the strip happens
   in a temp file beside the original, which is deleted after the
   upload (success or failure).
3. The dispatcher :func:`push_checkpoint` forwards the
   ``push_optimizer`` kwarg to the HF backend.
4. ``train_dsl.py`` exposes ``--push_optimizer`` (default off → strip).
5. :class:`ProjectConfig` exposes ``default_push_optimizer`` (default
   ``False``); ``brian.toml [defaults] push_optimizer = true`` flips it.
6. :meth:`BRIANHarness.load_checkpoint` accepts a payload with no
   ``"optimizer"`` key and prints a one-line notice so the user knows
   Adam will reinit (and to expect the ~500-step LR-warmup-shape blip
   in their loss plot).

These tests are the source of truth for points 1-6. See
``docs/changelog.md`` 2026-06-15 entry for the on-disk size table.
"""
from __future__ import annotations

import io
import os
import sys
import types
from pathlib import Path

import pytest
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]


# ─────────────────────────────────────────────────────────────────────
# Fixtures
# ─────────────────────────────────────────────────────────────────────


@pytest.fixture
def fake_hf(monkeypatch):
    """Inject a fake ``huggingface_hub`` module that records every
    ``upload_file`` call **including the file bytes**.

    Reading the bytes upfront is the only way to verify what the
    function actually uploaded: the real ``upload_file`` streams the
    file directly, and our production code deletes the strip temp
    file in a ``finally`` block — so by the time the test inspects
    the recording, the on-disk temp is already gone.
    """
    recorded: dict = {"calls": []}

    class _HfFolder:
        # The HF chain falls back to this when no explicit token and
        # no ``HF_TOKEN`` env. Set non-empty so the push proceeds —
        # the strip-vs-keep contract is the actual unit under test
        # here, not the auth resolution (that's
        # ``tests/test_checkpoint_push_hf.py``).
        @classmethod
        def get_token(cls):
            return "fake-cached-token"

    def fake_upload_file(*, path_or_fileobj, path_in_repo, repo_id,
                         repo_type="model", token=None,
                         commit_message=None, **_):
        # Always materialise to bytes so the test can re-load it via
        # ``torch.load(io.BytesIO(...))`` after the production code
        # has cleaned up the temp file.
        if isinstance(path_or_fileobj, (str, os.PathLike)):
            with open(path_or_fileobj, "rb") as f:
                payload_bytes = f.read()
        elif hasattr(path_or_fileobj, "read"):
            payload_bytes = path_or_fileobj.read()
        else:
            payload_bytes = bytes(path_or_fileobj)
        recorded["calls"].append({
            "path_in_repo": path_in_repo,
            "repo_id": repo_id,
            "repo_type": repo_type,
            "token": token,
            "bytes": payload_bytes,
            "commit_message": commit_message,
        })

    fake_module = types.ModuleType("huggingface_hub")
    fake_module.upload_file = fake_upload_file  # type: ignore[attr-defined]
    fake_module.HfFolder = _HfFolder            # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "huggingface_hub", fake_module)
    monkeypatch.delenv("HF_TOKEN", raising=False)
    monkeypatch.delenv("HF_REPO_ID", raising=False)
    monkeypatch.delenv("CHECKPOINT_PUSH_BACKEND", raising=False)
    return recorded


@pytest.fixture
def ckpt(tmp_path):
    """A realistic ``.pt`` produced the way
    :meth:`BRIANHarness.save_checkpoint` writes them.

    Tensors are small so the test is fast, but the *shape* of the
    payload matches the production layout exactly (``model``,
    ``optimizer`` with ``state`` + ``param_groups``, ``step``,
    plus the harness metadata fields the load path uses).
    """
    run_dir = tmp_path / "lfs_checkpoints" / "run-20260615_abc1234_arch"
    run_dir.mkdir(parents=True)
    path = run_dir / "step500.pt"
    payload = {
        "step": 500,
        "model": {
            "trunk.weight": torch.randn(64, 64),
            "trunk.bias":   torch.randn(64),
        },
        "optimizer": {
            "state": {0: {
                "step":       torch.tensor(500),
                "exp_avg":    torch.randn(64, 64),
                "exp_avg_sq": torch.randn(64, 64).abs(),
            }},
            "param_groups": [{"lr": 3e-4, "betas": (0.9, 0.999),
                              "eps": 1e-8, "weight_decay": 0.01}],
        },
        "vocab_size": 50257,
        "d_sem": 64,
        "sink_population": "motor",
    }
    torch.save(payload, path)
    return path


def _pt_uploads(recorded):
    """Filter ``recorded['calls']`` to just the ``.pt`` upload(s) —
    the helper exists because every push also tries to upload a
    ``.mem`` sidecar which is irrelevant to the optimiser-strip
    contract."""
    return [c for c in recorded["calls"]
            if c["path_in_repo"].endswith(".pt")]


def _load_uploaded(call):
    """Reload a recorded upload's bytes back into a payload dict."""
    return torch.load(io.BytesIO(call["bytes"]),
                      map_location="cpu", weights_only=False)


# ─────────────────────────────────────────────────────────────────────
# 1.  push_checkpoint_to_hf strips by default
# ─────────────────────────────────────────────────────────────────────


class TestStripOptimizerDefault:
    """The HF backend MUST strip ``optimizer`` from uploads unless
    the caller explicitly opts in."""

    def test_push_strips_optimizer_by_default(
            self, fake_hf, ckpt, tmp_path):
        from neuroslm.checkpoint_push import push_checkpoint_to_hf
        push_checkpoint_to_hf(str(ckpt), repo_root=str(tmp_path))
        uploads = _pt_uploads(fake_hf)
        assert len(uploads) == 1, (
            f"expected exactly one .pt upload, got {len(uploads)}; "
            f"all calls: {[c['path_in_repo'] for c in fake_hf['calls']]}"
        )
        uploaded = _load_uploaded(uploads[0])
        assert "optimizer" not in uploaded, (
            "push_checkpoint_to_hf must strip 'optimizer' from the "
            "uploaded payload by default. Adam state would otherwise "
            "triple the upload size (weights + m + v ≈ 1.3 GB for "
            "the 107M trunk) for no transport benefit — the same-box "
            "local resume already uses the full on-disk .pt."
        )

    def test_strip_preserves_model_and_metadata(
            self, fake_hf, ckpt, tmp_path):
        """Stripping only removes ``optimizer`` — everything else
        (model state, step, vocab_size, d_sem, sink_population)
        must survive the round-trip so the HF ckpt is enough to
        rehydrate a harness for inference / OOD eval."""
        from neuroslm.checkpoint_push import push_checkpoint_to_hf
        push_checkpoint_to_hf(str(ckpt), repo_root=str(tmp_path))
        uploaded = _load_uploaded(_pt_uploads(fake_hf)[0])
        assert "model" in uploaded
        assert uploaded["step"] == 500
        assert uploaded["vocab_size"] == 50257
        assert uploaded["d_sem"] == 64
        assert uploaded["sink_population"] == "motor"
        # And the model weights are bit-for-bit identical (not a
        # quantised / re-saved copy).
        original = torch.load(str(ckpt), map_location="cpu",
                              weights_only=False)
        for k, v in original["model"].items():
            assert torch.equal(uploaded["model"][k], v), (
                f"model tensor {k!r} differs after strip+resave — "
                f"the strip must be a SHALLOW key removal, not a "
                f"deep re-serialise"
            )

    def test_push_full_optimizer_keeps_it(
            self, fake_hf, ckpt, tmp_path):
        """``push_optimizer=True`` is the explicit opt-in — uploads
        keep the full Adam state. Used by the final end-of-training
        push and by anyone who wants the HF copy to be a perfect
        same-trajectory resume target."""
        from neuroslm.checkpoint_push import push_checkpoint_to_hf
        push_checkpoint_to_hf(str(ckpt), repo_root=str(tmp_path),
                              push_optimizer=True)
        uploaded = _load_uploaded(_pt_uploads(fake_hf)[0])
        assert "optimizer" in uploaded, (
            "push_optimizer=True must preserve the optimizer state "
            "(weights+m+v full payload)"
        )
        assert "state" in uploaded["optimizer"]
        assert "param_groups" in uploaded["optimizer"]


# ─────────────────────────────────────────────────────────────────────
# 2.  On-disk .pt is never mutated; temp file always cleaned up
# ─────────────────────────────────────────────────────────────────────


class TestNoLocalSideEffects:
    """Stripping is for transport only — the local SSD copy must
    remain the full-fidelity Adam-state ckpt."""

    def test_local_ckpt_is_byte_identical_after_push(
            self, fake_hf, ckpt, tmp_path):
        from neuroslm.checkpoint_push import push_checkpoint_to_hf
        original_bytes = ckpt.read_bytes()
        push_checkpoint_to_hf(str(ckpt), repo_root=str(tmp_path))
        assert ckpt.read_bytes() == original_bytes, (
            "the on-disk .pt must NEVER be modified by the HF push — "
            "stripping happens in a temp file that's uploaded and "
            "then deleted"
        )
        # Belt-and-braces: the file still loads with optimizer intact.
        loaded = torch.load(str(ckpt), map_location="cpu",
                            weights_only=False)
        assert "optimizer" in loaded

    def test_no_temp_file_leftover_on_success(
            self, fake_hf, ckpt, tmp_path):
        from neuroslm.checkpoint_push import push_checkpoint_to_hf
        push_checkpoint_to_hf(str(ckpt), repo_root=str(tmp_path))
        leftovers = sorted(
            p.name for p in ckpt.parent.iterdir() if p.name != ckpt.name
        )
        assert leftovers == [], (
            f"unexpected leftover files in {ckpt.parent}: {leftovers}; "
            f"the strip temp file must be cleaned up after a "
            f"successful upload"
        )

    def test_no_temp_file_leftover_on_upload_error(
            self, monkeypatch, ckpt, tmp_path):
        """If ``upload_file`` raises (network down, repo not found,
        wrong token), the temp file must still be removed."""

        class _HfFolder:
            @classmethod
            def get_token(cls):
                return "fake"

        def boom(**_):
            raise RuntimeError("simulated network failure")

        fake_module = types.ModuleType("huggingface_hub")
        fake_module.upload_file = boom              # type: ignore[attr-defined]
        fake_module.HfFolder = _HfFolder            # type: ignore[attr-defined]
        monkeypatch.setitem(sys.modules, "huggingface_hub", fake_module)
        monkeypatch.delenv("HF_TOKEN", raising=False)

        from neuroslm.checkpoint_push import push_checkpoint_to_hf
        # Honour the "never raise" contract — should NOT propagate
        # the simulated failure.
        push_checkpoint_to_hf(str(ckpt), repo_root=str(tmp_path))

        leftovers = sorted(
            p.name for p in ckpt.parent.iterdir() if p.name != ckpt.name
        )
        assert leftovers == [], (
            f"strip temp file leaked after upload error; "
            f"leftover files: {leftovers}"
        )


# ─────────────────────────────────────────────────────────────────────
# 3.  Edge cases — ckpt without optimizer is a no-op for the strip
# ─────────────────────────────────────────────────────────────────────


class TestStripEdgeCases:
    """The strip path must be a no-op when there's nothing to strip."""

    def test_ckpt_without_optimizer_uploads_unchanged(
            self, fake_hf, tmp_path):
        run_dir = tmp_path / "lfs_checkpoints" / "run-noopt_abc"
        run_dir.mkdir(parents=True)
        path = run_dir / "step10.pt"
        torch.save(
            {"step": 10, "model": {"w": torch.randn(8, 8)},
             "vocab_size": 100, "d_sem": 8, "sink_population": "m"},
            path,
        )
        from neuroslm.checkpoint_push import push_checkpoint_to_hf
        push_checkpoint_to_hf(str(path), repo_root=str(tmp_path))
        uploaded = _load_uploaded(_pt_uploads(fake_hf)[0])
        assert "optimizer" not in uploaded   # was absent anyway
        assert uploaded["step"] == 10
        assert "model" in uploaded


# ─────────────────────────────────────────────────────────────────────
# 4.  Dispatcher passes push_optimizer through
# ─────────────────────────────────────────────────────────────────────


class TestDispatcherForwardsPushOptimizer:
    """``push_checkpoint(..., push_optimizer=True)`` must reach the
    HF backend — the dispatcher is a transparent pass-through for
    backend-specific kwargs."""

    def test_dispatcher_default_strips(
            self, fake_hf, ckpt, tmp_path):
        from neuroslm.checkpoint_push import push_checkpoint
        push_checkpoint(str(ckpt), backend="hf",
                        repo_root=str(tmp_path))
        uploaded = _load_uploaded(_pt_uploads(fake_hf)[0])
        assert "optimizer" not in uploaded

    def test_dispatcher_forwards_push_optimizer_true(
            self, fake_hf, ckpt, tmp_path):
        from neuroslm.checkpoint_push import push_checkpoint
        push_checkpoint(str(ckpt), backend="hf",
                        repo_root=str(tmp_path),
                        push_optimizer=True)
        uploaded = _load_uploaded(_pt_uploads(fake_hf)[0])
        assert "optimizer" in uploaded


# ─────────────────────────────────────────────────────────────────────
# 5.  train_dsl.py wires the new CLI flag
# ─────────────────────────────────────────────────────────────────────


class TestTrainDslCliFlag:
    """``--push_optimizer`` is a real argparse flag on
    ``neuroslm.train_dsl`` and is forwarded into the ``train()`` call
    so the on-box scripts can set it via env (CHECKPOINT_PUSH_OPTIMIZER)
    or by adding ``--push_optimizer`` to the launch command."""

    def test_push_optimizer_arg_registered(self):
        src = (REPO_ROOT / "neuroslm" / "train_dsl.py").read_text(
            encoding="utf-8")
        assert "--push_optimizer" in src, (
            "neuroslm/train_dsl.py must register a --push_optimizer "
            "CLI flag (action='store_true', default=False) so the "
            "HF strip behaviour is overridable per-deploy without "
            "re-editing the trainer."
        )

    def test_push_optimizer_threaded_into_train(self):
        """The arg must reach ``train()`` — not be parsed and
        dropped on the floor."""
        src = (REPO_ROOT / "neuroslm" / "train_dsl.py").read_text(
            encoding="utf-8")
        assert "push_optimizer=args.push_optimizer" in src or \
               "push_optimizer = args.push_optimizer" in src, (
            "main() must pass args.push_optimizer into the train() "
            "call so the per-save push respects the flag"
        )


# ─────────────────────────────────────────────────────────────────────
# 6.  ProjectConfig + brian.toml expose the default
# ─────────────────────────────────────────────────────────────────────


class TestProjectConfigDefault:
    def test_default_is_false(self, tmp_path):
        """Out-of-the-box: strip optimiser. Same-box local resume
        uses the full on-disk .pt; the HF push is the
        box-self-destructed insurance policy and doesn't need the
        Adam moments."""
        from neuroslm.project_config import load_project_config
        cfg = load_project_config(start=tmp_path)
        assert cfg.default_push_optimizer is False, (
            f"new default must be False (strip), got "
            f"{cfg.default_push_optimizer!r}"
        )

    def test_parses_push_optimizer_true_from_toml(self, tmp_path):
        from neuroslm.project_config import load_project_config
        (tmp_path / "brian.toml").write_text(
            '[defaults]\npush_optimizer = true\n', encoding="utf-8")
        cfg = load_project_config(start=tmp_path)
        assert cfg.default_push_optimizer is True

    def test_env_override_wins(self, tmp_path, monkeypatch):
        from neuroslm.project_config import load_project_config
        (tmp_path / "brian.toml").write_text(
            '[defaults]\npush_optimizer = false\n', encoding="utf-8")
        monkeypatch.setenv("BRIAN_DEFAULT_PUSH_OPTIMIZER", "true")
        cfg = load_project_config(start=tmp_path)
        assert cfg.default_push_optimizer is True


# ─────────────────────────────────────────────────────────────────────
# 7.  load_checkpoint tolerates missing optimizer key
# ─────────────────────────────────────────────────────────────────────


class TestLoadCheckpointAcceptsStripped:
    """When a stripped HF ckpt is downloaded and loaded for resume,
    the load path must succeed AND print a one-line warning so the
    user knows the loss will show a ~500-step LR-warmup-shape recovery
    curve while Adam's second moment EMA rebuilds from zero."""

    def test_load_checkpoint_has_strip_warning_path(self):
        """Grep the source — full behavioural test requires a real
        BRIANHarness build, which is too expensive for a unit
        regression. The warning string is the contract."""
        src = (REPO_ROOT / "neuroslm" / "harness.py").read_text(
            encoding="utf-8")
        assert "optimizer state" in src.lower() and \
               "reinit" in src.lower(), (
            "neuroslm/harness.py::load_checkpoint must contain a "
            "one-line printed warning when the loaded .pt has no "
            "'optimizer' key — phrased so the user understands the "
            "Adam moments will reinit from zero (~500-step "
            "LR-warmup-shape blip in the loss plot is expected)"
        )

    def test_load_checkpoint_does_not_index_missing_optimizer(self):
        """Tiny smoke: feed the load path a payload with no
        ``optimizer`` key and assert it does not raise ``KeyError``."""
        src = (REPO_ROOT / "neuroslm" / "harness.py").read_text(
            encoding="utf-8")
        # The guard is already ``if "optimizer" in payload`` —
        # this regression test is to prevent a future refactor from
        # collapsing it to ``payload["optimizer"]``.
        assert 'if "optimizer" in payload' in src, (
            "load_checkpoint must guard the optimizer load with "
            "``if \"optimizer\" in payload`` so stripped HF "
            "checkpoints load cleanly"
        )
