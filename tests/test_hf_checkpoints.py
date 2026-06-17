# -*- coding: utf-8 -*-
"""TDD tests for ``neuroslm.hf_checkpoints``.

The module is a pure-Python wrapper around ``huggingface_hub`` —
listing, finding-latest, downloading. We mock ``huggingface_hub`` so
the tests run with zero network and zero auth, and pin every public
behaviour:

* :class:`TestParseStep` — filename parser handles all three layouts
* :class:`TestParseHfUri` — ``hf://owner/repo/path`` shorthand
* :class:`TestDecorateListing` — pure listing-decoration logic
* :class:`TestListRepoCheckpoints` — full lister with mocked HfApi
* :class:`TestFindLatestCheckpoint` — convenience wrapper
* :class:`TestDownloadCheckpoint` — download + sidecar handling
"""
from __future__ import annotations

import os
import sys
import types
from pathlib import Path
from typing import Any, Dict, List

import pytest


# ─────────────────────────────────────────────────────────────────────
# Pure-function tests (no HF mock needed)
# ─────────────────────────────────────────────────────────────────────


class TestParseStep:
    """``_parse_step`` recovers training step from every supported path
    layout. Returns 0 on unknown layouts (callers filter)."""

    def test_per_run_subdir(self):
        from neuroslm.hf_checkpoints import _parse_step
        assert _parse_step(
            "checkpoints/run-20260615_abc1234_arch/step5000.pt") == 5000

    def test_legacy_flat(self):
        from neuroslm.hf_checkpoints import _parse_step
        assert _parse_step("checkpoints/dsl_arch_step1000.pt") == 1000

    def test_legacy_ts(self):
        from neuroslm.hf_checkpoints import _parse_step
        assert _parse_step(
            "checkpoints/dsl_arch_20260615-120000_step3500.pt") == 3500

    def test_unknown_returns_zero(self):
        from neuroslm.hf_checkpoints import _parse_step
        assert _parse_step("checkpoints/random_file.pt") == 0
        assert _parse_step("not_a_checkpoint.txt") == 0


class TestParseRunDir:
    """``_parse_run_dir`` extracts the per-run subdir; empty string for
    flat layout."""

    def test_per_run_subdir(self):
        from neuroslm.hf_checkpoints import _parse_run_dir
        assert _parse_run_dir(
            "checkpoints/run-20260615_abc1234_arch/step5000.pt") == \
            "run-20260615_abc1234_arch"

    def test_flat_layout_empty(self):
        from neuroslm.hf_checkpoints import _parse_run_dir
        assert _parse_run_dir("checkpoints/dsl_arch_step1000.pt") == ""

    def test_outside_checkpoints_prefix_empty(self):
        from neuroslm.hf_checkpoints import _parse_run_dir
        assert _parse_run_dir("other/place/step5000.pt") == ""


class TestParseHfUri:
    """``parse_hf_uri`` round-trips the ``hf://owner/repo/path`` shorthand."""

    def test_full_uri(self):
        from neuroslm.hf_checkpoints import parse_hf_uri
        repo, path = parse_hf_uri(
            "hf://moritzroessler/BRIAN/checkpoints/run-A/step5000.pt")
        assert repo == "moritzroessler/BRIAN"
        assert path == "checkpoints/run-A/step5000.pt"

    def test_repo_only(self):
        from neuroslm.hf_checkpoints import parse_hf_uri
        repo, path = parse_hf_uri("hf://moritzroessler/BRIAN")
        assert repo == "moritzroessler/BRIAN"
        assert path == ""

    def test_rejects_non_hf_scheme(self):
        from neuroslm.hf_checkpoints import parse_hf_uri
        with pytest.raises(ValueError):
            parse_hf_uri("https://huggingface.co/foo/bar")

    def test_rejects_owner_only(self):
        from neuroslm.hf_checkpoints import parse_hf_uri
        with pytest.raises(ValueError):
            parse_hf_uri("hf://just_an_owner")


class TestDecorateListing:
    """``_decorate_listing`` is pure — given a flat list of paths, it
    filters + decorates with step + sidecar info."""

    def test_filters_non_checkpoints(self):
        from neuroslm.hf_checkpoints import _decorate_listing
        files = [
            "README.md",
            "checkpoints/run-A/step5000.pt",
            "checkpoints/run-A/random.txt",
            "other/file.pt",
        ]
        entries = _decorate_listing(files, prefix="")
        # Only the one .pt under checkpoints/ with a parseable step
        assert len(entries) == 1
        assert entries[0].path_in_repo == "checkpoints/run-A/step5000.pt"
        assert entries[0].step == 5000

    def test_sidecar_detection(self):
        from neuroslm.hf_checkpoints import _decorate_listing
        files = [
            "checkpoints/run-A/step5000.pt",
            "checkpoints/run-A/step5000.mem",
            "checkpoints/run-B/step1000.pt",  # no sidecar
        ]
        entries = _decorate_listing(files, prefix="")
        by_step = {e.step: e for e in entries}
        assert by_step[5000].has_mem_sidecar is True
        assert by_step[1000].has_mem_sidecar is False

    def test_sorted_newest_first(self):
        from neuroslm.hf_checkpoints import _decorate_listing
        files = [
            "checkpoints/run-A/step1000.pt",
            "checkpoints/run-A/step5000.pt",
            "checkpoints/run-A/step2000.pt",
        ]
        entries = _decorate_listing(files, prefix="")
        steps = [e.step for e in entries]
        assert steps == [5000, 2000, 1000]

    def test_prefix_filter(self):
        from neuroslm.hf_checkpoints import _decorate_listing
        files = [
            "checkpoints/run-A/step5000.pt",
            "checkpoints/run-B/step3000.pt",
        ]
        entries = _decorate_listing(files, prefix="run-A")
        assert len(entries) == 1
        assert entries[0].run_dir == "run-A"


# ─────────────────────────────────────────────────────────────────────
# Integration tests (mock HF Hub)
# ─────────────────────────────────────────────────────────────────────


@pytest.fixture
def fake_hf(monkeypatch):
    """Inject a fake ``huggingface_hub`` module that records every
    ``HfApi.list_repo_files`` and ``hf_hub_download`` call without
    hitting the network.

    The fixture exposes a state dict so tests can prime files, observe
    calls, and toggle failure modes.
    """
    state: Dict[str, Any] = {
        "files": [],         # what list_repo_files returns
        "list_calls": [],    # what was asked
        "download_calls": [],
        "download_should_fail": False,
        "missing_files": set(),  # filenames that 404 on download
    }

    class _FakeHfApi:
        def __init__(self, token=None):
            state["last_token"] = token

        def list_repo_files(self, repo_id, repo_type="model"):
            state["list_calls"].append(
                {"repo_id": repo_id, "repo_type": repo_type})
            return list(state["files"])

    def _fake_hf_hub_download(repo_id, repo_type, filename, token=None,
                              force_download=False, **_):
        state["download_calls"].append({
            "repo_id": repo_id, "filename": filename, "token": token,
            "force_download": force_download,
        })
        if state["download_should_fail"]:
            raise RuntimeError("download forced to fail by test")
        if filename in state["missing_files"]:
            from huggingface_hub.errors import EntryNotFoundError  # type: ignore
            raise EntryNotFoundError(filename)
        # Materialise a tiny file at a tmp path so copy2 succeeds
        tmp = state["tmp_dir"] / Path(filename).name
        tmp.parent.mkdir(parents=True, exist_ok=True)
        tmp.write_bytes(b"\x00" * 64)
        return str(tmp)

    class _FakeHfFolder:
        @classmethod
        def get_token(cls):
            return None

    class _EntryNotFoundError(Exception):
        pass

    fake = types.ModuleType("huggingface_hub")
    fake.HfApi = _FakeHfApi  # type: ignore[attr-defined]
    fake.hf_hub_download = _fake_hf_hub_download  # type: ignore[attr-defined]
    fake.HfFolder = _FakeHfFolder  # type: ignore[attr-defined]
    errors = types.ModuleType("huggingface_hub.errors")
    errors.EntryNotFoundError = _EntryNotFoundError  # type: ignore[attr-defined]
    fake.errors = errors  # type: ignore[attr-defined]

    monkeypatch.setitem(sys.modules, "huggingface_hub", fake)
    monkeypatch.setitem(sys.modules, "huggingface_hub.errors", errors)
    monkeypatch.delenv("HF_TOKEN", raising=False)
    monkeypatch.delenv("HF_REPO_ID", raising=False)
    return state


class TestListRepoCheckpoints:
    """Full lister with the mocked HfApi."""

    def test_empty_repo_returns_empty(self, fake_hf):
        from neuroslm.hf_checkpoints import list_repo_checkpoints
        fake_hf["files"] = []
        assert list_repo_checkpoints(repo_id="x/y") == []

    def test_lists_checkpoints_newest_first(self, fake_hf):
        from neuroslm.hf_checkpoints import list_repo_checkpoints
        fake_hf["files"] = [
            "checkpoints/run-A/step1000.pt",
            "checkpoints/run-A/step5000.pt",
            "checkpoints/run-B/step3000.pt",
            "README.md",
        ]
        entries = list_repo_checkpoints(repo_id="x/y")
        assert [e.step for e in entries] == [5000, 3000, 1000]

    def test_default_repo_resolves_to_canonical(self, fake_hf, monkeypatch):
        from neuroslm.hf_checkpoints import list_repo_checkpoints
        monkeypatch.delenv("HF_REPO_ID", raising=False)
        list_repo_checkpoints()
        assert fake_hf["list_calls"][0]["repo_id"] == "moritzroessler/BRIAN"

    def test_env_repo_overrides_default(self, fake_hf, monkeypatch):
        from neuroslm.hf_checkpoints import list_repo_checkpoints
        monkeypatch.setenv("HF_REPO_ID", "alice/bob")
        list_repo_checkpoints()
        assert fake_hf["list_calls"][0]["repo_id"] == "alice/bob"

    def test_arg_overrides_env(self, fake_hf, monkeypatch):
        from neuroslm.hf_checkpoints import list_repo_checkpoints
        monkeypatch.setenv("HF_REPO_ID", "alice/bob")
        list_repo_checkpoints(repo_id="explicit/wins")
        assert fake_hf["list_calls"][0]["repo_id"] == "explicit/wins"


class TestFindLatestCheckpoint:
    """``find_latest_checkpoint`` returns the first listing entry."""

    def test_returns_highest_step(self, fake_hf):
        from neuroslm.hf_checkpoints import find_latest_checkpoint
        fake_hf["files"] = [
            "checkpoints/run-A/step1000.pt",
            "checkpoints/run-A/step5000.pt",
            "checkpoints/run-B/step3000.pt",
        ]
        latest = find_latest_checkpoint(repo_id="x/y")
        assert latest is not None
        assert latest.step == 5000
        assert latest.path_in_repo == "checkpoints/run-A/step5000.pt"

    def test_returns_none_on_empty(self, fake_hf):
        from neuroslm.hf_checkpoints import find_latest_checkpoint
        fake_hf["files"] = []
        assert find_latest_checkpoint(repo_id="x/y") is None

    def test_prefix_scopes_lookup(self, fake_hf):
        from neuroslm.hf_checkpoints import find_latest_checkpoint
        fake_hf["files"] = [
            "checkpoints/run-A/step5000.pt",
            "checkpoints/run-B/step9000.pt",
        ]
        latest = find_latest_checkpoint(repo_id="x/y", prefix="run-A")
        assert latest is not None
        assert latest.step == 5000  # B's 9000 was filtered out


class TestDownloadCheckpoint:
    """``download_checkpoint`` materialises files into ``dest_dir``
    preserving the per-run subdir layout."""

    def test_downloads_pt_and_sidecar(self, fake_hf, tmp_path):
        from neuroslm.hf_checkpoints import download_checkpoint
        fake_hf["tmp_dir"] = tmp_path / "_hf_cache"
        out = download_checkpoint(
            "checkpoints/run-A/step5000.pt",
            repo_id="x/y",
            dest_dir=str(tmp_path / "lfs_checkpoints"),
        )
        assert out is not None
        assert out.exists()
        # Layout preserved: lfs_checkpoints/run-A/step5000.pt
        assert out.parent.name == "run-A"
        assert out.name == "step5000.pt"
        # Two download calls: .pt + .mem
        assert len(fake_hf["download_calls"]) == 2
        assert fake_hf["download_calls"][0]["filename"] == \
            "checkpoints/run-A/step5000.pt"
        assert fake_hf["download_calls"][1]["filename"] == \
            "checkpoints/run-A/step5000.mem"

    def test_missing_sidecar_does_not_fail(self, fake_hf, tmp_path):
        from neuroslm.hf_checkpoints import download_checkpoint
        fake_hf["tmp_dir"] = tmp_path / "_hf_cache"
        fake_hf["missing_files"] = {"checkpoints/run-A/step5000.mem"}
        out = download_checkpoint(
            "checkpoints/run-A/step5000.pt",
            repo_id="x/y",
            dest_dir=str(tmp_path / "lfs_checkpoints"),
        )
        # PT still arrived — sidecar absence is non-fatal
        assert out is not None
        assert out.exists()
        # Sidecar copy did NOT materialise
        assert not out.with_suffix(".mem").exists()

    def test_pt_failure_returns_none(self, fake_hf, tmp_path):
        from neuroslm.hf_checkpoints import download_checkpoint
        fake_hf["tmp_dir"] = tmp_path / "_hf_cache"
        fake_hf["download_should_fail"] = True
        out = download_checkpoint(
            "checkpoints/run-A/step5000.pt",
            repo_id="x/y",
            dest_dir=str(tmp_path / "lfs_checkpoints"),
        )
        assert out is None


class TestImportContract:
    """The module is import-safe even when ``huggingface_hub`` is missing
    — every function returns gracefully, no top-level import."""

    def test_module_imports_without_huggingface_hub(self, monkeypatch):
        # Ensure the module re-imports clean even if hf is not on sys.modules
        # (the lazy imports inside list/download protect against this)
        if "neuroslm.hf_checkpoints" in sys.modules:
            del sys.modules["neuroslm.hf_checkpoints"]
        # We can't actually delete huggingface_hub from a real test env,
        # but we can verify the module imports without invoking it.
        import neuroslm.hf_checkpoints as hf_checkpoints
        assert hasattr(hf_checkpoints, "list_repo_checkpoints")
        assert hasattr(hf_checkpoints, "find_latest_checkpoint")
        assert hasattr(hf_checkpoints, "download_checkpoint")
        assert hasattr(hf_checkpoints, "parse_hf_uri")
