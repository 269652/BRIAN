"""Defensive contract: ``_load_lm_cached`` must handle legacy ``.bin``
checkpoints (e.g. ``gpt2``) on torch < 2.6.

Root-cause record
=================

On the 2026-06-14 vast.ai deploy (instance ``40920991``, A100 SXM4),
the run died with::

    File "/workspace/brian/neuroslm/experts.py", line 119, in _load_lm_cached
        lm = AutoModelForCausalLM.from_pretrained(model_id)
    ...
    ValueError: Due to a serious vulnerability issue in `torch.load`,
    even with `weights_only=True`, we now require users to upgrade
    torch to at least v2.6 in order to use the function. This version
    restriction does not apply when loading files with safetensors.
    See the vulnerability report here
    https://nvd.nist.gov/vuln/detail/CVE-2025-32434

The vast.ai pre-built image ships ``torch == 2.5.x``. The ``gpt2``
HuggingFace repo only ships ``pytorch_model.bin`` (no safetensors),
so the bare ``from_pretrained("gpt2")`` call refuses to load.

Fix policy
==========

We do not control the container's torch version, so the loader must
adapt:

  1. **Prefer safetensors** when available — passing
     ``use_safetensors=True`` makes ``from_pretrained`` skip the
     ``.bin`` path entirely (and the CVE check) when the repo has a
     ``model.safetensors`` file.
  2. **Fall back to legacy** ``.bin`` with ``weights_only=False`` for
     repos like ``gpt2`` that only ship the legacy format. This is
     safe in our use case because every ``model_id`` we load comes
     from a hard-coded ``arch.neuro`` config (no user-controlled
     paths), and ``weights_only=False`` is the only way to load these
     pre-safetensors checkpoints on torch < 2.6 without upgrading.

The fallback emits a one-time warning so the choice is visible in
training logs.
"""
from __future__ import annotations

from unittest.mock import patch, MagicMock

import pytest


transformers = pytest.importorskip("transformers")


@pytest.fixture(autouse=True)
def _clear_cache():
    """Each test starts with an empty ``_LM_CACHE`` so the loader's
    branch logic is exercised."""
    from neuroslm import experts

    experts._LM_CACHE.clear()
    yield
    experts._LM_CACHE.clear()


# ──────────────────────────────────────────────────────────────────────
# Contract 1 — prefer safetensors
# ──────────────────────────────────────────────────────────────────────


class TestPrefersSafetensors:
    """When safetensors are available the loader must pass
    ``use_safetensors=True`` so the legacy ``torch.load`` CVE check
    is bypassed entirely."""

    def test_first_call_passes_use_safetensors_true(self):
        from neuroslm import experts

        mock_lm = MagicMock()
        with patch(
            "transformers.AutoModelForCausalLM.from_pretrained",
            return_value=mock_lm,
        ) as m:
            experts._load_lm_cached("any/model-id")
        # The loader must have asked for safetensors on first attempt.
        # The exact kwarg passing convention (positional vs kw) doesn't
        # matter as long as ``use_safetensors`` ended up True OR we
        # explicitly passed safe-only behaviour.
        first_call = m.call_args_list[0]
        assert first_call.kwargs.get("use_safetensors") is True, (
            f"first from_pretrained call must request safetensors; "
            f"got kwargs={first_call.kwargs}"
        )


# ──────────────────────────────────────────────────────────────────────
# Contract 2 — fall back to weights_only=False for legacy .bin repos
# ──────────────────────────────────────────────────────────────────────


class TestFallbackForLegacyBin:
    """When the safetensors load fails because the repo only ships
    ``.bin`` (the case for ``gpt2`` itself), the loader retries with
    ``use_safetensors=False`` + ``weights_only=False`` so the legacy
    checkpoint loads on torch < 2.6 without forcing a system upgrade."""

    def test_fallback_when_safetensors_missing(self):
        from neuroslm import experts

        mock_lm = MagicMock()
        # The error message that surfaces from HF when the repo doesn't
        # ship safetensors AND torch is too old to load .bin safely is
        # the literal CVE-2025-32434 message; match the substring.
        cve_error = ValueError(
            "Due to a serious vulnerability issue in `torch.load`, "
            "even with `weights_only=True`, we now require users to "
            "upgrade torch to at least v2.6 ..."
        )

        call_count = {"n": 0}

        def _from_pretrained(*args, **kwargs):
            call_count["n"] += 1
            if call_count["n"] == 1:
                # First attempt (use_safetensors=True) fails with the
                # CVE error because the repo only has .bin.
                raise cve_error
            return mock_lm

        with patch(
            "transformers.AutoModelForCausalLM.from_pretrained",
            side_effect=_from_pretrained,
        ) as m:
            out = experts._load_lm_cached("gpt2")

        assert out is mock_lm
        # Loader must have retried.
        assert call_count["n"] == 2, (
            f"expected exactly one retry, got {call_count['n']} calls"
        )
        # Second attempt must explicitly opt-out of safetensors AND
        # pass weights_only=False so torch < 2.6 will load the .bin.
        second = m.call_args_list[1]
        assert second.kwargs.get("use_safetensors") is False, (
            f"retry must set use_safetensors=False; got {second.kwargs}"
        )
        assert second.kwargs.get("weights_only") is False, (
            f"retry must set weights_only=False to bypass the CVE "
            f"check on legacy .bin; got {second.kwargs}"
        )

    def test_non_cve_errors_propagate(self):
        """The fallback must NOT swallow unrelated errors (e.g. network
        failures, missing model ids). Only the documented CVE message
        triggers the retry."""
        from neuroslm import experts

        unrelated = RuntimeError("repo does not exist on the Hub")
        with patch(
            "transformers.AutoModelForCausalLM.from_pretrained",
            side_effect=unrelated,
        ):
            with pytest.raises(RuntimeError, match="repo does not exist"):
                experts._load_lm_cached("nonexistent/model")


# ──────────────────────────────────────────────────────────────────────
# Contract 3 — caching survives the safetensors/fallback dispatch
# ──────────────────────────────────────────────────────────────────────


class TestCacheStillWorksAcrossDispatch:
    """The cache contract from
    ``tests/training/test_lm_expert_caching.py`` must still hold —
    the safetensors-aware loader is not allowed to break per-call
    sharing of the frozen module."""

    def test_second_call_skips_from_pretrained_entirely(self):
        from neuroslm import experts

        mock_lm = MagicMock()
        with patch(
            "transformers.AutoModelForCausalLM.from_pretrained",
            return_value=mock_lm,
        ) as m:
            a = experts._load_lm_cached("any/model-id")
            b = experts._load_lm_cached("any/model-id")
        assert a is b
        # ``from_pretrained`` invoked once on the first call only —
        # the cache returned ``a`` on the second call.
        assert m.call_count == 1, (
            f"second call must be a cache hit (no from_pretrained); "
            f"got call_count={m.call_count}"
        )
