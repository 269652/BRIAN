"""Contracts for ``BRIANHarness.topology_summary`` parameter breakdown.

After the 2026-06-15 LFS-bloat fix, the topology summary print became
the user-facing answer to "why does my checkpoint say 1.1 B parameters
but the LFS limit is 2 GiB?". The answer is that ~1.1 B of those
parameters are frozen HuggingFace experts loaded into
``multi_cortex.experts.*`` which are NEVER persisted to the checkpoint.

This test file pins the print format so the breakdown stays visible:

  * The "total" number is honest about the full forward graph.
  * The "trainable" number is what the optimiser actually updates.
  * The "checkpoint" number is what hits disk (≤ trainable, because
    even some trainable heads can be excluded via
    ``_CKPT_EXTERNAL_PREFIXES``).
  * The MB estimate makes "is this going to blow LFS?" answerable
    at a glance.

Without this contract anyone could revert the print to a single
``sum(p.numel())`` and the user would have to ask the same question
again on the next training run.
"""
from __future__ import annotations

from pathlib import Path

import pytest
import torch.nn as nn

from neuroslm.dsl.codegen import CodeGenerator
from neuroslm.dsl.multifile import compile_folder
from neuroslm.dsl.training_config import TrainingConfig
from neuroslm.harness import BRIANHarness


_ARCH_ROOT = Path(__file__).resolve().parent.parent / "architectures" / "master"


# ──────────────────────────────────────────────────────────────────────
# Synthetic ensemble that mimics multi_cortex.experts.* layout
# (small enough for CI, no HF download).
# ──────────────────────────────────────────────────────────────────────


class _BigFrozenExpert(nn.Module):
    """Stand-in for an HF expert: much bigger than the rest of the
    harness and explicitly ``requires_grad=False``. Attribute name MUST
    be ``lm`` to match the real ``LMExpert`` layout."""
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
    """Mimics LMExpertEnsemble: ``experts`` (frozen, big) +
    ``router`` (trainable, small). Attribute names MUST be ``experts``
    and ``router`` so the ``multi_cortex.experts.`` prefix matches."""
    def __init__(self, n_experts: int = 3, hidden: int = 256, vocab: int = 1024):
        super().__init__()
        self.experts = nn.ModuleList(
            [_BigFrozenExpert(hidden=hidden, vocab=vocab) for _ in range(n_experts)]
        )
        self.router = nn.Linear(hidden, n_experts)


# ──────────────────────────────────────────────────────────────────────
# Fixtures (mirror the proven pattern from
# tests/test_checkpoint_excludes_frozen_experts.py)
# ──────────────────────────────────────────────────────────────────────


@pytest.fixture(scope="module")
def _compiled_circuit_factory():
    """Module-scoped circuit IR (compile is slow); per-test circuits
    are built from it so each test gets a fresh submodule tree."""
    ir = compile_folder(_ARCH_ROOT)
    Cls = CodeGenerator(ir, module_name="TopoSumTestCircuit").compile_to_module()
    return lambda: Cls(d_sem=64)


def _fresh_harness(circuit_factory) -> BRIANHarness:
    """Build a harness + splice in the fake multi_cortex post-hoc so
    the ``multi_cortex.experts.`` filter has something to chew on.

    The fake experts are sized so that frozen >> trainable — this
    mirrors the production setup where the trunk is ~30 M and the
    three HF experts are ~1.1 B together. We use 512×512 + 512×4096
    per expert × 3 experts ≈ 7 M frozen, dominating the ~1.3 M
    trainable circuit so the trainable-fraction assertion is robust.
    """
    cfg = TrainingConfig()
    cfg.genetics.enabled = False
    cfg.grad_accum_steps = 1
    h = BRIANHarness(
        circuit=circuit_factory(),
        vocab_size=512,
        d_sem=64,
        training_config=cfg,
    )
    # hidden=512, vocab=4096, 3 experts → ~3 × (3*512*512 + 512*4096) ≈ 8 M frozen
    h.multi_cortex = _FakeMultiCortex(n_experts=3, hidden=512, vocab=4096)
    return h


# ──────────────────────────────────────────────────────────────────────
# topology_summary() contract
# ──────────────────────────────────────────────────────────────────────


class TestTopologySummaryParamBreakdown:
    def test_summary_prints_trainable_and_frozen_separately(self, _compiled_circuit_factory):
        """The summary line must split ``parameters = N`` into total +
        trainable + frozen so the reader sees the LFS-relevant number."""
        h = _fresh_harness(_compiled_circuit_factory)
        text = h.topology_summary()
        assert "trainable" in text, (
            f"topology_summary must expose the trainable param count; "
            f"got:\n{text}"
        )
        assert "frozen" in text, (
            f"topology_summary must expose the frozen param count; "
            f"got:\n{text}"
        )

    def test_summary_reports_checkpoint_param_count_and_mb(self, _compiled_circuit_factory):
        """The summary must report what the *checkpoint* actually
        contains — both as a param count and as an estimated MB
        (fp32) so 'will this fit LFS?' is answerable in one glance."""
        h = _fresh_harness(_compiled_circuit_factory)
        text = h.topology_summary()
        assert "checkpoint" in text.lower(), (
            f"topology_summary must report the checkpoint param count; "
            f"got:\n{text}"
        )
        assert "MB" in text or "GB" in text, (
            f"topology_summary must include an estimated checkpoint "
            f"size in MB or GB; got:\n{text}"
        )

    def test_trainable_count_excludes_frozen_experts(self, _compiled_circuit_factory):
        """The 'trainable' number must NOT include the big frozen
        expert weights (they have ``requires_grad=False``)."""
        h = _fresh_harness(_compiled_circuit_factory)
        n_trainable = sum(p.numel() for p in h.parameters() if p.requires_grad)
        n_total     = sum(p.numel() for p in h.parameters())
        # The fake experts are big enough that trainable < 50 % of total.
        assert n_trainable < n_total // 2, (
            f"expected trainable << total in this fixture, got "
            f"trainable={n_trainable} total={n_total}"
        )
        # And the print must agree.
        text = h.topology_summary()
        # Extract the trainable number from "trainable X,XXX,XXX"
        import re
        m = re.search(r"trainable\s+([\d,]+)", text)
        assert m is not None, f"no 'trainable N' in summary:\n{text}"
        printed = int(m.group(1).replace(",", ""))
        assert printed == n_trainable, (
            f"printed trainable={printed:,} but actual trainable count "
            f"is {n_trainable:,}; summary diverged from reality:\n{text}"
        )

    def test_checkpoint_count_matches_persistable_state_dict(self, _compiled_circuit_factory):
        """The ``checkpoint = N`` figure in the summary must equal
        ``sum(t.numel() for t in _persistable_state_dict().values())``
        — i.e. exactly what ``save_checkpoint`` will write."""
        h = _fresh_harness(_compiled_circuit_factory)
        actual = sum(t.numel() for t in h._persistable_state_dict().values())
        text = h.topology_summary()
        import re
        m = re.search(r"checkpoint\s*=\s*([\d,]+)\s*params", text)
        assert m is not None, (
            f"summary must report 'checkpoint = N params'; got:\n{text}"
        )
        printed = int(m.group(1).replace(",", ""))
        assert printed == actual, (
            f"summary's checkpoint={printed:,} disagrees with actual "
            f"_persistable_state_dict count {actual:,}; the fix is "
            f"useless if the print lies:\n{text}"
        )

    def test_checkpoint_count_strictly_less_than_total(self, _compiled_circuit_factory):
        """In any harness that has frozen external weights (the
        production setup), the checkpoint count must be strictly less
        than the total parameter count — otherwise the LFS fix didn't
        actually exclude anything."""
        h = _fresh_harness(_compiled_circuit_factory)
        n_total = sum(p.numel() for p in h.parameters())
        n_saved = sum(t.numel() for t in h._persistable_state_dict().values())
        assert n_saved < n_total, (
            f"expected saved ({n_saved:,}) < total ({n_total:,}) when "
            f"frozen experts are attached; LFS fix may have regressed"
        )

    def test_summary_does_not_count_frozen_experts_twice(self, _compiled_circuit_factory):
        """Defensive: total = trainable + frozen, with no double-counting."""
        h = _fresh_harness(_compiled_circuit_factory)
        text = h.topology_summary()
        import re
        m_total = re.search(r"parameters\s*=\s*([\d,]+)", text)
        m_trn   = re.search(r"trainable\s+([\d,]+)", text)
        m_frz   = re.search(r"frozen\s+([\d,]+)", text)
        assert m_total and m_trn and m_frz, (
            f"summary missing one of total/trainable/frozen:\n{text}"
        )
        total = int(m_total.group(1).replace(",", ""))
        trn   = int(m_trn.group(1).replace(",", ""))
        frz   = int(m_frz.group(1).replace(",", ""))
        assert total == trn + frz, (
            f"total ({total:,}) != trainable ({trn:,}) + frozen "
            f"({frz:,}); summary numbers are inconsistent:\n{text}"
        )
