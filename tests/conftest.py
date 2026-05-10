"""Shared pytest fixtures for the NeuroSLM test suite.

Builds tiny CPU-only configs so each test runs in <1s. Heavier integration
tests (e.g. full Brain forward) live in their own files and use a session
fixture so the Brain is only instantiated once.
"""
from __future__ import annotations
import os
import sys
import pytest
import torch

# Ensure the project root is importable without requiring `pip install -e .`
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)


@pytest.fixture(autouse=True)
def _seed_torch():
    """Deterministic seed per test."""
    torch.manual_seed(0)


@pytest.fixture()
def device() -> torch.device:
    return torch.device("cpu")


@pytest.fixture()
def tiny_cfg():
    from neuroslm.config import tiny
    cfg = tiny()
    cfg.vocab_size = 256
    return cfg


@pytest.fixture(scope="session")
def tiny_brain():
    """Session-scoped Brain instance (expensive to build).

    Tests that mutate state should make a local copy or reset between calls;
    most tests only read forward-pass outputs, so sharing is safe.
    """
    from neuroslm.config import tiny
    from neuroslm.brain import Brain
    cfg = tiny()
    cfg.vocab_size = 256
    torch.manual_seed(0)
    brain = Brain(cfg)
    brain.eval()
    return brain


@pytest.fixture()
def random_ids():
    """Standard (1, 16) input batch for forward-pass tests."""
    return torch.randint(0, 256, (1, 16))


@pytest.fixture()
def random_pair():
    ids = torch.randint(0, 256, (1, 16))
    tgt = torch.randint(0, 256, (1, 16))
    return ids, tgt
