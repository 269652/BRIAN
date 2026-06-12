"""Feature-flag ablation contract — TDD spec for the four opt-in toggles.

Adds ablation flags so the model topology can be A/B tested cleanly:

    use_tdw              — swap GlobalWorkspace ↔ TopologicalDifferentialWorkspace
    use_diff_attn        — force every cortex block to DiffTransformerBlock
                           (default keeps today's [Std, Diff, MoD] cycle)
    use_tonnetz_prior    — add an adjacency-loss regulariser that penalises
                           token-cooccurrence graphs with spectral gap below
                           a configurable threshold (default 0.3)
    use_expert_ensemble  — wired-up reservation for the in-flight
                           neuroslm/experts.py module; currently a no-op so
                           the flag schema is stable

All four default to False so building a `BrainConfig()` produces the same
forward behaviour as the current master.  See CLAUDE.md §10 (architecture
changes are experiments) — flipping any flag should be treated as a
distinct experimental condition with its own short eval run.
"""
from __future__ import annotations

import pytest
import torch

from neuroslm.config import BrainConfig, tiny


# ─────────────────────────────────────────────────────────────────────────
# Phase 1: BrainConfig flag schema
# Default False for every new flag → backward compatibility guarantee.
# ─────────────────────────────────────────────────────────────────────────

def test_brainconfig_has_use_tdw_flag():
    cfg = BrainConfig()
    assert hasattr(cfg, "use_tdw")
    assert cfg.use_tdw is False


def test_brainconfig_has_use_diff_attn_flag():
    cfg = BrainConfig()
    assert hasattr(cfg, "use_diff_attn")
    assert cfg.use_diff_attn is False


def test_brainconfig_has_use_tonnetz_prior_flag():
    cfg = BrainConfig()
    assert hasattr(cfg, "use_tonnetz_prior")
    assert cfg.use_tonnetz_prior is False


def test_brainconfig_has_use_expert_ensemble_flag():
    cfg = BrainConfig()
    assert hasattr(cfg, "use_expert_ensemble")
    assert cfg.use_expert_ensemble is False


def test_tiny_preset_inherits_all_defaults_false():
    """The `tiny()` factory must not silently flip any of the new flags on."""
    cfg = tiny()
    assert cfg.use_tdw is False
    assert cfg.use_diff_attn is False
    assert cfg.use_tonnetz_prior is False
    assert cfg.use_expert_ensemble is False


# ─────────────────────────────────────────────────────────────────────────
# Phase 2: Workspace swapping (use_tdw)
# Default → GlobalWorkspace.  use_tdw=True → TopologicalDifferentialWorkspace.
# Shape contract preserved across both (same call site in brain.py:1366).
# ─────────────────────────────────────────────────────────────────────────

def _build_minimal_brain(**flags):
    cfg = tiny()
    cfg.vocab_size = 256
    for k, v in flags.items():
        setattr(cfg, k, v)
    torch.manual_seed(0)
    from neuroslm.brain import Brain
    return Brain(cfg)


def test_default_brain_uses_global_workspace():
    brain = _build_minimal_brain()
    from neuroslm.modules.workspace import GlobalWorkspace
    assert isinstance(brain.gws, GlobalWorkspace)


def test_use_tdw_swaps_in_topological_workspace():
    brain = _build_minimal_brain(use_tdw=True)
    from neuroslm.modules.workspace import TopologicalDifferentialWorkspace
    assert isinstance(brain.gws, TopologicalDifferentialWorkspace)


def test_tdw_workspace_preserves_output_shape():
    """Drop-in must produce (B, n_slots, d_sem) just like GlobalWorkspace."""
    brain = _build_minimal_brain(use_tdw=True)
    cfg = brain.cfg
    candidates = torch.randn(2, 6, cfg.d_sem)
    ne_temp = torch.ones(2)
    out = brain.gws(candidates, ne_temp=ne_temp)
    assert out.shape == (2, cfg.gws_slots, cfg.d_sem)


# ─────────────────────────────────────────────────────────────────────────
# Phase 3: Differential-attention propagation (use_diff_attn)
# DiffTransformerBlock already exists and is woven into the default
# interleaved cortex.  This flag is a higher-order ablation switch:
#     False (default) → today's [Std, Diff, MoD] cycle  (unchanged)
#     True            → every non-baseline cortex block is DiffTransformerBlock
# ─────────────────────────────────────────────────────────────────────────

def test_default_cortex_keeps_interleaved_pattern():
    """Backward compatibility: default flags ⇒ today's [Std, Diff, MoD] cycle."""
    brain = _build_minimal_brain()
    block_types = {type(b).__name__ for b in brain.language.blocks}
    # tiny has lang_layers=2 → indexes 0,1 → {TransformerBlock, DiffTransformerBlock}
    assert "TransformerBlock" in block_types
    assert "DiffTransformerBlock" in block_types


def test_use_diff_attn_makes_all_blocks_differential():
    """When True every cortex block uses DifferentialAttention."""
    from neuroslm.modules.differential_attention import DiffTransformerBlock
    brain = _build_minimal_brain(use_diff_attn=True)
    for blk in brain.language.blocks:
        assert isinstance(blk, DiffTransformerBlock), (
            f"with use_diff_attn=True every block must be DiffTransformerBlock, "
            f"got {type(blk).__name__}")


def test_use_diff_attn_forward_runs():
    """All-DiffAttn cortex must still produce finite logits."""
    brain = _build_minimal_brain(use_diff_attn=True)
    brain.eval()
    ids = torch.randint(0, 256, (1, 8))
    with torch.no_grad():
        out = brain.forward_lm(ids)
    assert "logits" in out
    assert torch.isfinite(out["logits"]).all()


# ─────────────────────────────────────────────────────────────────────────
# Phase 4: Tonnetz adjacency-loss prior (use_tonnetz_prior)
# Penalises low spectral gap on the token-cooccurrence Laplacian.
# Soft hinge:  penalty = relu(threshold − λ_1).  Default off → no module,
# no extra loss term, no extra parameters.
# ─────────────────────────────────────────────────────────────────────────

def test_default_has_no_tonnetz_prior():
    brain = _build_minimal_brain()
    assert getattr(brain, "tonnetz_prior", None) is None


def test_use_tonnetz_prior_instantiates_module():
    brain = _build_minimal_brain(use_tonnetz_prior=True)
    assert brain.tonnetz_prior is not None


def test_tonnetz_prior_returns_scalar_tensor():
    """0-D tensor so it composes via `total_loss + penalty`."""
    from neuroslm.modules.tonnetz_prior import TonnetzPrior
    prior = TonnetzPrior(vocab_size=256, d_embed=32, gap_threshold=0.3)
    embeddings = torch.randn(256, 32, requires_grad=True)
    sample_ids = torch.randint(0, 256, (4, 16))
    penalty = prior(embeddings, sample_ids)
    assert penalty.ndim == 0
    assert torch.isfinite(penalty)


def test_tonnetz_prior_penalty_nonneg():
    """Adjacency loss is non-negative by construction (relu hinge)."""
    from neuroslm.modules.tonnetz_prior import TonnetzPrior
    prior = TonnetzPrior(vocab_size=256, d_embed=32, gap_threshold=0.3)
    embeddings = torch.randn(256, 32)
    sample_ids = torch.randint(0, 256, (4, 16))
    penalty = prior(embeddings, sample_ids)
    assert penalty.item() >= 0.0


def test_tonnetz_prior_gradient_flows():
    """Backward through the prior must populate the embedding's .grad.

    For random Gaussian embeddings with ~50 unique tokens the cooccurrence
    graph is fairly well-connected (λ_1 ≈ 1.5 empirically). To force the
    hinge to bite we set ``gap_threshold = 2.0`` — well above the typical
    random-baseline λ_1 — so the penalty is guaranteed > 0 and the
    backward pass populates a non-trivial gradient.
    """
    from neuroslm.modules.tonnetz_prior import TonnetzPrior
    prior = TonnetzPrior(vocab_size=256, d_embed=32, gap_threshold=2.0)
    embeddings = torch.randn(256, 32, requires_grad=True)
    sample_ids = torch.randint(0, 256, (4, 16))
    penalty = prior(embeddings, sample_ids)
    assert penalty.item() > 0.0, "threshold above typical λ_1 must produce a positive penalty"
    penalty.backward()
    assert embeddings.grad is not None
    assert embeddings.grad.abs().sum() > 0


def test_tonnetz_prior_threshold_monotone():
    """Higher threshold ⇒ stricter constraint ⇒ ≥ penalty."""
    from neuroslm.modules.tonnetz_prior import TonnetzPrior
    torch.manual_seed(0)
    embeddings = torch.randn(256, 32)
    sample_ids = torch.randint(0, 256, (4, 16))
    p_low  = TonnetzPrior(256, 32, gap_threshold=0.0)(embeddings, sample_ids)
    p_high = TonnetzPrior(256, 32, gap_threshold=0.9)(embeddings, sample_ids)
    assert p_high.item() >= p_low.item() - 1e-6


# ─────────────────────────────────────────────────────────────────────────
# Phase 5: use_expert_ensemble flag (wired no-op placeholder)
# experts.py is in-flight on disk (unstaged).  Reserve the flag now so the
# schema is stable; the consumer path is a no-op until that module lands.
# ─────────────────────────────────────────────────────────────────────────

def test_use_expert_ensemble_flag_constructs():
    """Flipping the flag must not break Brain construction."""
    brain = _build_minimal_brain(use_expert_ensemble=True)
    assert brain is not None


def test_use_expert_ensemble_flag_observable():
    """Flag must remain readable on cfg for ablation report bookkeeping."""
    brain = _build_minimal_brain(use_expert_ensemble=True)
    assert getattr(brain.cfg, "use_expert_ensemble", False) is True


# ─────────────────────────────────────────────────────────────────────────
# Phase 6: Ablation matrix smoke test — all flags on simultaneously
# ─────────────────────────────────────────────────────────────────────────

def test_all_flags_on_constructs_and_forwards():
    brain = _build_minimal_brain(
        use_tdw=True,
        use_diff_attn=True,
        use_tonnetz_prior=True,
        use_expert_ensemble=True,
    )
    brain.eval()
    ids = torch.randint(0, 256, (1, 8))
    with torch.no_grad():
        out = brain.forward_lm(ids)
    assert "logits" in out
    assert torch.isfinite(out["logits"]).all()
