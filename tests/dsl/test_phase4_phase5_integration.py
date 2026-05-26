# -*- coding: utf-8 -*-
"""Phase 4 + Phase 5 integration tests.

Phase 4: train.py's --neuro flag produces the same effective config as
the legacy --preset path.

Phase 5: the evolutionary engine's CircuitGenotype can be materialized
to a trainable Brain via materialize_genotype_to_brain(). Closes the
search → train loop.

Phase 3 (forward path expressible entirely in DSL) is deliberately NOT
covered here — the Brain class stays in Python as the interpreter. Full
DSL-expressible forward is documented in docs/DSL_REFACTOR.md as future
research work.
"""
from __future__ import annotations
import os
import sys
import dataclasses

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from neuroslm.dsl.compiler import compile_to_brain_config
from neuroslm.dsl.evolutionary import (
    materialize_genotype_to_brain, neuro_file_to_genotype,
)
from neuroslm.config import PRESETS, BrainConfig


HERE = os.path.dirname(os.path.abspath(__file__))
NEURO_FILE = os.path.join(
    HERE, '..', '..', 'neuroslm', 'dsl', 'rcc_bowtie.neuro')


def test_phase4_neuro_flag_path_matches_preset():
    """Phase 4: simulating what train.py does when --neuro is passed.
    The resulting BrainConfig must be identical to PRESETS['rcc_bowtie_30m_p2']()
    so existing training code paths work without modification."""
    # This mirrors the exact resolution in train.py:
    #   if args.neuro: cfg = compile_to_brain_config(args.neuro)
    #   else:          cfg = PRESETS[args.preset]()
    cfg_dsl = compile_to_brain_config(NEURO_FILE)
    cfg_py = PRESETS['rcc_bowtie_30m_p2']()

    mismatches = []
    for f in dataclasses.fields(BrainConfig):
        if getattr(cfg_dsl, f.name) != getattr(cfg_py, f.name):
            mismatches.append(f.name)
    assert not mismatches, (
        f"Phase 4: --neuro vs --preset mismatch on {len(mismatches)} fields:\n  "
        + "\n  ".join(mismatches[:10]))
    print(f"[1] Phase 4: --neuro <file> == --preset rcc_bowtie_30m_p2 "
          f"({len(dataclasses.fields(BrainConfig))} fields)  PASS")


def test_phase4_synthetic_preset_tag():
    """train.py uses os.path.splitext+basename to derive a preset tag for
    checkpoint naming when --neuro is given. Verify the tag is sensible."""
    tag = os.path.splitext(os.path.basename(NEURO_FILE))[0]
    assert tag == 'rcc_bowtie', f"expected 'rcc_bowtie', got {tag!r}"
    # Tag must be filename-safe (no slashes, dots, etc.)
    assert '/' not in tag and '\\' not in tag and '.' not in tag, (
        f"tag {tag!r} contains path-unsafe chars")
    print(f"[2] Phase 4: synthetic preset tag = {tag!r}  PASS")


def test_phase5_genotype_round_trip():
    """Phase 5: load a .neuro file as a seed genotype, then materialize
    it back to a trainable Brain. The round-trip closes the search loop."""
    seed = neuro_file_to_genotype(NEURO_FILE)
    assert seed.source.startswith('# -*-') or 'config {' in seed.source, (
        "genotype source should contain the file content")
    assert seed.generation == 0
    assert len(seed.source) > 1000, (
        f"genotype source seems truncated, only {len(seed.source)} chars")

    # Materialize -> Brain (scaled down for fast CPU test)
    scale = {
        'd_hidden': 64, 'd_sem': 64, 'lang_layers': 2, 'lang_heads': 4,
        'lang_ctx': 32, 'dmn_layers': 1, 'pfc_layers': 1, 'pfc_heads': 4,
        'world_layers': 1, 'forward_layers': 1, 'hippo_capacity': 64,
        'vocab_size': 64,
    }
    brain = materialize_genotype_to_brain(seed, scale_overrides=scale)
    n_params = sum(p.numel() for p in brain.parameters())
    assert n_params > 1_000_000, f"expected >1M params, got {n_params:,}"
    # Sanity: the trunk's PCT structure should be present (it's in the cfg)
    assert hasattr(brain, 'language')
    assert brain.language.pct is not None, "PCT trunk should be present"
    print(f"[3] Phase 5: genotype round-trip "
          f"(DSL -> IR -> Brain with {n_params:,} params)  PASS")


def test_phase5_compiles_to_runnable():
    """Phase 5: the materialized Brain must actually run a forward pass.
    A model that compiles but doesn't forward is useless for fitness."""
    import torch
    seed = neuro_file_to_genotype(NEURO_FILE)
    scale = {
        'd_hidden': 64, 'd_sem': 64, 'lang_layers': 2, 'lang_heads': 4,
        'lang_ctx': 32, 'dmn_layers': 1, 'pfc_layers': 1, 'pfc_heads': 4,
        'world_layers': 1, 'forward_layers': 1, 'hippo_capacity': 64,
        'vocab_size': 64,
    }
    brain = materialize_genotype_to_brain(seed, scale_overrides=scale)
    brain.eval()
    ids = torch.randint(0, 64, (1, 8))
    with torch.no_grad():
        out = brain.forward_lm(ids, targets=torch.randint(0, 64, (1, 8)))
    assert 'loss' in out
    assert 'logits' in out
    assert out['logits'].shape == (1, 8, 64), (
        f"expected logits (1,8,64), got {tuple(out['logits'].shape)}")
    loss = float(out['loss'].item())
    assert 0 < loss < 100, f"loss seems off: {loss}"
    print(f"[4] Phase 5: forward_lm runs end-to-end (loss={loss:.3f})  PASS")


if __name__ == "__main__":
    print("=" * 60)
    print("DSL Phase 4 + Phase 5 integration")
    print("=" * 60)
    test_phase4_neuro_flag_path_matches_preset()
    test_phase4_synthetic_preset_tag()
    test_phase5_genotype_round_trip()
    test_phase5_compiles_to_runnable()
    print("=" * 60)
    print("ALL TESTS PASSED")
