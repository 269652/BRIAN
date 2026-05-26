# -*- coding: utf-8 -*-
"""Phase 1 EXTENDED: full Brain-instance equivalence between

    A) `Brain(cfg)` where cfg = `PRESETS['rcc_bowtie_30m_p2']()`         (Python)
    B) `Brain(cfg)` where cfg = `compile_to_brain_config('rcc_bowtie.neuro')` (DSL)

If these are equivalent under matched random seeds, the DSL config block
is operationally indistinguishable from the Python preset — the first
piece of "DSL as source of truth" is true.

Checks (each as a separate test):
  1. Module structure: same set of named submodules (Brain.* attribute set)
  2. Parameter shapes: every named_parameter has the same shape on both
  3. State dict: same keys, same shapes
  4. Forward output equality: with matched seed, identical logits + loss
  5. Backward output equality: same gradients on every parameter
  6. Step-1 weight update equality: after one optimizer step, weights match

These are STRONG equivalence claims (not just config-equality). They run on
CPU with a tiny config-override to keep the test fast (~10-20 s).

PHASE-2+ extensions:
  - The same test pattern applies to the codegen'd Brain class (whenever
    Phase 2 lands). Re-target by swapping `Brain` for the generated class.
"""
from __future__ import annotations
import os
import sys
import dataclasses
import math

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from neuroslm.config import PRESETS, BrainConfig
from neuroslm.brain import Brain
from neuroslm.dsl.compiler import compile_to_brain_config


HERE = os.path.dirname(os.path.abspath(__file__))
NEURO_FILE = os.path.join(
    HERE, '..', '..', 'neuroslm', 'dsl', 'rcc_bowtie.neuro')


def _scale_down(cfg: BrainConfig) -> BrainConfig:
    """Shrink any cfg to a CPU-friendly size for fast equivalence tests.

    Halves d_hidden, narrows context, drops layer counts and bio capacity.
    Applied IDENTICALLY to both the DSL-built cfg and the Python-built cfg
    so the comparison stays meaningful — they just both run smaller.
    """
    cfg.d_sem = 64
    cfg.d_hidden = 64
    cfg.lang_layers = 2
    cfg.lang_heads = 4
    cfg.lang_ctx = 32
    cfg.dmn_layers = 1
    cfg.pfc_layers = 1
    cfg.pfc_heads = 4
    cfg.world_layers = 1
    cfg.forward_layers = 1
    cfg.hippo_capacity = 128
    cfg.vocab_size = 64
    cfg.gradient_checkpointing = False
    return cfg


def _build_pair():
    """Construct two Brain instances under matched seeds — one from Python
    preset, one from DSL. Both shrunk to a tiny CPU-friendly size.

    The IMPORTANT invariant: torch.manual_seed must be called immediately
    BEFORE each Brain() construction so they see the identical RNG stream.
    """
    cfg_py = _scale_down(PRESETS['rcc_bowtie_30m_p2']())
    cfg_dsl = _scale_down(compile_to_brain_config(NEURO_FILE))

    # Sanity: configs must already be equal field-wise (this is the Phase 1
    # config-compile test in summary form — full check is in
    # test_config_compile.py)
    for f in dataclasses.fields(BrainConfig):
        assert getattr(cfg_py, f.name) == getattr(cfg_dsl, f.name), (
            f"cfg mismatch on {f.name}: py={getattr(cfg_py, f.name)!r} "
            f"dsl={getattr(cfg_dsl, f.name)!r}")

    torch.manual_seed(12345)
    brain_py = Brain(cfg_py)

    torch.manual_seed(12345)
    brain_dsl = Brain(cfg_dsl)

    return brain_py, brain_dsl, cfg_py, cfg_dsl


# ──────────────────────────────────────────────────────────────────────
# Test 1 — module structure
# ──────────────────────────────────────────────────────────────────────

def test_named_modules_match():
    """The set of named submodules must be exactly identical."""
    brain_py, brain_dsl, _, _ = _build_pair()
    modules_py = sorted(name for name, _ in brain_py.named_modules())
    modules_dsl = sorted(name for name, _ in brain_dsl.named_modules())
    only_py = set(modules_py) - set(modules_dsl)
    only_dsl = set(modules_dsl) - set(modules_py)
    if only_py or only_dsl:
        raise AssertionError(
            f"named_modules differ:\n"
            f"  only in Python:  {sorted(only_py)[:10]}\n"
            f"  only in DSL:     {sorted(only_dsl)[:10]}")
    print(f"[1] named_modules match ({len(modules_py)} modules)  PASS")


# ──────────────────────────────────────────────────────────────────────
# Test 2 — parameter shapes match
# ──────────────────────────────────────────────────────────────────────

def test_param_shapes_match():
    """Every named parameter must exist on both with the same shape."""
    brain_py, brain_dsl, _, _ = _build_pair()
    shapes_py = {n: tuple(p.shape) for n, p in brain_py.named_parameters()}
    shapes_dsl = {n: tuple(p.shape) for n, p in brain_dsl.named_parameters()}

    only_py = set(shapes_py) - set(shapes_dsl)
    only_dsl = set(shapes_dsl) - set(shapes_py)
    diff_shape = {
        n: (shapes_py[n], shapes_dsl[n])
        for n in (set(shapes_py) & set(shapes_dsl))
        if shapes_py[n] != shapes_dsl[n]
    }
    if only_py or only_dsl or diff_shape:
        raise AssertionError(
            f"param shapes differ:\n"
            f"  only in Python: {sorted(only_py)[:10]}\n"
            f"  only in DSL:    {sorted(only_dsl)[:10]}\n"
            f"  diff shape:     {dict(list(diff_shape.items())[:5])}")
    print(f"[2] param shapes match ({len(shapes_py)} params)  PASS")


# ──────────────────────────────────────────────────────────────────────
# Test 3 — state_dict keys + shapes
# ──────────────────────────────────────────────────────────────────────

def test_state_dict_keys_match():
    """state_dict keys + tensor shapes must be identical. This is the
    invariant that determines whether a checkpoint trained under one path
    can be loaded into the other."""
    brain_py, brain_dsl, _, _ = _build_pair()
    sd_py = brain_py.state_dict()
    sd_dsl = brain_dsl.state_dict()

    only_py = set(sd_py) - set(sd_dsl)
    only_dsl = set(sd_dsl) - set(sd_py)
    if only_py or only_dsl:
        raise AssertionError(
            f"state_dict keys differ:\n"
            f"  only in Python: {sorted(only_py)[:10]}\n"
            f"  only in DSL:    {sorted(only_dsl)[:10]}")

    for k in sd_py:
        if sd_py[k].shape != sd_dsl[k].shape:
            raise AssertionError(
                f"state_dict[{k!r}] shape differs: "
                f"py={tuple(sd_py[k].shape)} dsl={tuple(sd_dsl[k].shape)}")

    print(f"[3] state_dict keys + shapes match ({len(sd_py)} entries)  PASS")


# ──────────────────────────────────────────────────────────────────────
# Test 4 — total parameter count
# ──────────────────────────────────────────────────────────────────────

def test_param_count_matches():
    brain_py, brain_dsl, _, _ = _build_pair()
    n_py = sum(p.numel() for p in brain_py.parameters())
    n_dsl = sum(p.numel() for p in brain_dsl.parameters())
    assert n_py == n_dsl, f"param count differs: py={n_py}, dsl={n_dsl}"
    print(f"[4] total param count: {n_py:,} (matched)  PASS")


# ──────────────────────────────────────────────────────────────────────
# Test 5 — forward output equality (under matched seed)
# ──────────────────────────────────────────────────────────────────────

def test_forward_output_equality():
    """Same seed → same Brain init → same input → identical logits + loss.

    Bit-exact equality is too strong (floating-point summation order, etc.)
    so we check `torch.allclose` with conservative tolerances. If this
    fails, something in the construction path is non-deterministic across
    the two Brain instances even though their state_dicts match.
    """
    brain_py, brain_dsl, _, _ = _build_pair()
    brain_py.eval()
    brain_dsl.eval()

    # Lock RNG for any dropout-like inference-time path
    torch.manual_seed(7)
    ids = torch.randint(0, 64, (1, 16))
    targets = torch.randint(0, 64, (1, 16))

    with torch.no_grad():
        out_py = brain_py.forward_lm(ids, targets=targets)
        out_dsl = brain_dsl.forward_lm(ids, targets=targets)

    # logits must match
    assert torch.allclose(out_py['logits'], out_dsl['logits'], atol=1e-5, rtol=1e-4), (
        f"logits differ: max abs diff = "
        f"{(out_py['logits'] - out_dsl['logits']).abs().max().item():.6f}")

    # loss must match
    loss_py = float(out_py['loss'].item())
    loss_dsl = float(out_dsl['loss'].item())
    assert math.isclose(loss_py, loss_dsl, rel_tol=1e-4, abs_tol=1e-5), (
        f"loss differs: py={loss_py:.6f} dsl={loss_dsl:.6f}")

    print(f"[5] forward output equality: loss_py={loss_py:.6f} "
          f"loss_dsl={loss_dsl:.6f}  PASS")


# ──────────────────────────────────────────────────────────────────────
# Test 6 — backward gradients match
# ──────────────────────────────────────────────────────────────────────

def test_backward_gradients_match():
    """After backward() on the same loss, every param's .grad must match.

    Note: train-mode dropout consumes global RNG state, so we re-seed
    IMMEDIATELY before each forward pass to ensure both brains see the
    same dropout masks. Without this, sequential forward calls would
    produce divergent gradients even with identical model state — not
    because the models differ, but because dropout RNG drifted.
    """
    brain_py, brain_dsl, _, _ = _build_pair()
    brain_py.train()
    brain_dsl.train()

    torch.manual_seed(11)
    ids = torch.randint(0, 64, (1, 16))
    targets = torch.randint(0, 64, (1, 16))

    torch.manual_seed(99)
    out_py = brain_py.forward_lm(ids, targets=targets)
    torch.manual_seed(99)
    out_dsl = brain_dsl.forward_lm(ids, targets=targets)
    out_py['loss'].backward()
    out_dsl['loss'].backward()

    grads_py = {n: p.grad for n, p in brain_py.named_parameters() if p.grad is not None}
    grads_dsl = {n: p.grad for n, p in brain_dsl.named_parameters() if p.grad is not None}

    only_py = set(grads_py) - set(grads_dsl)
    only_dsl = set(grads_dsl) - set(grads_py)
    if only_py or only_dsl:
        raise AssertionError(
            f"params with gradients differ between runs:\n"
            f"  only py has grad: {sorted(only_py)[:5]}\n"
            f"  only dsl has grad: {sorted(only_dsl)[:5]}")

    mismatches = []
    for k in grads_py:
        gp, gd = grads_py[k], grads_dsl[k]
        if gp.shape != gd.shape:
            mismatches.append((k, 'shape', tuple(gp.shape), tuple(gd.shape)))
            continue
        if not torch.allclose(gp, gd, atol=1e-5, rtol=1e-4):
            diff = (gp - gd).abs().max().item()
            if diff > 1e-4:
                mismatches.append((k, 'value', diff))
    if mismatches:
        raise AssertionError(
            f"gradient mismatches ({len(mismatches)} params):\n  "
            + "\n  ".join(str(m) for m in mismatches[:8]))
    print(f"[6] gradients match ({len(grads_py)} params w/ non-None grad)  PASS")


# ──────────────────────────────────────────────────────────────────────
# Test 7 — step-1 weight update equality (full SGD round trip)
# ──────────────────────────────────────────────────────────────────────

def test_step_1_weight_update_equality():
    """End-to-end: same seed → same init → same forward → same backward →
    same optimizer.step() → same weights. The strongest equivalence claim:
    if Brain-from-DSL and Brain-from-Python produce identical models after
    one full training step, they're operationally indistinguishable for
    arbitrary training trajectories (up to floating-point determinism).
    """
    brain_py, brain_dsl, _, _ = _build_pair()
    brain_py.train()
    brain_dsl.train()

    opt_py = torch.optim.AdamW(brain_py.parameters(), lr=1e-3)
    opt_dsl = torch.optim.AdamW(brain_dsl.parameters(), lr=1e-3)

    torch.manual_seed(13)
    ids = torch.randint(0, 64, (1, 16))
    targets = torch.randint(0, 64, (1, 16))

    # Re-seed before each forward (train mode dropout consumes RNG state)
    torch.manual_seed(101)
    out_py = brain_py.forward_lm(ids, targets=targets)
    torch.manual_seed(101)
    out_dsl = brain_dsl.forward_lm(ids, targets=targets)
    out_py['loss'].backward()
    out_dsl['loss'].backward()
    opt_py.step()
    opt_dsl.step()

    diffs = []
    for (n_py, p_py), (n_dsl, p_dsl) in zip(brain_py.named_parameters(),
                                            brain_dsl.named_parameters()):
        assert n_py == n_dsl, f"param-order mismatch: {n_py} vs {n_dsl}"
        d = (p_py.detach() - p_dsl.detach()).abs().max().item()
        if d > 1e-5:
            diffs.append((n_py, d))
    if diffs:
        raise AssertionError(
            f"post-step weight mismatches ({len(diffs)} params):\n  "
            + "\n  ".join(f"{n}: max|Δ|={d:.2e}" for n, d in diffs[:8]))
    print(f"[7] step-1 weights match (full round trip)  PASS")


if __name__ == "__main__":
    print("=" * 60)
    print("Brain instance equivalence: DSL vs Python preset")
    print("=" * 60)
    test_named_modules_match()
    test_param_shapes_match()
    test_state_dict_keys_match()
    test_param_count_matches()
    test_forward_output_equality()
    test_backward_gradients_match()
    test_step_1_weight_update_equality()
    print("=" * 60)
    print("ALL EQUIVALENCE TESTS PASSED")
