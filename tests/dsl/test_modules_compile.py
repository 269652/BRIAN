# -*- coding: utf-8 -*-
"""Phase 2 equivalence test: DSL `module {}` blocks compile to instantiated
nn.Module objects matching what Brain.__init__ produces directly.

Phase 1 proved that `cfg = compile_to_brain_config(.neuro) == PRESETS[X]()`.
Phase 2 proves that `compile_to_modules(.neuro, cfg)["language"]` builds
a `LanguageCortex` with the same structure as `brain.language`.

Subsequent Phase 2c work expands the DSL `module {}` declarations to cover
the remaining ~30 Brain submodules (thalamus, world_m, hippo, dmn, pfc, ...).
"""
from __future__ import annotations
import os
import sys
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from neuroslm.dsl.compiler import (
    NeuroMLCompiler, compile_to_brain_config, compile_to_modules,
)
from neuroslm.config import PRESETS, BrainConfig
from neuroslm.modules.language import LanguageCortex


HERE = os.path.dirname(os.path.abspath(__file__))
NEURO_FILE = os.path.join(
    HERE, '..', '..', 'neuroslm', 'dsl', 'rcc_bowtie.neuro')


def _scale_down(cfg: BrainConfig) -> BrainConfig:
    """Shrink cfg to CPU-friendly size for fast tests."""
    cfg.d_sem = 64
    cfg.d_hidden = 64
    cfg.lang_layers = 2
    cfg.lang_heads = 4
    cfg.lang_ctx = 32
    cfg.vocab_size = 64
    return cfg


def test_modules_block_parses():
    """The compiler must extract the `module language {}` block + populate
    program.modules with one ModuleIR carrying class='LanguageCortex'."""
    program = NeuroMLCompiler.compile_file(NEURO_FILE)
    assert program.modules is not None
    assert len(program.modules) >= 1, (
        f"expected >=1 module block in rcc_bowtie.neuro, got "
        f"{len(program.modules)}")
    names = [m.name for m in program.modules]
    assert "language" in names, f"missing `module language`, got {names}"
    lang_mod = next(m for m in program.modules if m.name == "language")
    assert lang_mod.class_name == "LanguageCortex", (
        f"expected class=LanguageCortex, got {lang_mod.class_name!r}")
    assert "vocab_size" in lang_mod.args
    assert lang_mod.args["vocab_size"] == "cfg.vocab_size"
    print(f"[1] module block parses: {len(program.modules)} modules "
          f"({names})  PASS")


def test_compile_to_modules_instantiates_language():
    """compile_to_modules must produce a LanguageCortex instance with
    the exact constructor args declared in the DSL."""
    program = NeuroMLCompiler.compile_file(NEURO_FILE)
    cfg = _scale_down(compile_to_brain_config(NEURO_FILE))

    modules = compile_to_modules(program, cfg)
    assert "language" in modules, f"`language` module not built. Got: {list(modules)}"
    lang = modules["language"]
    assert isinstance(lang, LanguageCortex), f"got {type(lang).__name__}"

    # Verify the constructed module has the right basic structure
    assert hasattr(lang, "tok_emb"), "LanguageCortex missing tok_emb"
    assert hasattr(lang, "blocks"), "LanguageCortex missing blocks"
    assert len(lang.blocks) == cfg.lang_layers, (
        f"expected {cfg.lang_layers} blocks, got {len(lang.blocks)}")

    # Embedding dim matches cfg
    assert lang.tok_emb.weight.shape == (cfg.vocab_size, cfg.d_hidden), (
        f"tok_emb shape {tuple(lang.tok_emb.weight.shape)} != "
        f"({cfg.vocab_size}, {cfg.d_hidden})")

    # PCT trunk + SGB enabled per the config
    assert lang.pct is not None, "PCT trunk should be built (cfg flag on)"

    print(f"[2] compile_to_modules -> LanguageCortex with "
          f"{sum(p.numel() for p in lang.parameters()):,} params  PASS")


def test_dsl_module_matches_python_construction():
    """Equivalence test: a LanguageCortex built from DSL has the same
    state_dict keys + shapes as one built directly via Python.

    Bit-exact weight equality requires matched RNG seed (done here);
    if this passes, the DSL `module {}` declaration is functionally
    indistinguishable from the Python constructor call.
    """
    program = NeuroMLCompiler.compile_file(NEURO_FILE)
    cfg = _scale_down(compile_to_brain_config(NEURO_FILE))

    # DSL path
    torch.manual_seed(42)
    dsl_modules = compile_to_modules(program, cfg)
    lang_dsl = dsl_modules["language"]

    # Python path — same constructor with the same args, same seed
    torch.manual_seed(42)
    lang_py = LanguageCortex(
        vocab_size=cfg.vocab_size,
        d_hidden=cfg.d_hidden,
        d_sem=cfg.d_sem,
        n_layers=cfg.lang_layers,
        n_heads=cfg.lang_heads,
        max_ctx=cfg.lang_ctx,
        n_kv_heads=cfg.lang_kv_heads,
        dropout=cfg.dropout,
        use_predictive_coding_trunk=cfg.use_predictive_coding_trunk,
        pct_mode=cfg.pct_mode,
        pct_lambda_fe=cfg.pct_lambda_fe,
        pct_hidden_mult=cfg.pct_hidden_mult,
        fe_gate_enable=cfg.fe_gate_enable,
        fe_gate_center=cfg.fe_gate_center,
        fe_gate_width=cfg.fe_gate_width,
    )

    sd_dsl = lang_dsl.state_dict()
    sd_py = lang_py.state_dict()

    only_dsl = set(sd_dsl) - set(sd_py)
    only_py = set(sd_py) - set(sd_dsl)
    assert not only_dsl and not only_py, (
        f"state_dict key mismatch:\n  only DSL: {sorted(only_dsl)[:5]}\n"
        f"  only Python: {sorted(only_py)[:5]}")

    for k in sd_dsl:
        if sd_dsl[k].shape != sd_py[k].shape:
            raise AssertionError(
                f"shape mismatch at {k!r}: dsl={tuple(sd_dsl[k].shape)} "
                f"py={tuple(sd_py[k].shape)}")

    # Bit-exact equality under same seed
    n_diff = sum(1 for k in sd_dsl if not torch.equal(sd_dsl[k], sd_py[k]))
    assert n_diff == 0, f"{n_diff} parameters differ between DSL and Python construction"

    print(f"[3] DSL-built LanguageCortex == Python-built "
          f"({len(sd_dsl)} state_dict entries, bit-exact)  PASS")


def test_unknown_class_rejected():
    """Module with unknown class must fail loudly."""
    from neuroslm.dsl.compiler import NeuroMLError
    bad = """
    neurotransmitter dopamine { base_concentration: 0.1 }
    population pfc { count: 100 }
    synapse pfc -> pfc { weight: 0.5 }
    module weird_thing {
        class: "ThisClassDoesNotExist"
        args: { }
    }
    """
    program = NeuroMLCompiler.compile(bad)
    cfg = BrainConfig()
    try:
        compile_to_modules(program, cfg)
    except NeuroMLError as e:
        assert "ThisClassDoesNotExist" in str(e), f"unexpected error: {e}"
        print("[4] unknown class rejected: PASS")
        return
    raise AssertionError("should have rejected unknown class")


if __name__ == "__main__":
    print("=" * 60)
    print("DSL Phase 2 — module compile tests")
    print("=" * 60)
    test_modules_block_parses()
    test_compile_to_modules_instantiates_language()
    test_dsl_module_matches_python_construction()
    test_unknown_class_rejected()
    print("=" * 60)
    print("ALL TESTS PASSED")
