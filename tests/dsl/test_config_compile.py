# -*- coding: utf-8 -*-
"""Phase 1 equivalence test: DSL `config { ... }` block compiles to a
BrainConfig that matches the working `rcc_bowtie_30m_p2` Python preset.

When this passes, the DSL file is officially the source of truth for
config — editing `neuroslm/dsl/rcc_bowtie.neuro:config { ... }` is
equivalent to editing the `rcc_bowtie_30m_p2()` function in config.py.

Phases 2-5 (Brain class, forward path, train.py switch, evolutionary loop)
are tracked in `docs/DSL_REFACTOR.md`.
"""
from __future__ import annotations
import os
import sys
import dataclasses

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from neuroslm.dsl.compiler import (
    NeuroMLCompiler, compile_to_brain_config, to_brain_config, ConfigIR,
)
from neuroslm.config import PRESETS, BrainConfig


HERE = os.path.dirname(os.path.abspath(__file__))
NEURO_FILE = os.path.join(
    HERE, '..', '..', 'neuroslm', 'dsl', 'rcc_bowtie.neuro')


def test_config_block_parses():
    """The compiler must extract the config block and populate ConfigIR.fields."""
    program = NeuroMLCompiler.compile_file(NEURO_FILE)
    assert program.config is not None, "ProgramIR.config should not be None"
    assert isinstance(program.config, ConfigIR)
    assert len(program.config.fields) >= 10, (
        f"expected >=10 config fields in rcc_bowtie.neuro, got "
        f"{len(program.config.fields)}: {list(program.config.fields)}")
    # Sanity-check that a known field is present
    assert 'use_rcc_bowtie' in program.config.fields
    assert 'rcc_freeze_nt_modulation' in program.config.fields
    print(f"[1] config block parses: {len(program.config.fields)} fields  PASS")


def test_bool_coercion():
    """`true` / `false` strings → Python bool."""
    cfg = compile_to_brain_config(NEURO_FILE)
    assert cfg.use_rcc_bowtie is True
    assert cfg.rcc_freeze_nt_modulation is True
    assert cfg.use_predictive_coding_trunk is True
    assert cfg.use_smooth_gated_bus is True
    assert cfg.fe_gate_enable is True
    print("[2] bool coercion: PASS")


def test_numeric_coercion():
    """int / float coercion preserves exact values."""
    cfg = compile_to_brain_config(NEURO_FILE)
    assert cfg.d_sem == 192, f"d_sem expected 192, got {cfg.d_sem}"
    assert cfg.lang_ctx == 1024
    assert cfg.world_layers == 1
    assert cfg.forward_layers == 1
    assert cfg.hippo_capacity == 2048
    assert cfg.warmup_steps == 300
    assert cfg.weight_decay == 0.05
    assert cfg.pct_lambda_fe == 2.0
    assert cfg.baseline_lang_layers == 12
    print("[3] numeric coercion: PASS")


def test_equivalence_with_python_preset():
    """The headline test: DSL-compiled config == Python preset config.

    Iterates every BrainConfig field, compares the DSL-compiled value to
    the Python-preset value, fails loudly on any mismatch.
    """
    cfg_from_dsl = compile_to_brain_config(NEURO_FILE)
    cfg_from_python = PRESETS['rcc_bowtie_30m_p2']()

    mismatches = []
    for f in dataclasses.fields(BrainConfig):
        v_dsl = getattr(cfg_from_dsl, f.name)
        v_py = getattr(cfg_from_python, f.name)
        if v_dsl != v_py:
            mismatches.append((f.name, v_dsl, v_py))

    if mismatches:
        lines = [f"{n}: DSL={d!r}, Python={p!r}" for (n, d, p) in mismatches]
        raise AssertionError(
            f"DSL <-> Python preset mismatch ({len(mismatches)} field(s)):\n  "
            + "\n  ".join(lines))
    print(f"[4] equivalence: cfg_from_dsl == PRESETS['rcc_bowtie_30m_p2']()  PASS  "
          f"({len(dataclasses.fields(BrainConfig))} fields checked)")


_MINIMAL_PROGRAM = """
neurotransmitter dopamine { base_concentration: 0.1 }
population pfc { count: 100 }
synapse pfc -> pfc { weight: 0.5 }
"""


def test_unknown_field_rejected():
    """Typos in the DSL config block must fail loudly, not silently ignored."""
    bad_source = _MINIMAL_PROGRAM + """
    config {
        d_sem: 192
        this_field_does_not_exist: 42
    }
    """
    program = NeuroMLCompiler.compile(bad_source)
    try:
        to_brain_config(program)
    except Exception as e:
        assert 'this_field_does_not_exist' in str(e), f"unexpected error: {e}"
        print("[5] unknown field rejected: PASS")
        return
    raise AssertionError("compiler should have rejected unknown field")


def test_empty_config_returns_defaults():
    """A .neuro file with no config block returns a default BrainConfig."""
    program = NeuroMLCompiler.compile(_MINIMAL_PROGRAM)
    cfg = to_brain_config(program)
    default = BrainConfig()
    for f in dataclasses.fields(BrainConfig):
        assert getattr(cfg, f.name) == getattr(default, f.name), (
            f"empty-config compile should match BrainConfig defaults; "
            f"field {f.name} differs")
    print("[6] empty config returns defaults: PASS")


if __name__ == "__main__":
    print("=" * 60)
    print("DSL Phase 1 — config compile equivalence tests")
    print("=" * 60)
    test_config_block_parses()
    test_bool_coercion()
    test_numeric_coercion()
    test_equivalence_with_python_preset()
    test_unknown_field_rejected()
    test_empty_config_returns_defaults()
    print("=" * 60)
    print("ALL TESTS PASSED")
