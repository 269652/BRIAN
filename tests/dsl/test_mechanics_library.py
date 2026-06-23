# -*- coding: utf-8 -*-
"""RED-first contract for the reusable `.neuro` mechanic library.

The repo ships a documentation-first, machine-readable catalog of neural
mechanisms as `mechanic NAME { ... }` blocks (parsed by
`neuroslm.dsl.mechanic_parser`). This catalog is the vocabulary that the
auto-evolution / mutation engine draws from when it wants to graft a new
mechanism into an architecture, so it must:

  * live in three top-level folders by role —
        mechanics/   computational primitives (attention, FFN, norm, mixing)
        dynamics/    training & optimization (optimizers, schedules, losses)
        structures/  wiring patterns (blocks, residual, routing, tying)
  * parse cleanly with `parse_mechanic_file`
  * carry the load-bearing fields every consumer reads
        (category, summary, equation, impl, when_to_use)
  * declare a `default` for every configurable param
  * have globally-unique mechanic names
  * be fully enumerated in the top-level index `mechanics.md`
  * cover the standard + active-research mechanisms in modern LMs

Every contract below is RED until the library + index are authored.
"""
from __future__ import annotations

import pathlib

import pytest

from neuroslm.dsl.mechanic_parser import MechanicSpec, parse_mechanic_file

REPO = pathlib.Path(__file__).resolve().parents[2]
LIB_DIRS = ("mechanics", "dynamics", "structures")
INDEX = REPO / "mechanics.md"

# Standard + active-research mechanisms that the catalog MUST cover.
# Names are the `mechanic <name>` identifiers (folder is informational).
REQUIRED = {
    "mechanics": {
        # position
        "rope", "alibi", "nope",
        # attention sparsity / efficiency
        "gqa", "mqa", "mla", "sliding_window_attention",
        "attention_sink", "flash_attention",
        # attention stabilizers
        "qk_norm", "logit_soft_cap",
        # feed-forward
        "swiglu", "geglu", "gated_mlp",
        # sparse experts
        "sparse_moe", "expert_choice_routing", "shared_expert",
        # normalization
        "rmsnorm", "layernorm", "deepnorm",
        # alternative sequence mixers
        "mamba_ssm", "linear_attention", "gated_linear_attention",
        "retnet", "hyena",
    },
    "dynamics": {
        "muon", "adamw", "lion",
        "cosine_schedule", "wsd_schedule",
        "z_loss", "label_smoothing", "dropout",
        "weight_decay", "gradient_clipping", "ema_weights",
    },
    "structures": {
        "prenorm_block", "postnorm_block", "parallel_block",
        "residual_stream", "moe_block", "mod_block",
        "weight_tying", "sandwich_norm", "depth_scaled_init",
    },
}

REQUIRED_FIELDS = ("category", "summary", "equation", "impl", "when_to_use")


# ── discovery helpers ────────────────────────────────────────────────────


def _neuro_files(folder: str) -> list[pathlib.Path]:
    d = REPO / folder
    return sorted(d.glob("*.neuro")) if d.is_dir() else []


def _all_neuro_files() -> list[pathlib.Path]:
    out: list[pathlib.Path] = []
    for d in LIB_DIRS:
        out.extend(_neuro_files(d))
    return out


def _all_specs() -> list[tuple[pathlib.Path, MechanicSpec]]:
    pairs: list[tuple[pathlib.Path, MechanicSpec]] = []
    for f in _all_neuro_files():
        for spec in parse_mechanic_file(f.read_text(encoding="utf-8")):
            pairs.append((f, spec))
    return pairs


# ── structure ────────────────────────────────────────────────────────────


class TestLibraryLayout:
    @pytest.mark.parametrize("folder", LIB_DIRS)
    def test_folder_exists(self, folder):
        assert (REPO / folder).is_dir(), f"missing library folder: {folder}/"

    @pytest.mark.parametrize("folder", LIB_DIRS)
    def test_folder_non_empty(self, folder):
        assert _neuro_files(folder), f"{folder}/ contains no .neuro files"

    def test_index_exists(self):
        assert INDEX.is_file(), "missing top-level index mechanics.md"


# ── parse validity ───────────────────────────────────────────────────────


class TestEveryFileParses:
    def test_every_file_yields_a_spec(self):
        for f in _all_neuro_files():
            specs = parse_mechanic_file(f.read_text(encoding="utf-8"))
            assert specs, f"{f.name}: no mechanic block parsed"

    def test_required_fields_present(self):
        for f, spec in _all_specs():
            for field in REQUIRED_FIELDS:
                assert getattr(spec, field), (
                    f"{f.name}::{spec.name}: empty required field {field!r}"
                )

    def test_params_declare_default(self):
        for f, spec in _all_specs():
            for pname, pspec in spec.params.items():
                assert pspec.default is not None, (
                    f"{f.name}::{spec.name}: param {pname!r} has no default"
                )

    def test_names_globally_unique(self):
        seen: dict[str, str] = {}
        for f, spec in _all_specs():
            assert spec.name not in seen, (
                f"duplicate mechanic name {spec.name!r} in {f.name} "
                f"and {seen[spec.name]}"
            )
            seen[spec.name] = f.name


# ── coverage ─────────────────────────────────────────────────────────────


class TestCoverage:
    @pytest.mark.parametrize("folder", LIB_DIRS)
    def test_required_mechanics_present(self, folder):
        names = {
            spec.name
            for f in _neuro_files(folder)
            for spec in parse_mechanic_file(f.read_text(encoding="utf-8"))
        }
        missing = REQUIRED[folder] - names
        assert not missing, f"{folder}/ missing required mechanics: {sorted(missing)}"


# ── index sync ───────────────────────────────────────────────────────────


class TestIndexSync:
    def test_index_lists_every_mechanic(self):
        if not INDEX.is_file():
            pytest.fail("mechanics.md does not exist")
        text = INDEX.read_text(encoding="utf-8")
        for f, spec in _all_specs():
            assert spec.name in text, (
                f"mechanics.md does not mention {spec.name!r} (from {f.name})"
            )
