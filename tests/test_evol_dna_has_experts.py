"""Pin the canonical contract: ``dna/master/arch.dna`` carries the new
``multi_cortex.experts: [...]`` MoE roster.

Background
==========
Up until June 2026 the master DNA carried the legacy ``weights: "gpt2"``
multi-cortex config, which drives the broken random-projection chain
documented in scripts/diagnose_cortex_init.py (initial CE pinned at
~ln(V) because pretrained GPT-2 features were funneled through a
Xavier-init Linear → LayerNorm → tied-to-random-embed head).

The new MoE path (``experts: [...]``) makes every expert return logits
**directly** in trunk-vocab space via its own pretrained LM head — no
random projection. Smoking-gun CE drops from ~10.85 to ~3-5 nats at
step 0 (see tests/training/test_lm_expert_harness_integration.py).

This test ensures the master DNA never silently regresses back to the
legacy chain — a recompile from architectures/master must always carry
the ``experts: [...]`` block.

History
=======
2026-06-15: ``dna/evol/`` was deprecated (evolved-genome staging tree,
distinct concern). The single canonical DNA is now ``dna/master/arch.dna``
and is recompiled from ``architectures/master/`` by the pre-deploy hook.

When this test fails
====================
Re-run the recompile:

    python -c "from neuroslm.compiler.ribosome import RibosomeCompiler; \\
               RibosomeCompiler().compile_file('architectures/master', \\
                                               'dna/master/arch.dna')"

(or use the helper in ``scripts/recompile_evol_dna.py`` once added).
See CLAUDE.md §12 for the policy.
"""
from __future__ import annotations

import re
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parent.parent
# Canonical DNA path — `dna/master/arch.dna` is the source of truth
# (and the only DNA tracked under `dna/`). `dna/evol/` was retired on
# 2026-06-15; the evolved-genome staging tree lives elsewhere now.
DNA_PATH = REPO_ROOT / "dna" / "master" / "arch.dna"
RCC_ARCH_NEURO = REPO_ROOT / "architectures" / "master" / "arch.neuro"


def _read_unfolded_dsl(dna_path: Path) -> str:
    """Load DNA + unfold to its embedded DSL string."""
    from neuroslm.compiler.ribosome import LatentDNA, RibosomeCompiler

    dna = LatentDNA.load(str(dna_path))
    compiler = RibosomeCompiler()
    return compiler.dna_translator.translate(dna)


def _find_experts_code(dsl: str) -> str:
    """Return the ``experts: [ { ... }, ... ]`` code block (NOT comments).

    Filters out comment placeholders like ``experts: [...]`` (literal
    three-dot ellipsis used in the rationale comments) by requiring the
    first non-whitespace character inside the brackets to be ``{`` — the
    real roster always starts with a dict literal.
    """
    # Walk every `experts:` occurrence; return the first one whose
    # bracketed body contains a `{` (the dict literal). Comment-only
    # placeholders like `experts: [...]` will have `.`, not `{`.
    pos = 0
    while True:
        m = re.search(r"experts:\s*\[", dsl[pos:])
        if not m:
            return ""
        absolute_start = pos + m.start()
        bracket_open = pos + m.end() - 1
        # Balance brackets to find the matching ]
        depth = 0
        i = bracket_open
        while i < len(dsl):
            c = dsl[i]
            if c == "[":
                depth += 1
            elif c == "]":
                depth -= 1
                if depth == 0:
                    block = dsl[absolute_start:i + 1]
                    # Inner: skip the leading whitespace; first
                    # meaningful char must be '{' to be a real roster.
                    inner = block[block.index("[") + 1:-1].lstrip()
                    if inner.startswith("{"):
                        return block
                    break  # skip this match, look for the next one
            i += 1
        pos = bracket_open + 1


# ──────────────────────────────────────────────────────────────────────
# Contracts
# ──────────────────────────────────────────────────────────────────────


class TestEvolDnaCarriesExpertsRoster:
    """The compiled ``dna/master/arch.dna`` must contain the new MoE
    ``experts:`` block. If this fails, the DNA was compiled from a
    stale arch.neuro — recompile per CLAUDE.md §12.

    (Class name preserved for git-log continuity — the contract pinned
    here used to live under ``dna/evol/`` before the 2026-06-15 cleanup
    that retired the dual-tree convention.)"""

    def test_dna_file_exists(self):
        assert DNA_PATH.is_file(), (
            f"{DNA_PATH} not found; cannot validate MoE roster. "
            f"Recompile with: RibosomeCompiler().compile_file("
            f"'architectures/master', 'dna/master/arch.dna')"
        )

    def test_unfolded_dsl_has_experts_code_block(self):
        dsl = _read_unfolded_dsl(DNA_PATH)
        block = _find_experts_code(dsl)
        assert block, (
            "dna/master/arch.dna unfolds to a DSL with NO `experts: [...]` "
            "code block. The DNA is stale — it still encodes the legacy "
            "`weights: \"gpt2\"` random-projection chain. Recompile with: "
            "RibosomeCompiler().compile_file('architectures/master', "
            "'dna/master/arch.dna')"
        )

    def test_unfolded_dsl_has_three_default_experts(self):
        dsl = _read_unfolded_dsl(DNA_PATH)
        block = _find_experts_code(dsl)
        # 2026-06-15: canonical roster is
        # { smollm2_360m (general) | CodeGPT-small-py (code)
        #                          | Qwen2.5-0.5B (reasoning) }.
        # smollm2_360m supersedes the legacy `gpt2` slot — see
        # `architectures/master/arch.neuro` line ~208 for rationale.
        for expected_id in (
            "smollm2_360m",
            "microsoft/CodeGPT-small-py",
            "Qwen/Qwen2.5-0.5B",
        ):
            assert expected_id in block, (
                f"expected expert id {expected_id!r} missing from "
                f"recompiled DNA's experts: roster"
            )

    def test_training_config_loads_experts_from_dna(self, tmp_path):
        """End-to-end: unfold the DNA into a temp folder (preserving the
        modular tree), then load via the same path the harness uses
        in production. ``cfg.multi_cortex.experts`` must be the populated
        roster, not None."""
        from neuroslm.compiler.ribosome import RibosomeCompiler
        from neuroslm.dsl.training_config import (
            load_training_config_from_arch,
            ExpertSpec,
        )

        # Unfold to a temp tree (this is the same primitive
        # `brian dna unfold` uses; preserves modules/ and lib/).
        out_arch_neuro = tmp_path / "arch.neuro"
        RibosomeCompiler().unfold_file(str(DNA_PATH), str(out_arch_neuro))
        cfg = load_training_config_from_arch(tmp_path)

        assert cfg.multi_cortex.enabled, (
            "master DNA must enable multi_cortex"
        )
        assert cfg.multi_cortex.experts is not None, (
            "master DNA must declare a non-empty experts: roster"
        )
        assert len(cfg.multi_cortex.experts) >= 1
        for e in cfg.multi_cortex.experts:
            assert isinstance(e, ExpertSpec)
            assert e.id and e.domain


class TestRccBowtieAndEvolArchAreInSync:
    """Historically (pre-2026-06-15) ``architectures/master/`` and
    ``architectures/evol/`` were two parallel trees that had to stay
    in sync. That dual-tree convention was retired: ``architectures/evol/``
    is now reserved exclusively for the evolved-genome staging output
    of the genetic-evolution loop (the "DNA mutated by the trainer at
    runtime"), NOT a hand-edited mirror of master.

    The MoE roster's source of truth is ``architectures/master/arch.neuro``
    and its compiled snapshot ``dna/master/arch.dna``. Cross-checks
    against ``TestEvolDnaCarriesExpertsRoster`` (which pins the DNA
    contents directly) enforce correctness on CI; this class remains as
    a documentation marker plus a skip-only safety net for any developer
    who still has a legacy ``architectures/evol/arch.neuro`` checked
    out locally."""

    def test_multi_cortex_experts_match(self):
        legacy_evol = REPO_ROOT / "architectures" / "evol" / "arch.neuro"
        if not legacy_evol.is_file():
            pytest.skip(
                "architectures/evol/arch.neuro not present — the "
                "dual-tree (master/evol) sync convention was retired "
                "on 2026-06-15; canonical source is architectures/master/. "
                "TestEvolDnaCarriesExpertsRoster covers the DNA contract."
            )
        rcc = RCC_ARCH_NEURO.read_text(encoding="utf-8")
        evol = legacy_evol.read_text(encoding="utf-8")
        rcc_block = _find_experts_code(rcc)
        evol_block = _find_experts_code(evol)
        assert rcc_block, (
            f"no experts: [...] block in {RCC_ARCH_NEURO}; "
            "the canonical arch is missing the MoE roster"
        )
        assert rcc_block == evol_block, (
            f"experts: rosters diverge between\n"
            f"  {RCC_ARCH_NEURO} (canonical)\n"
            f"  {legacy_evol} (legacy mirror)\n"
            f"You have a stale architectures/evol/arch.neuro checked "
            f"out — either delete it or resync from master. The "
            f"canonical source is now architectures/master/."
        )
