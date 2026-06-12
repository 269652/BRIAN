# -*- coding: utf-8 -*-
"""End-to-end: the real ``architectures/lib/`` shared library compiles.

This is the integration smoke test for the canonical layout:

    architectures/
        lib/
            equations.neuro
            features/
                hyperbolic_attention.neuro

It builds a minimal architecture that imports the hyperbolic_attention
feature from the real (not synthetic) shared lib and asserts the
full multi-file resolution machinery — PathResolver, lazy lib loading,
equation extraction, feature extraction, endpoint parsing — all the way
through ``compile_folder`` returns a coherent ``ProgramIR``.

If this test ever breaks, the wiring of a new mechanism has drifted from
the layout convention documented in
``architectures/lib/equations.neuro``.
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterator

import pytest

from neuroslm.dsl.multifile import compile_folder


REPO_ROOT = Path(__file__).resolve().parents[2]
SHARED_LIB = REPO_ROOT / "architectures" / "lib"


@pytest.fixture
def consumer_arch(tmp_path: Path) -> Iterator[Path]:
    """Build a minimal arch dir whose arch.neuro imports the real
    shared feature. The arch lives under the real repo's
    ``architectures/`` so ``@brian/`` resolution finds the real
    shared lib."""
    # Place the consumer arch under the real architectures/ dir so
    # repo_root auto-discovery finds the real pyproject.toml.
    arch_dir = REPO_ROOT / "architectures" / "_test_consumer_tmp"
    arch_dir.mkdir(parents=True, exist_ok=True)
    arch_file = arch_dir / "arch.neuro"
    arch_file.write_text(
        """
        import { hyperbolic_attention } from "@brian/features/hyperbolic_attention"

        architecture consumer {
            d_sem: 64,
            dt: 0.01
        }

        population pre  { count: 16, dynamics: "rate_code" }
        population post { count: 16, dynamics: "rate_code" }

        synapse pre -> post {
            feature: "hyperbolic_attention.edge",
            weight: 1.0
        }
        """,
        encoding="utf-8",
    )
    yield arch_dir
    # Cleanup: never leave a test arch in the real architectures/ dir
    arch_file.unlink(missing_ok=True)
    try:
        arch_dir.rmdir()
    except OSError:
        pass


class TestSharedLibIntegration:
    def test_shared_lib_files_exist(self):
        """Sanity: the real files referenced below actually exist.
        If this fails, the layout convention has been broken."""
        assert (SHARED_LIB / "equations.neuro").is_file()
        assert (
            SHARED_LIB / "features" / "hyperbolic_attention.neuro"
        ).is_file()

    def test_consumer_arch_compiles_with_shared_feature(self, consumer_arch):
        program = compile_folder(consumer_arch)
        # The hyperbolic_attention_eq equation was loaded from
        # @brian/equations (transitively, via the feature file).
        eq_names = {e.name for e in program.equation_decls}
        assert "hyperbolic_attention_eq" in eq_names, (
            f"shared equation missing; got {sorted(eq_names)}"
        )

    def test_feature_is_extracted_with_endpoint(self, consumer_arch):
        program = compile_folder(consumer_arch)
        # compile_folder may emit the feature twice (THSD context append);
        # pick the first canonical instance and assert its shape.
        attn = [
            f for f in program.features
            if f.name == "hyperbolic_attention"
        ]
        assert len(attn) >= 1, (
            f"feature not extracted; got {[f.name for f in program.features]}"
        )
        feat = attn[0]
        assert (
            feat.impl
            == "neuroslm.modules.hyperbolic_attention.HyperbolicMultiHeadAttention"
        )
        assert feat.active is False, (
            "default state must be OFF (clean baseline)"
        )
        ep_names = {ep.name for ep in feat.endpoints}
        assert "edge" in ep_names

    def test_synapse_carries_feature_ref(self, consumer_arch):
        program = compile_folder(consumer_arch)
        syns = [s for s in program.synapses if s.source == "pre"]
        # compile_folder currently emits arch.neuro twice (once piecewise,
        # once raw for THSD context) so synapses may be duplicated; the
        # contract we care about is that every matching synapse carries
        # the correct feature_ref.
        assert len(syns) >= 1
        assert all(
            s.feature_ref == "hyperbolic_attention.edge" for s in syns
        ), f"got mismatched feature_refs: {[s.feature_ref for s in syns]}"
