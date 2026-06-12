# -*- coding: utf-8 -*-
"""End-to-end DSL contract: a `feature` block in `arch.neuro` may
reference an equation imported from a `lib/equations.neuro` sibling.

This pins the bridge between two existing primitives (multi-file
import, equation extraction) and the new `feature` block — without
this test, breaking the bridge during a future refactor would only
surface at full-architecture compile time.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from neuroslm.dsl.multifile import compile_folder


def _write(root: Path, rel: str, content: str) -> Path:
    p = root / rel
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(content, encoding="utf-8")
    return p


def _build_minimal_arch(root: Path, *, active: str = "false") -> Path:
    """Materialise an arch directory with one lib equation and one
    feature block referencing it. Returns the arch root."""
    _write(
        root,
        "lib/equations.neuro",
        """
        export equation hyperbolic_attention_eq {
            params: [d_model, n_heads, c],
            formula: "softmax(-d_hyp(Q, K) / sqrt(d_head)) @ V"
        }
        """,
    )
    _write(
        root,
        "arch.neuro",
        f"""
        import {{ hyperbolic_attention_eq }} from "@/lib/equations"

        architecture toy {{
            d_sem: 64,
            dt: 0.01
        }}

        population dummy {{
            count: 16,
            dynamics: "rate_code"
        }}

        feature hyperbolic_attention {{
            equation: hyperbolic_attention_eq,
            active: {active},
            params: {{ d_model: 64, n_heads: 4, c: 1.0 }}
        }}
        """,
    )
    return root


class TestFeatureReferencesImportedEquation:
    def test_feature_compiles_with_imported_equation(self, tmp_path: Path):
        arch = _build_minimal_arch(tmp_path)
        program = compile_folder(arch)
        # Equation made it through the multi-file resolver
        eq_names = {e.name for e in program.equation_decls}
        assert "hyperbolic_attention_eq" in eq_names
        # Feature was extracted. compile_folder currently emits arch.neuro
        # twice (once piecewise, once raw for THSD context), so the
        # feature may appear duplicated; the contract is that at least
        # one canonical instance exists with the right shape.
        attn_feats = [
            f for f in program.features
            if f.name == "hyperbolic_attention"
        ]
        assert len(attn_feats) >= 1
        feat = attn_feats[0]
        assert feat.equation_ref == "hyperbolic_attention_eq"
        assert feat.active is False

    def test_feature_active_true_round_trips(self, tmp_path: Path):
        arch = _build_minimal_arch(tmp_path, active="true")
        program = compile_folder(arch)
        attn_feats = [
            f for f in program.features
            if f.name == "hyperbolic_attention"
        ]
        assert all(f.active is True for f in attn_feats)

    def test_feature_params_dict_is_carried_through(self, tmp_path: Path):
        arch = _build_minimal_arch(tmp_path)
        program = compile_folder(arch)
        attn_feats = [
            f for f in program.features
            if f.name == "hyperbolic_attention"
        ]
        params = attn_feats[0].params
        assert params["d_model"] == 64
        assert params["n_heads"] == 4
        # c=1.0 should parse as numeric (the extractor coerces)
        assert float(params["c"]) == pytest.approx(1.0)

    def test_feature_referencing_unknown_equation_is_a_compile_error(
        self, tmp_path: Path
    ):
        """The compile-time validation in `_extract_features` must
        catch this; otherwise the failure shifts to model-build time
        where the error message is much harder to trace."""
        _write(
            tmp_path,
            "lib/equations.neuro",
            """
            export equation only_this_one_exists {
                params: [x],
                formula: "y = x"
            }
            """,
        )
        _write(
            tmp_path,
            "arch.neuro",
            """
            import { only_this_one_exists } from "@/lib/equations"

            architecture toy { d_sem: 32, dt: 0.01 }
            population dummy { count: 8, dynamics: "rate_code" }

            feature broken {
                equation: nonexistent_equation,
                active: false
            }
            """,
        )
        with pytest.raises(Exception) as excinfo:
            compile_folder(tmp_path)
        msg = str(excinfo.value)
        assert "nonexistent_equation" in msg or "broken" in msg
