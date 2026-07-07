# -*- coding: utf-8 -*-
"""`brian discover simplify` compiles an nn_lang layer to NGL and simplifies it."""
import json

from neuroslm import cli

FFN_DSL = """
layer FFN(D, H) {
    param gamma: (D,) init=ones
    param w1: (H, D) init=xavier
    param w2: (H, D) init=xavier
    param w3: (D, H) init=xavier
    forward(x) {
        h = rmsnorm(x, gamma)
        m = swiglu(h, w1, w2, w3)
        return x + m
    }
}
"""

BLOATED_DSL = """
layer Bloated(D) {
    param gamma: (D,) init=ones
    forward(x) {
        h = rmsnorm(x, gamma)
        a = h + h
        b = a - h
        return b
    }
}
"""


def test_simplify_layer_file_reports_reduction(tmp_path):
    src = tmp_path / "bloated.layer"
    src.write_text(BLOATED_DSL)
    out = tmp_path / "simp.json"
    rc = cli.main(["--no-verify", "discover", "simplify",
                   "--layer-file", str(src), "--out", str(out)])
    assert rc == 0
    data = json.loads(out.read_text())
    assert data["mode"] == "simplify"
    assert data["before"] >= data["after"]
    assert data["equivalent"] is True


def test_simplify_compiles_real_arch_block(tmp_path):
    src = tmp_path / "ffn.layer"
    src.write_text(FFN_DSL)
    out = tmp_path / "ffn.json"
    rc = cli.main(["--no-verify", "discover", "simplify",
                   "--layer-file", str(src), "--out", str(out)])
    assert rc == 0
    data = json.loads(out.read_text())
    # FFN is already minimal — simplify must not break equivalence
    assert data["equivalent"] is True
    assert data["after"] >= 1
