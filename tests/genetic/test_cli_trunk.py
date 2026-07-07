# -*- coding: utf-8 -*-
"""`brian discover trunk` runs the neuroanatomic auto-evolve end-to-end."""
import json
from neuroslm import cli


def test_discover_trunk_runs_and_writes_json(tmp_path):
    out = tmp_path / "trunk.json"
    rc = cli.main(["--no-verify", "discover", "trunk",
                   "--pop", "6", "--generations", "2", "--steps", "15",
                   "--seed", "0", "--out", str(out)])
    assert rc == 0
    data = json.loads(out.read_text())
    assert data["mode"] == "trunk"
    assert "baseline_val_ppl" in data and "best_val_ppl" in data
    assert "best_plausibility" in data
    assert data["best_program"]
