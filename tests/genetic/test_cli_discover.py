# -*- coding: utf-8 -*-
"""The `brian discover` CLI wires the NGL discovery harness end-to-end."""
import json

from neuroslm import cli


def test_discover_optimizer_runs_and_writes_json(tmp_path, capsys):
    out = tmp_path / "run.json"
    rc = cli.main([
        "--no-verify", "discover", "optimizer",
        "--pop", "12", "--generations", "3", "--steps", "20",
        "--seed", "0", "--out", str(out),
    ])
    assert rc == 0
    assert out.exists()
    data = json.loads(out.read_text())
    assert data["mode"] == "optimizer"
    assert "best_final_loss" in data
    assert "sgd_baseline_loss" in data
    assert data["best_program"]  # NGL source of the discovered rule


def test_discover_flow_runs(tmp_path):
    out = tmp_path / "flow.json"
    rc = cli.main([
        "--no-verify", "discover", "flow",
        "--pop", "10", "--generations", "2", "--steps", "15",
        "--seed", "0", "--out", str(out),
    ])
    assert rc == 0
    data = json.loads(out.read_text())
    assert data["mode"] == "flow"
    assert "best_ei" in data
