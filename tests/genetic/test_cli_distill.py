# -*- coding: utf-8 -*-
"""`brian discover distill` runs the λ-schedule auto-evolve end-to-end."""
import json
from neuroslm import cli


def test_discover_distill_runs_and_writes_json(tmp_path):
    out = tmp_path / "distill.json"
    rc = cli.main(["--no-verify", "discover", "distill",
                   "--pop", "6", "--generations", "2", "--steps", "20",
                   "--seed", "0", "--out", str(out)])
    assert rc == 0
    data = json.loads(out.read_text())
    assert data["mode"] == "distill"
    assert "baseline_final_loss" in data and "best_final_loss" in data
    assert "best_plausibility" in data
    assert data["best_program"]


def test_discover_distill_save_writes_modulation(tmp_path, monkeypatch):
    monkeypatch.setenv("BRIAN_MODULATIONS_DIR", str(tmp_path))
    rc = cli.main(["--no-verify", "discover", "distill",
                   "--pop", "6", "--generations", "2", "--steps", "20",
                   "--seed", "0", "--save", "cli_test_distill"])
    assert rc == 0
    saved = tmp_path / "cli_test_distill.neuro"
    assert saved.exists()
    from neuroslm.genetic.distill_evolve import install_distillation_schedule_from_store

    class _FakeHarness:
        def install_distillation_schedule(self, fn):
            self.fn = fn

    h = _FakeHarness()
    report = install_distillation_schedule_from_store(
        h, "cli_test_distill", store_dir=tmp_path)
    assert report["installed"] == "cli_test_distill"
    assert callable(h.fn)
