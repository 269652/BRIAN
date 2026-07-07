# -*- coding: utf-8 -*-
"""`brian modulation` lifecycle + `brian discover trunk --save`."""
import json
from neuroslm import cli


def test_discover_trunk_save_then_modulation_lifecycle(tmp_path, monkeypatch):
    monkeypatch.setenv("BRIAN_MODULATIONS_DIR", str(tmp_path))
    # discover + save a modulation
    rc = cli.main(["--no-verify", "discover", "trunk",
                   "--pop", "6", "--generations", "2", "--steps", "12",
                   "--seed", "0", "--save", "gainX"])
    assert rc == 0
    assert (tmp_path / "gainX.neuro").exists()

    # list shows it
    rc = cli.main(["--no-verify", "modulation", "list"])
    assert rc == 0

    # show prints the .neuro
    rc = cli.main(["--no-verify", "modulation", "show", "gainX"])
    assert rc == 0

    # merge two copies
    cli.main(["--no-verify", "discover", "trunk", "--pop", "6", "--generations",
              "2", "--steps", "12", "--seed", "1", "--save", "gainY"])
    rc = cli.main(["--no-verify", "modulation", "merge", "gainX", "gainY",
                   "--name", "combo"])
    assert rc == 0
    assert (tmp_path / "combo.neuro").exists()

    # drop
    rc = cli.main(["--no-verify", "modulation", "drop", "gainX"])
    assert rc == 0
    assert not (tmp_path / "gainX.neuro").exists()
