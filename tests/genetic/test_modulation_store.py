# -*- coding: utf-8 -*-
"""Modulations persist as modulations/*.neuro and merge/drop via the store.

A discovered NGL neuromodulation is a first-class, versionable artifact: it
serializes to a `.neuro` `modulation { … }` block (round-trip exact), can be
listed, shown, merged (gains compose) or thrown away — all from `brian`.
"""
import torch

from neuroslm.genetic.language import Instruction, Memory, Program
from neuroslm.genetic.modulation_store import (
    ModulationRecord,
    ModulationStore,
    parse_program,
    program_to_neuro,
    merge_programs,
)


def _run(prog, h):
    mem = Memory(prog.n_scalar, prog.n_tensor)
    mem.write("t0", h)
    prog.execute(mem)
    return mem.read(prog.out_reg)


def _tanh_gain():
    return Program([Instruction("tanh", "t2", ("t0",))], n_scalar=4, n_tensor=6, out_reg="t2")


class TestProgramSerialization:
    def test_source_roundtrip_preserves_behaviour(self):
        prog = Program(
            [
                Instruction("rms", "t2", ("t0",)),
                Instruction("sigmoid", "t3", ("t2",)),
                Instruction("cscale", "t4", ("t3",), const=0.5),
            ],
            n_scalar=4, n_tensor=8, out_reg="t4",
        )
        text = prog.to_source()
        back = parse_program(text, n_scalar=4, n_tensor=8)
        h = torch.randn(2, 3, 5)
        assert torch.allclose(_run(prog, h), _run(back, h), atol=1e-6)

    def test_config_op_roundtrips(self):
        prog = Program(
            [Instruction("causal_self_attention", "t5", ("t0", "t1", "t2", "t3"),
                         config=(("n_heads", 4.0), ("n_kv_heads", 2.0),
                                 ("max_ctx", 16.0), ("rope_base", 10000.0)))],
            n_scalar=4, n_tensor=8, out_reg="t5",
        )
        back = parse_program(prog.to_source(), n_scalar=4, n_tensor=8)
        assert back.instructions[0].op == "causal_self_attention"
        assert dict(back.instructions[0].config)["n_heads"] == 4.0


class TestNeuroFormat:
    def test_neuro_block_has_name_and_program(self):
        rec = ModulationRecord(name="trunk_tanh", program=_tanh_gain(),
                               metrics={"val_ppl": 7.75, "plausibility": 0.6})
        text = program_to_neuro(rec)
        assert "modulation trunk_tanh {" in text
        assert "program {" in text
        assert "tanh(t0)" in text
        assert "val_ppl" in text


class TestStore:
    def test_save_list_show_drop(self, tmp_path):
        store = ModulationStore(tmp_path)
        rec = ModulationRecord(name="gainA", program=_tanh_gain(),
                               metrics={"val_ppl": 7.7})
        path = store.save(rec)
        assert path.exists() and path.suffix == ".neuro"
        names = [r.name for r in store.list_all()]
        assert "gainA" in names
        loaded = store.get("gainA")
        h = torch.randn(4, 5)
        assert torch.allclose(_run(loaded.program, h), _run(rec.program, h), atol=1e-6)
        store.drop("gainA")
        assert "gainA" not in [r.name for r in store.list_all()]

    def test_merge_composes_gains(self, tmp_path):
        store = ModulationStore(tmp_path)
        # g1 = tanh(h) ; g2 = sigmoid(h) ; merged applies them in sequence
        g1 = Program([Instruction("tanh", "t2", ("t0",))], 4, 6, "t2")
        g2 = Program([Instruction("sigmoid", "t2", ("t0",))], 4, 6, "t2")
        store.save(ModulationRecord("g1", g1, {}))
        store.save(ModulationRecord("g2", g2, {}))
        merged = merge_programs([g1, g2])
        h = torch.randn(3, 4)
        expected = torch.sigmoid(torch.tanh(h))  # g2(g1(h))
        assert torch.allclose(_run(merged, h), expected, atol=1e-6)
        rec = store.merge(["g1", "g2"], "combo")
        assert rec.name == "combo"
        assert torch.allclose(_run(store.get("combo").program, h), expected, atol=1e-6)
