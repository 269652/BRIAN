# -*- coding: utf-8 -*-
"""Flow + compute heat profiler for NGL execution.

Records, per instruction, how much *information* flows out (activation norm) and
how much *computation* it costs (estimated FLOPs), so we can see where heavy
compute and heavy flow are — and rank "low-hanging fruit" (high flow, low
compute) for the search to target first.
"""
import math

import torch

from neuroslm.genetic.language import Memory, Program, Instruction
from neuroslm.genetic.attention_primitives import single_head_attention_program
from neuroslm.genetic.profile import profile_program, ExecutionProfile


class TestProfileRecording:
    def test_records_a_node_per_instruction(self):
        prog = Program(
            [
                Instruction("add", "t2", ("t0", "t1")),
                Instruction("mul", "t3", ("t2", "t2")),
            ],
            n_scalar=2, n_tensor=6, out_reg="t3",
        )
        prof = profile_program(prog, {"t0": torch.randn(8), "t1": torch.randn(8)})
        assert isinstance(prof, ExecutionProfile)
        assert len(prof.nodes) == 2
        assert prof.nodes[0].op == "add"
        assert prof.nodes[1].op == "mul"

    def test_flow_is_output_norm(self):
        prog = Program([Instruction("cscale", "t2", ("t0",), const=3.0)],
                       n_scalar=2, n_tensor=4, out_reg="t2")
        x = torch.ones(4)
        prof = profile_program(prog, {"t0": x})
        # cscale(ones*3) → norm = 3 * ||ones(4)|| = 3*2 = 6
        assert math.isclose(prof.nodes[0].flow, 6.0, rel_tol=1e-5)

    def test_matmul_costs_more_compute_than_add(self):
        prog = Program(
            [
                Instruction("add", "t3", ("t0", "t0")),
                Instruction("matmul", "t4", ("t1", "t2")),
            ],
            n_scalar=2, n_tensor=8, out_reg="t4",
        )
        prof = profile_program(prog, {
            "t0": torch.randn(16, 16),
            "t1": torch.randn(16, 16),
            "t2": torch.randn(16, 16),
        })
        add_node = [n for n in prof.nodes if n.op == "add"][0]
        mm_node = [n for n in prof.nodes if n.op == "matmul"][0]
        assert mm_node.flops > add_node.flops * 5


class TestRankings:
    def test_heavy_compute_and_hot_flow_rankings(self):
        prof = profile_program(single_head_attention_program(scale=0.35), {
            "t0": torch.randn(2, 8, 8),
            "t1": torch.randn(8, 8) * 0.1,
            "t2": torch.randn(8, 8) * 0.1,
            "t3": torch.randn(8, 8) * 0.1,
            "t4": torch.randn(8, 8) * 0.1,
        })
        heavy = prof.heavy_compute(top=3)
        assert len(heavy) == 3
        # the two matmuls / linears dominate compute
        assert any(n.op in ("matmul", "linear") for n in heavy)
        hot = prof.hot_flow(top=3)
        assert len(hot) == 3
        assert all(h.flow >= 0 for h in hot)

    def test_low_hanging_prefers_high_flow_low_compute(self):
        # construct nodes: one cheap+high-flow, one expensive+low-flow
        prog = Program(
            [
                Instruction("cscale", "t2", ("t0",), const=100.0),   # cheap, high flow
                Instruction("matmul", "t3", ("t1", "t1")),           # expensive
            ],
            n_scalar=2, n_tensor=6, out_reg="t3",
        )
        prof = profile_program(prog, {
            "t0": torch.ones(32),
            "t1": torch.randn(24, 24) * 1e-3,   # tiny values → low flow
        })
        lh = prof.low_hanging(top=1)
        assert lh[0].op == "cscale"   # high flow / low compute wins


class TestExport:
    def test_to_dict_is_serialisable(self):
        import json
        prog = Program([Instruction("add", "t2", ("t0", "t1"))], 2, 4, "t2")
        prof = profile_program(prog, {"t0": torch.randn(4), "t1": torch.randn(4)})
        d = prof.to_dict()
        assert "nodes" in d and "edges" in d
        json.dumps(d)  # must not raise
        # edges connect producer → consumer registers
        assert isinstance(d["edges"], list)
