# -*- coding: utf-8 -*-
"""Device-awareness: NGL executes correctly on whatever device its tensors live on.

The discovery harness runs on CPU by default but must scale to a T4/cuda when one
is present. The subtle correctness issue is NGL's constants (from `const`/eps
guards): they must follow the operand device, or a `cuda_tensor + cpu_scalar`
raises and the op silently falls back — corrupting the math. These contracts pin
the device-follow behaviour (verified on CPU; the cuda path is the same code) and
the `--device cuda` graceful fallback when no GPU is present.
"""
import math

import torch

from neuroslm.genetic.language import Instruction, Memory, Program
from neuroslm.genetic.optimizer import adam_program
from neuroslm.genetic.discovery import (
    benchmark_optimizer,
    run_optimizer_discovery,
    _resolve_device,
)
from neuroslm.genetic.neuro_evolve import run_trunk_evolution


class TestMemoryDevice:
    def test_default_device_is_cpu(self):
        mem = Memory(2, 4)
        assert mem.device.type == "cpu"
        assert mem.read("t0").device.type == "cpu"

    def test_device_tracks_written_tensor(self):
        mem = Memory(2, 4)
        mem.write("t0", torch.randn(3))
        assert mem.device.type == "cpu"  # cpu tensor keeps cpu
        # unwritten reads come back on the tracked device
        assert mem.read("t3").device == mem.device

    def test_const_result_matches_memory_device(self):
        # a const flows into a binary op with a written tensor; both must align
        prog = Program(
            [
                Instruction("const", "t2", (), const=0.5),
                Instruction("add", "t3", ("t0", "t2")),
            ],
            n_scalar=2, n_tensor=6, out_reg="t3",
        )
        mem = Memory(2, 6)
        mem.write("t0", torch.ones(4))
        prog.execute(mem)
        out = mem.read("t3")
        assert out.device == mem.device
        assert torch.allclose(out, torch.full((4,), 1.5))


class TestResolveDevice:
    def test_cpu_resolves_to_cpu(self):
        assert _resolve_device("cpu").type == "cpu"

    def test_cuda_falls_back_to_cpu_when_unavailable(self):
        dev = _resolve_device("cuda")
        if torch.cuda.is_available():
            assert dev.type == "cuda"
        else:
            assert dev.type == "cpu"  # graceful fallback, no crash

    def test_auto_picks_available(self):
        dev = _resolve_device("auto")
        assert dev.type in ("cpu", "cuda")


class TestHarnessDeviceThreading:
    def test_benchmark_accepts_device(self):
        res = benchmark_optimizer(adam_program(lr=0.05), steps=20, seed=0, device="cpu")
        assert math.isfinite(res.final_loss)

    def test_optimizer_discovery_device_cuda_falls_back(self):
        # requesting cuda on a CPU box must run, not crash
        outcome = run_optimizer_discovery(seed=0, pop_size=10, generations=3,
                                          steps=20, device="cuda")
        assert math.isfinite(outcome.best_final_loss)

    def test_trunk_evolution_accepts_device(self):
        outcome = run_trunk_evolution(seed=0, pop_size=6, generations=2,
                                      steps=15, device="cpu")
        assert math.isfinite(outcome.best_val_ppl)
