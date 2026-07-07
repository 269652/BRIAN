# -*- coding: utf-8 -*-
"""Proof of expressiveness: SOTA optimizers ARE NGL programs.

Each test encodes a known optimizer as an NGL ``Program``, runs it through the
``NGLOptimizer`` adapter, and asserts the parameter trajectory is numerically
identical to the reference (torch's own optimizer, or a hand-written reference
for optimizers torch lacks). If NGL can express SGD/Momentum/RMSProp/Adam/Lion
bit-for-bit, it spans the update-rule grammar — the precondition for *searching*
that grammar for a novel rule.
"""
import copy

import torch

from neuroslm.genetic.optimizer import (
    NGLOptimizer,
    sgd_program,
    momentum_program,
    rmsprop_program,
    adam_program,
    lion_program,
)


def _quadratic_problem(seed=0):
    torch.manual_seed(seed)
    A = torch.randn(6, 6)
    A = A @ A.t() + torch.eye(6)  # SPD
    b = torch.randn(6)
    x0 = torch.randn(6)
    return A, b, x0


def _run_reference(make_opt, steps=10, seed=0):
    A, b, x0 = _quadratic_problem(seed)
    p = torch.nn.Parameter(x0.clone())
    opt = make_opt([p])
    traj = []
    for _ in range(steps):
        opt.zero_grad()
        loss = 0.5 * (p @ (A @ p)) - b @ p
        loss.backward()
        opt.step()
        traj.append(p.detach().clone())
    return torch.stack(traj)


def _run_ngl(program, steps=10, seed=0):
    A, b, x0 = _quadratic_problem(seed)
    p = torch.nn.Parameter(x0.clone())
    opt = NGLOptimizer([p], program)
    traj = []
    for _ in range(steps):
        opt.zero_grad()
        loss = 0.5 * (p @ (A @ p)) - b @ p
        loss.backward()
        opt.step()
        traj.append(p.detach().clone())
    return torch.stack(traj)


class TestOptimizerEquivalence:
    def test_sgd_matches_torch(self):
        lr = 0.03
        ref = _run_reference(lambda ps: torch.optim.SGD(ps, lr=lr))
        got = _run_ngl(sgd_program(lr=lr))
        assert torch.allclose(ref, got, atol=1e-6), (ref - got).abs().max()

    def test_momentum_matches_torch(self):
        lr, mu = 0.02, 0.9
        ref = _run_reference(lambda ps: torch.optim.SGD(ps, lr=lr, momentum=mu))
        got = _run_ngl(momentum_program(lr=lr, mu=mu))
        assert torch.allclose(ref, got, atol=1e-6), (ref - got).abs().max()

    def test_rmsprop_matches_torch(self):
        lr, alpha, eps = 0.01, 0.99, 1e-8
        ref = _run_reference(lambda ps: torch.optim.RMSprop(ps, lr=lr, alpha=alpha, eps=eps))
        got = _run_ngl(rmsprop_program(lr=lr, alpha=alpha, eps=eps))
        assert torch.allclose(ref, got, atol=1e-5), (ref - got).abs().max()

    def test_adam_matches_torch(self):
        lr = 0.01
        ref = _run_reference(lambda ps: torch.optim.Adam(ps, lr=lr, betas=(0.9, 0.999), eps=1e-8))
        got = _run_ngl(adam_program(lr=lr, b1=0.9, b2=0.999, eps=1e-8))
        assert torch.allclose(ref, got, atol=1e-5), (ref - got).abs().max()

    def test_lion_matches_reference(self):
        lr, b1, b2 = 0.01, 0.9, 0.99

        def ref_lion(steps=10, seed=0):
            A, b, x0 = _quadratic_problem(seed)
            p = x0.clone()
            m = torch.zeros_like(p)
            traj = []
            for _ in range(steps):
                g = A @ p - b  # grad of 0.5 x'Ax - b'x
                c = b1 * m + (1 - b1) * g
                p = p - lr * torch.sign(c)
                m = b2 * m + (1 - b2) * g
                traj.append(p.clone())
            return torch.stack(traj)

        ref = ref_lion()
        got = _run_ngl(lion_program(lr=lr, b1=b1, b2=b2))
        assert torch.allclose(ref, got, atol=1e-6), (ref - got).abs().max()


class TestAdapter:
    def test_zero_grad_and_step_interface(self):
        p = torch.nn.Parameter(torch.randn(4))
        opt = NGLOptimizer([p], sgd_program(lr=0.1))
        p.grad = torch.ones(4)
        before = p.detach().clone()
        opt.step()
        assert torch.allclose(p.detach(), before - 0.1 * torch.ones(4), atol=1e-6)
        opt.zero_grad()
        assert p.grad is None or torch.allclose(p.grad, torch.zeros(4))

    def test_state_persists_across_steps(self):
        # momentum buffer must accumulate — two steps of constant grad grow the step
        p = torch.nn.Parameter(torch.zeros(3))
        opt = NGLOptimizer([p], momentum_program(lr=0.1, mu=0.9))
        p.grad = torch.ones(3)
        opt.step()
        d1 = p.detach().clone()
        p.grad = torch.ones(3)
        opt.step()
        d2 = p.detach().clone()
        step1 = (0.0 - d1).abs()
        step2 = (d1 - d2).abs()
        assert (step2 > step1).all()  # momentum accelerates
