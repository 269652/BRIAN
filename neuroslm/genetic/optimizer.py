# -*- coding: utf-8 -*-
"""SOTA optimizers expressed as NGL programs, plus a torch optimizer adapter.

An NGL **update rule** obeys this calling convention, per step and per param:

    t0 <- grad g          (written by the adapter)
    t1 <- param p         (written by the adapter)
    <state registers>     persist in the param's Memory across steps
    out_reg               the *update* to apply:  p <- p + out_reg

So SGD's program writes ``-lr*g`` into the out register and the adapter adds it.
State (momentum/velocity/step-count) lives in registers that the program both
reads and overwrites, which is exactly why a straight-line DAG can't do this and
NGL can.

The encoders below are the proof that NGL spans the update-rule grammar: each
reproduces its reference optimizer bit-for-bit (see tests/genetic/test_optimizer).
They are also the *seeds* the discovery GA mutates from.
"""
from __future__ import annotations

import math
from typing import Iterable, List

import torch

from neuroslm.genetic.language import Instruction, Memory, Program

# Register map (shared convention)
G = "t0"       # grad in
P = "t1"       # param in
# t2.. and s.. are free for state / scratch


# ---------------------------------------------------------------------------
# Optimizer encoders.
# ---------------------------------------------------------------------------
def sgd_program(lr: float = 0.01) -> Program:
    """p <- p - lr*g."""
    return Program(
        [Instruction("cscale", "t5", (G,), const=-lr)],
        n_scalar=8, n_tensor=16, out_reg="t5", meta={"name": "sgd", "lr": lr},
    )


def momentum_program(lr: float = 0.01, mu: float = 0.9) -> Program:
    """torch SGD+momentum: buf = mu*buf + g ; p <- p - lr*buf.

    First step reads buf=0 → buf=g, matching torch's ``buf = grad.clone()``.
    """
    buf = "t2"
    return Program(
        [
            Instruction("cscale", "t6", (buf,), const=mu),   # mu*buf
            Instruction("add", buf, ("t6", G)),               # buf = mu*buf + g
            Instruction("cscale", "t7", (buf,), const=-lr),   # -lr*buf
        ],
        n_scalar=8, n_tensor=16, out_reg="t7",
        meta={"name": "momentum", "lr": lr, "mu": mu},
    )


def rmsprop_program(lr: float = 0.01, alpha: float = 0.99, eps: float = 1e-8) -> Program:
    """v = alpha*v + (1-alpha)*g^2 ; p <- p - lr*g/(sqrt(v)+eps)."""
    v = "t2"
    return Program(
        [
            Instruction("square", "t6", (G,)),                    # g^2
            Instruction("cscale", "t7", ("t6",), const=1 - alpha),
            Instruction("cscale", "t8", (v,), const=alpha),
            Instruction("add", v, ("t8", "t7")),                  # v
            Instruction("sqrt", "t9", (v,)),                      # sqrt(v)
            Instruction("const", "s0", (), const=eps),
            Instruction("add", "t10", ("t9", "s0")),              # sqrt(v)+eps
            Instruction("div", "t11", (G, "t10")),                # g/(sqrt(v)+eps)
            Instruction("cscale", "t12", ("t11",), const=-lr),
        ],
        n_scalar=8, n_tensor=16, out_reg="t12",
        meta={"name": "rmsprop", "lr": lr, "alpha": alpha, "eps": eps},
    )


def adam_program(lr: float = 0.001, b1: float = 0.9, b2: float = 0.999,
                 eps: float = 1e-8) -> Program:
    """torch Adam with bias correction.

    m = b1*m + (1-b1)*g ;  v = b2*v + (1-b2)*g^2
    t <- t+1 ;  bc1 = 1-b1^t ;  bc2 = 1-b2^t
    denom = sqrt(v)/sqrt(bc2) + eps ;  p <- p - (lr/bc1)*m/denom

    b1^t / b2^t computed as exp(t*log(b)) (state-free bias correction).
    """
    m, v, tcnt = "t2", "t3", "s0"
    logb1, logb2 = math.log(b1), math.log(b2)
    return Program(
        [
            # m, v updates
            Instruction("cscale", "t6", (m,), const=b1),
            Instruction("cscale", "t7", (G,), const=1 - b1),
            Instruction("add", m, ("t6", "t7")),
            Instruction("square", "t8", (G,)),
            Instruction("cscale", "t9", (v,), const=b2),
            Instruction("cscale", "t10", ("t8",), const=1 - b2),
            Instruction("add", v, ("t9", "t10")),
            # step counter t <- t + 1
            Instruction("const", "s1", (), const=1.0),
            Instruction("add", tcnt, (tcnt, "s1")),
            # bc1 = 1 - exp(t*logb1)
            Instruction("cscale", "s2", (tcnt,), const=logb1),
            Instruction("exp", "s3", ("s2",)),                  # b1^t
            Instruction("const", "s4", (), const=1.0),
            Instruction("sub", "s5", ("s4", "s3")),             # bc1
            # bc2 = 1 - exp(t*logb2)
            Instruction("cscale", "s6", (tcnt,), const=logb2),
            Instruction("exp", "s7", ("s6",)),
            Instruction("sub", "s8", ("s4", "s7")),             # bc2  (s8)
            # denom = sqrt(v)/sqrt(bc2) + eps
            Instruction("sqrt", "t11", (v,)),                   # sqrt(v)
            Instruction("sqrt", "s9", ("s8",)),                 # sqrt(bc2)
            Instruction("div", "t12", ("t11", "s9")),
            Instruction("const", "s10", (), const=eps),
            Instruction("add", "t13", ("t12", "s10")),          # denom
            # step_size = lr/bc1 ; update = -step_size * m/denom
            Instruction("div", "t14", (m, "t13")),              # m/denom
            Instruction("const", "s11", (), const=lr),
            Instruction("div", "s12", ("s11", "s5")),           # lr/bc1
            Instruction("neg", "s13", ("s12",)),
            Instruction("cscale", "t15", ("t14",), const=1.0),  # copy m/denom
            Instruction("mul", "t15", ("t15", "s13")),          # * (-lr/bc1)
        ],
        n_scalar=16, n_tensor=16, out_reg="t15",
        meta={"name": "adam", "lr": lr, "b1": b1, "b2": b2, "eps": eps},
    )


def lion_program(lr: float = 0.01, b1: float = 0.9, b2: float = 0.99) -> Program:
    """Lion (Chen et al. 2023) — the optimizer *discovered* by program search.

    c = b1*m + (1-b1)*g ;  p <- p - lr*sign(c) ;  m = b2*m + (1-b2)*g
    m is updated last so ``c`` uses the pre-update buffer.
    """
    m = "t2"
    return Program(
        [
            Instruction("cscale", "t6", (m,), const=b1),
            Instruction("cscale", "t7", (G,), const=1 - b1),
            Instruction("add", "t8", ("t6", "t7")),      # c
            Instruction("sign", "t9", ("t8",)),
            Instruction("cscale", "t10", ("t9",), const=-lr),   # update
            Instruction("cscale", "t11", (m,), const=b2),
            Instruction("cscale", "t12", (G,), const=1 - b2),
            Instruction("add", m, ("t11", "t12")),       # m updated last
        ],
        n_scalar=8, n_tensor=16, out_reg="t10",
        meta={"name": "lion", "lr": lr, "b1": b1, "b2": b2},
    )


SEED_OPTIMIZERS = {
    "sgd": sgd_program,
    "momentum": momentum_program,
    "rmsprop": rmsprop_program,
    "adam": adam_program,
    "lion": lion_program,
}


# ---------------------------------------------------------------------------
# Adapter: run an NGL update-rule program as a torch optimizer.
# ---------------------------------------------------------------------------
class NGLOptimizer:
    """Minimal torch-style optimizer driven by an NGL ``Program``.

    One persistent ``Memory`` per parameter carries the program's state
    registers across ``step`` calls.
    """

    def __init__(self, params: Iterable[torch.nn.Parameter], program: Program):
        self.params: List[torch.nn.Parameter] = [p for p in params]
        self.program = program
        self._mem = {id(p): Memory(program.n_scalar, program.n_tensor) for p in self.params}

    def zero_grad(self, set_to_none: bool = True):
        for p in self.params:
            if p.grad is not None:
                if set_to_none:
                    p.grad = None
                else:
                    p.grad.zero_()

    @torch.no_grad()
    def step(self, closure=None):
        loss = closure() if closure is not None else None
        for p in self.params:
            if p.grad is None:
                continue
            mem = self._mem[id(p)]
            mem.write(G, p.grad.detach())
            mem.write(P, p.data)
            self.program.execute(mem)
            update = mem.read(self.program.out_reg)
            if update.shape != p.data.shape:
                # totality: reshape/broadcast the update onto the param
                try:
                    update = update.reshape(p.data.shape)
                except RuntimeError:
                    update = update.mean().expand_as(p.data)
            p.data.add_(update)
        return loss
