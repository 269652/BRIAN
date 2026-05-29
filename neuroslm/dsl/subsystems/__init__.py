# -*- coding: utf-8 -*-
"""DSL ports of Brain's aux-loss-producing subsystems.

Every subsystem in this package is bit-identical to its `neuroslm.modules.*`
counterpart on the same input/weights. The parity tests (`tests/dsl/test_*_parity.py`)
assert `torch.allclose` at atol 1e-6 on the forward AND on the parameter
gradient — guarding against the silent forward-only divergence pattern
(see `feedback_forward_vs_gradient_parity` memory).

Modules:
    motor   — MotorCortex (action head + thought projection + lang bias)
    forward — ForwardModel (next-state predictor, regularizer)
    world   — RecurrentStateSpaceModel (RSSM prior/posterior + KL)
    orch    — NeuralOrchestrator (cerebellum + entorhinal + claustrum)
"""
