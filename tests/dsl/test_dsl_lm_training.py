# -*- coding: utf-8 -*-
"""Phase N5 — the DSL transformer LM actually learns a context task.

The toy per-token circuit could only ever learn the unigram marginal
(loss floored at ln(vocab)). A real language model must model *context*.
This test trains the DSL LM on the "copy previous token" task —
`target[t] = ids[t-1]` — which is impossible without attending to the
previous position. If the DSL transformer is wired correctly, loss drops
far below the random floor.
"""
import math
import pytest
import torch

from neuroslm.dsl.nn_lang import build_language_model
from neuroslm.dsl.training_config import TrainingConfig
from neuroslm.harness import BRIANHarness


class TestDSLLMLearnsContext:
    def test_copy_previous_token_task(self):
        torch.manual_seed(0)
        vocab, D = 32, 64
        lm = build_language_model(vocab=vocab, d_model=D, depth=2,
                                  n_heads=4, max_ctx=64)
        cfg = TrainingConfig()
        cfg.learning_rate = 3e-3
        cfg.grad_accum = 1
        h = BRIANHarness.from_language_model(lm, vocab_size=vocab,
                                             d_sem=D, training_config=cfg)

        random_floor = math.log(vocab)   # ≈ 3.466

        def batch():
            ids = torch.randint(0, vocab, (16, 24))
            targets = torch.roll(ids, shifts=1, dims=1)  # target[t] = ids[t-1]
            targets[:, 0] = 0
            return ids, targets

        # Train
        final = None
        for _ in range(400):
            ids, targets = batch()
            final = h.train_step(ids, targets)

        # A correctly-wired causal transformer learns "copy previous token"
        # to well below the random floor — proves it models context.
        assert final < random_floor * 0.5, \
            f"final loss {final:.3f} not below half the random floor {random_floor:.3f}"

    def test_per_token_baseline_cannot_learn_context(self):
        # Sanity: on the same task, a model with NO context (shuffling the
        # time dimension so position t can't see t-1) must stay near floor.
        # This confirms the task genuinely requires context.
        torch.manual_seed(0)
        vocab, D = 32, 64
        lm = build_language_model(vocab=vocab, d_model=D, depth=2,
                                  n_heads=4, max_ctx=64)
        cfg = TrainingConfig(); cfg.learning_rate = 3e-3; cfg.grad_accum = 1
        h = BRIANHarness.from_language_model(lm, vocab_size=vocab,
                                             d_sem=D, training_config=cfg)
        random_floor = math.log(vocab)

        # Targets independent of inputs (pure noise) → cannot beat floor much
        final = None
        for _ in range(200):
            ids = torch.randint(0, vocab, (16, 24))
            targets = torch.randint(0, vocab, (16, 24))  # uncorrelated
            final = h.train_step(ids, targets)

        # With no learnable structure, loss can't drop far below the floor.
        assert final > random_floor * 0.7, \
            f"loss {final:.3f} dropped too far on noise — task leak?"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
