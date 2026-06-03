# -*- coding: utf-8 -*-
"""TDD spec for Intervention F — Frequency-balanced cross-entropy.

Motivation
──────────
The PR2 interventions are all defensive (don't memorise, don't
collapse, don't anisotropy). None of them push the model toward
the OOD token distribution. Combined with 80% chat data + dead
bio-circuitry, the LM gradient direction tiles capacity over
chat surface forms and gap_ratio widens.

freq_balance is the cheapest possible *direction-aware* signal:
precompute unigram frequencies on both train and target (OOD) corpus,
build a per-vocab weight vector
    w[v] = clip( (freq_target[v] / freq_train[v])^beta, w_min, w_max )
and multiply per-token CE by w[targets]. Tokens over-represented in
chat get downweighted; tokens unique to prose get upweighted.

Math reference: architectures/rcc_bowtie/lib/regularizers.neuro
::freq_balance_weight / freq_balance_normalize / freq_balance_loss.
"""
import pytest
import torch

from neuroslm.dsl.regularization import (
    FreqBalanceConfig, parse_regularization_block)
from neuroslm.regularizers import FreqBalanceReweighter


# ── Config + parser ─────────────────────────────────────────────────

class TestFreqBalanceConfig:
    def test_disabled_by_default(self):
        cfg = FreqBalanceConfig()
        assert cfg.enabled is False

    def test_default_beta_is_sqrt(self):
        cfg = FreqBalanceConfig()
        assert cfg.beta == pytest.approx(0.5)

    def test_default_clip_range(self):
        cfg = FreqBalanceConfig()
        assert cfg.w_min == pytest.approx(0.2)
        assert cfg.w_max == pytest.approx(5.0)


class TestFreqBalanceParser:
    def test_parser_accepts_block(self):
        body = """freq_balance: { enabled: true, beta: 0.75,
                                   w_min: 0.1, w_max: 10.0 }"""
        cfg = parse_regularization_block(body)
        assert cfg.freq_balance.enabled is True
        assert cfg.freq_balance.beta == pytest.approx(0.75)
        assert cfg.freq_balance.w_min == pytest.approx(0.1)
        assert cfg.freq_balance.w_max == pytest.approx(10.0)

    def test_omitted_block_means_disabled(self):
        body = """dar: { enabled: true }"""
        cfg = parse_regularization_block(body)
        assert cfg.freq_balance.enabled is False


# ── Module behaviour ────────────────────────────────────────────────

class TestFreqBalanceReweighter:
    def _make(self, vocab=8, beta=0.5, w_min=0.2, w_max=5.0,
              enabled=True):
        cfg = FreqBalanceConfig(
            enabled=enabled, beta=beta, w_min=w_min, w_max=w_max)
        return FreqBalanceReweighter(cfg, vocab_size=vocab)

    def test_disabled_returns_identity(self):
        r = self._make(enabled=False)
        per_tok = torch.tensor([1.0, 2.0, 3.0])
        targets = torch.tensor([0, 1, 2])
        out = r(per_tok, targets)
        # Equal to plain mean
        assert float(out) == pytest.approx(per_tok.mean().item())

    def test_uniform_frequencies_act_as_identity(self):
        """If train and target distributions match, weights are 1."""
        r = self._make(vocab=4, beta=1.0)
        r.set_frequencies(
            freq_train=torch.tensor([0.25, 0.25, 0.25, 0.25]),
            freq_target=torch.tensor([0.25, 0.25, 0.25, 0.25]),
        )
        per_tok = torch.tensor([1.0, 2.0, 3.0, 4.0])
        targets = torch.tensor([0, 1, 2, 3])
        out = r(per_tok, targets)
        # Mean-normalised weights all == 1
        assert float(out) == pytest.approx(per_tok.mean().item(), rel=1e-5)

    def test_overrepresented_train_token_is_downweighted(self):
        """A token over-represented in train (vs target) gets w < 1."""
        r = self._make(vocab=2, beta=1.0, w_min=0.01, w_max=100.0)
        r.set_frequencies(
            freq_train=torch.tensor([0.9, 0.1]),    # token 0 dominates train
            freq_target=torch.tensor([0.1, 0.9]),   # token 0 rare in target
        )
        w = r.weight_vector()
        # Before normalisation: ratio[0] = 0.1/0.9, ratio[1] = 0.9/0.1
        # After mean-norm, w_norm[0] < w_norm[1] < 1 doesn't hold but
        # w_norm[0] << w_norm[1] must hold.
        assert w[0] < w[1]
        assert w[0] < 1.0
        assert w[1] > 1.0

    def test_weights_clip_to_bounds(self):
        r = self._make(vocab=2, beta=2.0, w_min=0.5, w_max=2.0)
        r.set_frequencies(
            freq_train=torch.tensor([0.99, 0.01]),
            freq_target=torch.tensor([0.01, 0.99]),
        )
        w_raw = r._raw_weight_vector()  # pre-normalisation, clipped
        assert w_raw.min() >= 0.5 - 1e-6
        assert w_raw.max() <= 2.0 + 1e-6

    def test_mean_normalised(self):
        """w_norm has mean 1.0 → loss scale unchanged on average."""
        r = self._make(vocab=4, beta=0.5)
        r.set_frequencies(
            freq_train=torch.tensor([0.40, 0.30, 0.20, 0.10]),
            freq_target=torch.tensor([0.10, 0.20, 0.30, 0.40]),
        )
        w = r.weight_vector()
        assert float(w.mean()) == pytest.approx(1.0, rel=1e-5)

    def test_loss_lower_when_overrepresented_token_dominates_errors(self):
        """If all per-token CE concentrates on the over-represented
        token, freq_balance lowers the mean loss (downweighted)."""
        r = self._make(vocab=2, beta=1.0)
        r.set_frequencies(
            freq_train=torch.tensor([0.9, 0.1]),
            freq_target=torch.tensor([0.1, 0.9]),
        )
        per_tok = torch.tensor([2.0, 2.0, 2.0, 2.0])
        targets_all_0 = torch.tensor([0, 0, 0, 0])
        targets_all_1 = torch.tensor([1, 1, 1, 1])
        L_0 = float(r(per_tok, targets_all_0))
        L_1 = float(r(per_tok, targets_all_1))
        assert L_0 < L_1  # token 0 downweighted, contributes less

    def test_gradient_flows_through(self):
        """The weighted loss must remain a leaf in the autograd graph."""
        r = self._make(vocab=4, beta=0.5)
        r.set_frequencies(
            freq_train=torch.tensor([0.4, 0.3, 0.2, 0.1]),
            freq_target=torch.tensor([0.1, 0.2, 0.3, 0.4]),
        )
        per_tok = torch.tensor([1.0, 2.0, 3.0, 4.0], requires_grad=True)
        targets = torch.tensor([0, 1, 2, 3])
        out = r(per_tok, targets)
        out.backward()
        assert per_tok.grad is not None
        assert per_tok.grad.abs().sum() > 0

    def test_set_frequencies_validates_shape(self):
        r = self._make(vocab=4)
        with pytest.raises(ValueError, match="vocab"):
            r.set_frequencies(
                freq_train=torch.ones(3) / 3,
                freq_target=torch.ones(3) / 3,
            )

    def test_no_op_until_frequencies_set(self):
        """Until set_frequencies() is called, the reweighter returns
        the plain mean (it can't reweight without statistics)."""
        r = self._make(vocab=4)
        per_tok = torch.tensor([1.0, 2.0, 3.0, 4.0])
        targets = torch.tensor([0, 1, 2, 3])
        out = r(per_tok, targets)
        assert float(out) == pytest.approx(per_tok.mean().item())
