"""Lateral Habenula (LHb) — Anti-Reward Signal for NeuroSLM.

The lateral habenula is a small epithalamic nucleus often called the
"anti-reward center" of the brain. While the VTA fires for unexpected
rewards (positive RPE → DA spike), the LHb fires for:
  - Unexpected omission of reward (expected reward doesn't arrive)
  - Punishment prediction
  - Aversive outcomes

The LHb drives a *DA dip*: it inhibits VTA dopamine neurons, causing DA
levels to drop below baseline. This is the neural substrate of:
  - Disappointment: expected good thing didn't happen
  - Aversion learning: avoid stimuli that predict bad outcomes
  - Learned helplessness: chronic LHb hyperactivity → depression-like states
  - Motivational suppression: LHb activation reduces motivation to try

Without the LHb, the reward system is incomplete:
  VTA alone: learns "do more of what worked" (positive learning)
  VTA + LHb: learns "do more of what worked AND do less of what failed"

The LHb also interacts with the raphe nuclei (5HT) and drives 5HT dips
when expectations of positive outcomes are violated.

For NeuroSLM:
  - The LHb fires when the model expected a reward (high DA) but didn't
    get it — when lm_loss is high despite model confidence
  - It suppresses DA (via inhibitory projection) creating the DA dip
  - This drives stronger learning on missed predictions
  - Chronic LHb activation (repeated failures) triggers a fatigue/
    helplessness state that reduces overall drive

Implementation:
  Tracks rolling expected reward vs. actual reward.
  When expected > actual by threshold → fire, suppress DA + 5HT.
  Adapts threshold over time (opponent process adaptation).

References:
  Matsumoto & Hikosaka (2007): Lateral habenula as a source of negative reward
  Hikosaka (2010): The habenula: from stress evasion to value-based decisions
  Winter et al. (2011): The lateral habenular pathway in depression
"""
from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


class LateralHabenula(nn.Module):
    """Anti-reward center: fires when expected rewards are omitted.

    Parameters
    ----------
    n_nt           : number of neurotransmitters
    decay          : EMA decay for rolling expectations
    baseline_window: steps for computing baseline expectation
    activation_thr : reward-expectation gap that triggers LHb firing
    """

    def __init__(self, n_nt: int = 8,
                 decay: float = 0.95,
                 activation_thr: float = 0.15):
        super().__init__()
        self.n_nt  = n_nt
        self.decay = decay
        self.activation_thr = activation_thr

        # Running estimate of expected reward (smoothed)
        self.register_buffer("expected_reward", torch.tensor(0.5))
        # Running actual reward
        self.register_buffer("actual_reward",   torch.tensor(0.5))
        # Chronic frustration accumulator (LHb sensitization)
        self.register_buffer("frustration",     torch.tensor(0.0))

        # LHb firing → NT suppression pattern
        # DA and 5HT suppressed; NE slightly elevated (stress)
        nt_suppression = torch.zeros(n_nt)
        # Indices for DA=0, NE=1, 5HT=2, ACh=3 (conventional ordering)
        # Suppress DA and 5HT; mildly elevate NE
        nt_suppression[0] = -0.8   # DA suppression
        nt_suppression[2] = -0.4   # 5HT suppression
        nt_suppression[1] =  0.2   # NE mild elevation (stress)
        self.register_buffer("nt_suppression_pattern", nt_suppression)

        # Learned suppression gain (how strongly LHb affects each NT)
        self.suppression_gain = nn.Parameter(torch.ones(n_nt) * 0.5)

        # Opponent process: adaptation to chronic LHb activity
        # After sustained LHb firing, its effect diminishes (adaptation)
        self.register_buffer("adaptation", torch.tensor(0.0))

    # ------------------------------------------------------------------

    def update(self, actual_reward: torch.Tensor,
               da_level: Optional[torch.Tensor] = None) -> dict:
        """Update LHb state and compute NT modulation.

        actual_reward: (B,) or scalar — actual reward received this step
        da_level:      (B,) or scalar — current DA level (proxy for expected reward)

        Returns dict:
          lhb_firing:   scalar — how strongly LHb is firing [0, 1]
          nt_delta:     (n_nt,) — NT change to apply (negative = suppression)
          frustration:  scalar — chronic frustration level [0, 1]
        """
        actual = actual_reward.detach().mean() if torch.is_tensor(actual_reward) \
                 else torch.tensor(float(actual_reward))
        device = actual.device

        # Expected reward: use DA level as proxy (DA encodes expected value)
        if da_level is not None:
            expected = da_level.detach().mean().clamp(0, 1)
        else:
            expected = self.expected_reward.to(device)

        # Update rolling expected reward
        with torch.no_grad():
            self.expected_reward = (self.decay * self.expected_reward.to(device)
                                    + (1 - self.decay) * expected)
            self.actual_reward   = (self.decay * self.actual_reward.to(device)
                                    + (1 - self.decay) * actual)

        # Reward prediction error: positive RPE → VTA (handled elsewhere)
        # Negative RPE → LHb fires
        rpe = self.expected_reward.to(device) - self.actual_reward.to(device)  # positive when expected > actual

        # LHb fires on negative RPE (missed reward) above threshold
        lhb_raw = F.relu(rpe - self.activation_thr)   # only fire when gap > threshold
        lhb_firing = lhb_raw.clamp(0, 1)

        # Opponent process adaptation: chronic activation → reduced effect
        with torch.no_grad():
            self.adaptation = (0.99 * self.adaptation.to(device)
                               + 0.01 * lhb_firing)
            self.frustration = (0.999 * self.frustration.to(device)
                                + 0.001 * lhb_firing)

        # Effective firing after adaptation
        effective_firing = lhb_firing * (1.0 - 0.5 * self.adaptation.to(device))

        # NT modulation
        pattern = self.nt_suppression_pattern.to(device)
        gain    = torch.sigmoid(self.suppression_gain.to(device))
        nt_delta = effective_firing * pattern * gain   # (n_nt,)

        return {
            "lhb_firing":  effective_firing,
            "nt_delta":    nt_delta,
            "frustration": self.frustration.to(device),
            "rpe_neg":     rpe.clamp(0),  # magnitude of negative RPE
        }
