"""Full mesolimbic reward circuit.

Models the VTA → NAcc → mPFC loop with:
  - Reward Prediction Error (RPE) — TD-like signal
  - Incentive salience ("wanting" vs "liking")
  - Reward-tagged memory consolidation signals
  - Opponent process (hedonic adaptation)
  - CB1-gated disinhibition of DA release
  - D2 autoreceptor negative feedback on VTA

Neuroscience basis:
  - VTA DA neurons fire to unexpected rewards (positive RPE)
  - VTA DA neurons pause to unexpected omissions (negative RPE)
  - NAcc shell: hedonic "liking" (opioid/eCB hotspots)
  - NAcc core: incentive "wanting" (DA-driven motivation)
  - mPFC receives DA projections for working memory gating
  - D2 short autoreceptors on VTA terminals limit DA release
  - CB1 on GABA interneurons in VTA → disinhibits DA neurons

This circuit produces signals used by:
  - Memory system: tag episodes with reward/RPE for preferential consolidation
  - Genome system: fitness signal for evolutionary selection
  - Training: gradient scaling based on learning_gain
  - Attention: salient stimuli get DA-driven priority
"""
from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F


class MesolimbicCircuit(nn.Module):
    """Integrates reward signals into DA release, RPE computation,
    incentive salience, and memory consolidation tags."""

    def __init__(self, d_state: int = 32):
        super().__init__()
        self.d_state = d_state

        # --- Value estimator (critic) for RPE ---
        # Predicts expected value from current state representation
        self.value_net = nn.Sequential(
            nn.Linear(d_state, 32), nn.GELU(),
            nn.Linear(32, 16), nn.GELU(),
            nn.Linear(16, 1),
        )

        # --- Incentive salience network ---
        # "Wanting" signal: DA-modulated motivational drive
        # Inputs: [state_summary, da_level, novelty, uncertainty]
        self.wanting_net = nn.Sequential(
            nn.Linear(d_state + 3, 32), nn.GELU(),
            nn.Linear(32, 1), nn.Sigmoid(),
        )

        # --- Hedonic "liking" network ---
        # Opioid/eCB-mediated pleasure signal (less DA-dependent)
        # Inputs: [state_summary, ecb_level, reward]
        self.liking_net = nn.Sequential(
            nn.Linear(d_state + 2, 16), nn.GELU(),
            nn.Linear(16, 1), nn.Sigmoid(),
        )

        # --- D2 autoreceptor feedback ---
        # Models how D2 receptors on VTA terminals limit further DA release
        self.d2_gate = nn.Sequential(
            nn.Linear(2, 8), nn.GELU(),
            nn.Linear(8, 1), nn.Sigmoid(),
        )

        # --- CB1-gated disinhibition ---
        # CB1 on VTA GABA interneurons: high eCB → less GABA → more DA
        self.cb1_disinhibition = nn.Sequential(
            nn.Linear(2, 8), nn.GELU(),
            nn.Linear(8, 1), nn.Sigmoid(),
        )

        # --- Memory consolidation tagger ---
        # Decides how strongly to tag a memory for consolidation
        # Inputs: [rpe, salience, novelty, emotional_valence]
        self.consolidation_net = nn.Sequential(
            nn.Linear(4, 16), nn.GELU(),
            nn.Linear(16, 1), nn.Sigmoid(),
        )

        # --- Opponent process (hedonic adaptation) ---
        # After strong positive reward, a negative aftereffect builds up
        self.register_buffer("hedonic_baseline", torch.tensor(0.0))
        self.register_buffer("opponent_state", torch.tensor(0.0))
        self.register_buffer("ema_reward", torch.tensor(0.0))
        self.register_buffer("ema_rpe", torch.tensor(0.0))

        # RPE history for temporal difference
        self.register_buffer("prev_value", torch.tensor(0.0))

    def compute_rpe(self, state_vec: torch.Tensor, reward: torch.Tensor,
                    gamma: float = 0.95) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute temporal-difference RPE.
        state_vec: (B, d_state), reward: (B,)
        Returns: (rpe, current_value) both (B,)"""
        current_value = self.value_net(state_vec).squeeze(-1)  # (B,)
        # TD error: r + γV(s') - V(s)
        # We use current_value as V(s') and prev_value as V(s)
        prev_v = self.prev_value.expand_as(current_value)
        rpe = reward + gamma * current_value.detach() - prev_v
        # Update for next step
        self.prev_value = current_value.detach().mean()
        return rpe, current_value

    def compute_wanting(self, state_vec: torch.Tensor, da_level: torch.Tensor,
                        novelty: torch.Tensor, uncertainty: torch.Tensor) -> torch.Tensor:
        """Incentive salience: DA-modulated motivational drive. (B,)"""
        x = torch.cat([state_vec, da_level.unsqueeze(-1),
                       novelty.unsqueeze(-1), uncertainty.unsqueeze(-1)], dim=-1)
        return self.wanting_net(x).squeeze(-1)

    def compute_liking(self, state_vec: torch.Tensor, ecb_level: torch.Tensor,
                       reward: torch.Tensor) -> torch.Tensor:
        """Hedonic 'liking' signal — less DA-dependent. (B,)"""
        x = torch.cat([state_vec, ecb_level.unsqueeze(-1),
                       reward.unsqueeze(-1)], dim=-1)
        return self.liking_net(x).squeeze(-1)

    def d2_feedback(self, da_level: torch.Tensor, recent_release: torch.Tensor) -> torch.Tensor:
        """D2 autoreceptor: limits DA release when DA is already high.
        Returns a gating factor in [0, 1] that multiplies DA release demand."""
        x = torch.stack([da_level, recent_release], dim=-1)
        # Low gate = strong inhibition (high DA → don't release more)
        return self.d2_gate(x).squeeze(-1)

    def cb1_gate(self, ecb_level: torch.Tensor, gaba_level: torch.Tensor) -> torch.Tensor:
        """CB1-mediated disinhibition: high eCB suppresses GABA interneurons
        in VTA, allowing more DA release. Returns boost factor."""
        x = torch.stack([ecb_level, gaba_level], dim=-1)
        return self.cb1_disinhibition(x).squeeze(-1)

    def consolidation_strength(self, rpe: torch.Tensor, salience: torch.Tensor,
                               novelty: torch.Tensor, valence: torch.Tensor) -> torch.Tensor:
        """How strongly to tag current memory for consolidation. (B,) in [0,1]."""
        x = torch.stack([rpe, salience, novelty, valence], dim=-1)
        return self.consolidation_net(x).squeeze(-1)

    @torch.no_grad()
    def update_opponent_process(self, reward: torch.Tensor):
        """Opponent process: strong rewards build up a negative aftereffect.
        Models hedonic treadmill / tolerance to pleasure."""
        r_mean = reward.detach().mean()
        alpha = 0.01
        self.ema_reward = (1 - alpha) * self.ema_reward + alpha * r_mean
        # Opponent grows when reward is above baseline
        excess = (r_mean - self.hedonic_baseline).clamp(min=0.0)
        self.opponent_state = 0.95 * self.opponent_state + 0.05 * excess
        # Slowly shift baseline toward average reward (hedonic adaptation)
        self.hedonic_baseline = 0.999 * self.hedonic_baseline + 0.001 * self.ema_reward

    def effective_reward(self, raw_reward: torch.Tensor) -> torch.Tensor:
        """Reward after opponent process subtraction."""
        return raw_reward - self.opponent_state.to(raw_reward.device)

    def forward(self, state_vec: torch.Tensor, reward: torch.Tensor,
                da_level: torch.Tensor, ecb_level: torch.Tensor,
                gaba_level: torch.Tensor, novelty: torch.Tensor,
                salience: torch.Tensor, valence: torch.Tensor,
                uncertainty: torch.Tensor) -> dict:
        """Full mesolimbic computation for one tick.

        Args:
            state_vec: (B, d_state) compressed state from GWS/world model
            reward: (B,) external or intrinsic reward
            da_level: (B,) current DA level
            ecb_level: (B,) current eCB level
            gaba_level: (B,) current GABA level
            novelty: (B,) novelty signal
            salience: (B,) salience signal
            valence: (B,) emotional valence
            uncertainty: (B,) epistemic uncertainty

        Returns dict with:
            rpe: reward prediction error (B,)
            da_release_demand: how much DA to release (B,) — after D2/CB1 gating
            wanting: incentive salience (B,)
            liking: hedonic signal (B,)
            consolidation: memory tag strength (B,)
            learning_gain: gradient scaling factor (B,)
            value: estimated state value (B,)
        """
        # Opponent-adjusted reward
        eff_reward = self.effective_reward(reward)

        # RPE computation
        rpe, value = self.compute_rpe(state_vec, eff_reward)

        # DA release demand: positive RPE → release, modulated by D2 and CB1
        base_da_demand = torch.sigmoid(rpe)  # [0, 1]
        d2_factor = self.d2_feedback(da_level, base_da_demand)
        cb1_factor = self.cb1_gate(ecb_level, gaba_level)
        # D2 suppresses, CB1 boosts (disinhibition)
        da_release_demand = base_da_demand * d2_factor * (1.0 + 0.5 * cb1_factor)
        da_release_demand = da_release_demand.clamp(0.0, 1.0)

        # Incentive salience (wanting)
        wanting = self.compute_wanting(state_vec, da_level, novelty, uncertainty)

        # Hedonic signal (liking)
        liking = self.compute_liking(state_vec, ecb_level, eff_reward)

        # Memory consolidation strength
        consolidation = self.consolidation_strength(
            rpe.detach(), salience, novelty, valence)

        # Learning gain: scale gradients by mesolimbic signal
        # High RPE + high wanting → learn more from this experience
        learning_gain = torch.sigmoid(rpe.detach() + wanting.detach())

        # Update opponent process
        self.update_opponent_process(reward)
        # Update RPE EMA
        self.ema_rpe = 0.95 * self.ema_rpe + 0.05 * rpe.detach().mean()

        return {
            "rpe": rpe,
            "da_release_demand": da_release_demand,
            "wanting": wanting,
            "liking": liking,
            "consolidation": consolidation,
            "learning_gain": learning_gain,
            "value": value,
            "effective_reward": eff_reward,
            "opponent_state": self.opponent_state.clone(),
        }

    def info(self) -> dict:
        return {
            "hedonic_baseline": float(self.hedonic_baseline),
            "opponent_state": float(self.opponent_state),
            "ema_reward": float(self.ema_reward),
            "ema_rpe": float(self.ema_rpe),
            "prev_value": float(self.prev_value),
        }
