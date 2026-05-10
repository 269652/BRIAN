"""Theory of Mind module — modelling other agents' mental states.

Implements level-2 mentalizing: the model maintains separate mental
state representations for each tracked entity and can ask:
  • "What does entity X believe right now?"
  • "What does entity X want?"
  • "What will entity X likely do next?"
  • "What would entity X do if I said Y?"  (counterfactual)

Architecture:
  EntityEncoder   — projects entity profile (style + belief) → entity context
  MentalStateHead — cross-attention over entity context + current GWS state
  BeliefDecoder   — entity's current belief-state embedding
  DesireDecoder   — entity's likely goal / motivation embedding
  IntentionHead   — predicts entity's probable next action type distribution
  CounterfactualHead — simulates "if I do X, entity will respond Y"
  SocialRewardHead — correct predictions → intrinsic social reward signal

Novel aspects vs. any current SLM:
  1. Online entity-belief updating from observed responses (social prediction error)
  2. Counterfactual social simulation via a small forward model over entity states
  3. Emotional contagion: entity's predicted affect influences own NT state
  4. Social reward (DA) when prediction matches reality → social learning signal
  5. All within a single BrainModule that can be enabled/disabled independently
"""
from __future__ import annotations
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Dict, Tuple
import numpy as np

from .brain_module import BrainModule


class EntityEncoder(nn.Module):
    """Projects (style_emb + belief_vec + interaction_count) → entity context."""

    def __init__(self, d_style: int, d_belief: int, d_sem: int):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(d_style + d_belief + 1, d_sem),
            nn.LayerNorm(d_sem),
            nn.GELU(),
            nn.Linear(d_sem, d_sem),
        )

    def forward(self, style: torch.Tensor, belief: torch.Tensor,
                log_interactions: torch.Tensor) -> torch.Tensor:
        """
        style:            (B, d_style)
        belief:           (B, d_belief)
        log_interactions: (B, 1)
        → entity_ctx:     (B, d_sem)
        """
        x = torch.cat([style, belief, log_interactions], dim=-1)
        return self.proj(x)


class MentalStateDecoder(nn.Module):
    """Cross-attention: entity_ctx × current_situation → mental states."""

    def __init__(self, d_sem: int, n_heads: int = 4, n_layers: int = 2):
        super().__init__()
        self.layers = nn.ModuleList([
            nn.ModuleDict({
                "cross_attn": nn.MultiheadAttention(d_sem, n_heads, batch_first=True),
                "ff":  nn.Sequential(
                    nn.Linear(d_sem, d_sem * 2), nn.GELU(),
                    nn.Linear(d_sem * 2, d_sem)),
                "ln1": nn.LayerNorm(d_sem),
                "ln2": nn.LayerNorm(d_sem),
            })
            for _ in range(n_layers)
        ])
        # Belief, desire, intention projection heads
        self.belief_head    = nn.Linear(d_sem, d_sem)
        self.desire_head    = nn.Linear(d_sem, d_sem)
        self.intention_head = nn.Linear(d_sem, 14)   # 14 social action types

    def forward(self, entity_ctx: torch.Tensor,
                situation: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        entity_ctx: (B, d_sem) — from EntityEncoder
        situation:  (B, T, d_sem) or (B, d_sem) — GWS / language context

        Returns:
          belief_emb:    (B, d_sem)
          desire_emb:    (B, d_sem)
          intention_logits: (B, 14)
        """
        if situation.dim() == 2:
            situation = situation.unsqueeze(1)  # (B, 1, d_sem)

        x = entity_ctx.unsqueeze(1)  # (B, 1, d_sem) — query
        for layer in self.layers:
            attn_out, _ = layer["cross_attn"](x, situation, situation)
            x = layer["ln1"](x + attn_out)
            x = layer["ln2"](x + layer["ff"](x))

        h = x.squeeze(1)   # (B, d_sem)
        return (
            self.belief_head(h),
            self.desire_head(h),
            self.intention_head(h),
        )


class CounterfactualSimulator(nn.Module):
    """Simulates: 'If I produce action A, entity X will respond R'.

    Single-step entity state transition model:
      s'_entity = f(s_entity, my_action)
    where s_entity = (style_ctx, belief_ctx).
    Then predicts response type from s'_entity.
    """

    def __init__(self, d_sem: int, n_action_types: int = 14):
        super().__init__()
        self.transition = nn.Sequential(
            nn.Linear(d_sem + n_action_types, d_sem * 2),
            nn.GELU(),
            nn.Linear(d_sem * 2, d_sem),
            nn.LayerNorm(d_sem),
        )
        self.response_head = nn.Linear(d_sem, n_action_types)

    def forward(self, entity_state: torch.Tensor,
                my_action_logits: torch.Tensor) -> torch.Tensor:
        """
        entity_state:      (B, d_sem)
        my_action_logits:  (B, n_action_types) — what I'm about to do
        → predicted_response_logits: (B, n_action_types)
        """
        my_action_probs = F.softmax(my_action_logits, dim=-1)
        x = torch.cat([entity_state, my_action_probs], dim=-1)
        next_state = self.transition(x)
        return self.response_head(next_state)


class SocialPredictionError(nn.Module):
    """Computes social prediction error → intrinsic social reward.

    When actual response matches predicted response → low error → DA signal.
    When response is surprising → high error → NE signal + update ToM.
    """

    def __init__(self, d_sem: int, n_types: int = 14):
        super().__init__()
        self.pred_proj    = nn.Linear(d_sem, n_types)
        self.actual_proj  = nn.Linear(d_sem, n_types)
        self.reward_head  = nn.Linear(n_types * 2, 1)

    def forward(self, predicted_state: torch.Tensor,
                actual_response: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        predicted_state:  (B, d_sem)
        actual_response:  (B, d_sem)
        Returns:
          error:  (B,) — social prediction error (0..1)
          reward: (B,) — social reward signal (higher = prediction was correct)
        """
        pred_logits   = self.pred_proj(predicted_state)
        actual_logits = self.actual_proj(actual_response)
        pred_probs    = F.softmax(pred_logits, dim=-1)
        actual_probs  = F.softmax(actual_logits, dim=-1)

        # KL divergence as prediction error
        kl = F.kl_div(pred_probs.log(), actual_probs, reduction="none").sum(-1)
        error = torch.sigmoid(kl - 1.0)   # centred around kl=1

        # Reward: inverse of error, weighted by action certainty
        combined = torch.cat([pred_probs, actual_probs], dim=-1)
        reward = torch.sigmoid(self.reward_head(combined)).squeeze(-1)
        return error, reward


class TheoryOfMindModule(BrainModule):
    """Full Theory of Mind module for NeuroSLM.

    Takes:
      • entity_emb   — entity's style embedding from EntityStore
      • belief_vec   — entity's belief state vector (6-dim)
      • log_n_inter  — log(1 + interactions) → model confidence
      • situation    — current GWS slots or language context
      • my_action    — what the model is about to do (optional)

    Produces:
      • entity_belief_emb    — entity's believed world state
      • entity_desire_emb    — entity's likely motivation
      • intention_logits     — entity's probable next action (14 types)
      • counterfactual_resp  — predicted response to my_action
      • social_reward        — intrinsic reward from correct predictions
      • social_error         — error signal → triggers entity model update
      • affect_bleed         — entity's predicted emotion → own NT modulation

    During inference, entity state is updated online via social_prediction_error
    so the model's ToM improves the more it interacts with an entity.
    """

    N_ACTION_TYPES = 14   # matches SocialMarkovMemory.N_TYPES

    def __init__(self, d_sem: int, d_style: int = 64,
                 d_belief: int = 6, n_heads: int = 4):
        super().__init__()
        self.d_sem    = d_sem
        self.d_style  = d_style
        self.d_belief = d_belief

        self.entity_encoder     = EntityEncoder(d_style, d_belief, d_sem)
        self.mental_state_dec   = MentalStateDecoder(d_sem, n_heads)
        self.cf_simulator       = CounterfactualSimulator(d_sem, self.N_ACTION_TYPES)
        self.social_pred_error  = SocialPredictionError(d_sem, self.N_ACTION_TYPES)

        # Emotional contagion: entity's predicted valence → own NT modulation
        self.affect_proj = nn.Sequential(
            nn.Linear(d_sem, 4), nn.Sigmoid())  # 4 NTs: DA, NE, 5HT, ACh

        # Entity state recurrent update (GRU) — refines model over interactions
        self.entity_rnn = nn.GRUCell(d_sem, d_sem)
        self._entity_hidden: Dict[str, torch.Tensor] = {}

    def _disabled_output(self, entity_emb, belief_vec, log_n_inter,
                         situation, my_action_logits=None, **_):
        B = situation.shape[0] if hasattr(situation, "shape") else 1
        d = self.d_sem
        dev = situation.device if hasattr(situation, "device") else torch.device("cpu")
        dt = situation.dtype if hasattr(situation, "dtype") else self.affect_proj[0].weight.dtype
        return {
            "entity_belief":   torch.zeros(B, d, device=dev, dtype=dt),
            "entity_desire":   torch.zeros(B, d, device=dev, dtype=dt),
            "intention_logits":torch.zeros(B, self.N_ACTION_TYPES, device=dev, dtype=dt),
            "cf_response":     torch.zeros(B, self.N_ACTION_TYPES, device=dev, dtype=dt),
            "social_reward":   torch.zeros(B, device=dev, dtype=dt),
            "social_error":    torch.zeros(B, device=dev, dtype=dt),
            "affect_bleed":    torch.zeros(B, 4, device=dev, dtype=dt),
        }

    def forward(self,
                entity_emb:       torch.Tensor,    # (B, d_style)
                belief_vec:       torch.Tensor,    # (B, 6)
                log_n_inter:      torch.Tensor,    # (B, 1)
                situation:        torch.Tensor,    # (B, T, d_sem) or (B, d_sem)
                my_action_logits: Optional[torch.Tensor] = None,   # (B, 14)
                entity_id:        Optional[str] = None,
                actual_response:  Optional[torch.Tensor] = None,   # (B, d_sem)
               ) -> dict:

        # 1. Encode entity profile → entity context
        entity_ctx = self.entity_encoder(entity_emb, belief_vec, log_n_inter)

        # 2. Recurrent entity state update (refines model with each interaction)
        if entity_id is not None:
            if entity_id not in self._entity_hidden:
                self._entity_hidden[entity_id] = torch.zeros_like(entity_ctx)
            h_prev = self._entity_hidden[entity_id].to(entity_ctx.device)
            h_new  = self.entity_rnn(entity_ctx, h_prev)
            self._entity_hidden[entity_id] = h_new.detach()
            entity_ctx = h_new

        # 3. Decode mental states via cross-attention over situation
        belief_emb, desire_emb, intention_logits = self.mental_state_dec(
            entity_ctx, situation)

        # 4. Counterfactual simulation (if I'm about to act)
        if my_action_logits is not None:
            cf_resp = self.cf_simulator(entity_ctx, my_action_logits)
        else:
            B = entity_ctx.shape[0]
            cf_resp = torch.zeros(B, self.N_ACTION_TYPES, device=entity_ctx.device, dtype=entity_ctx.dtype)

        # 5. Social prediction error (if we observed actual response)
        if actual_response is not None:
            social_error, social_reward = self.social_pred_error(
                belief_emb, actual_response)
        else:
            B = entity_ctx.shape[0]
            social_error  = torch.zeros(B, device=entity_ctx.device, dtype=entity_ctx.dtype)
            social_reward = torch.zeros(B, device=entity_ctx.device, dtype=entity_ctx.dtype)

        # 6. Emotional contagion signal (entity's predicted affect → own NTs)
        affect_bleed = self.affect_proj(belief_emb)   # (B, 4): DA, NE, 5HT, ACh

        return {
            "entity_belief":   belief_emb,
            "entity_desire":   desire_emb,
            "intention_logits":intention_logits,
            "cf_response":     cf_resp,
            "social_reward":   social_reward,
            "social_error":    social_error,
            "affect_bleed":    affect_bleed,
        }

    def reset_entity(self, entity_id: str):
        """Clear recurrent state for an entity (new conversation)."""
        self._entity_hidden.pop(entity_id, None)

    def entity_context_vector(self, entity_id: str,
                               device: torch.device) -> torch.Tensor:
        """Return stored entity hidden state for GWS injection."""
        if entity_id in self._entity_hidden:
            return self._entity_hidden[entity_id].to(device)
        return torch.zeros(self.d_sem, device=device, dtype=self.affect_proj[0].weight.dtype)
