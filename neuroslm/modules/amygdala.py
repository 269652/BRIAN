"""Amygdala — Emotional Memory Tagging and Fear Conditioning for NeuroSLM.

The amygdala is a small almond-shaped structure in the medial temporal lobe
that does two critical things no other brain area does:

  1. Assigns emotional significance to events (positive or negative valence)
  2. Learns *stimulus→emotion* associations through Hebbian conditioning
     so that previously neutral stimuli acquire emotional charge over time

Biologically:
  Basolateral Amygdala (BLA): receives sensory input, learns CS-US associations
    CS = conditioned stimulus (neutral cue)
    US = unconditioned stimulus (reward or threat)
    After pairing: CS alone triggers the emotional response
  Central Amygdala (CeA): drives defensive/approach behaviors
  Amygdala→Hippocampus: emotionally charged events are better consolidated
  Amygdala→PFC: gates cognitive control based on emotional state
  PFC→Amygdala: top-down extinction (learned suppression of fear)

What this adds that SubconsciousCritic does NOT provide:
  - Critic: binary threat/survival detection from world+self state
  - Amygdala: *learns* CS→emotion associations; produces graded emotional
    valence; drives memory consolidation; tracks emotional history; enables
    extinction and counter-conditioning

ML implementation:
  1. Valence head: maps semantic representation → emotional valence score
  2. Arousal head: maps representation → arousal intensity (calm↔activated)
  3. Hebbian associative memory: stores stimulus→emotion pairs, updates
     via outer-product rule weighted by US intensity
  4. Extinction gate: PFC input can suppress conditioned responses
  5. Emotional memory consolidation signal: high-arousal events get a
     consolidation boost for hippocampus/hypergraph

The amygdala output feeds:
  - NT system: high threat → NE/CRH; high reward → DA
  - Hippocampus: consolidation boost for emotional events
  - Floating thought: emotional coloring that persists into next tick
  - Motor cortex: approach/avoidance biasing

References:
  LeDoux (1996): The Emotional Brain
  Phelps & LeDoux (2005): Contributions of the Amygdala to Emotion Processing
  Quirk & Mueller (2008): Neural Mechanisms of Extinction Learning and Retrieval
  Maren (2001): Neurobiology of Pavlovian Fear Conditioning
"""
from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple

from .brain_module import BrainModule


class HebbianAssociativeMemory(nn.Module):
    """Fast associative memory that learns stimulus→emotion pairs online.

    Uses outer-product Hebbian rule:
      W += lr * (emotion_vec ⊗ stimulus_key)
    scaled by |US| (unconditioned stimulus intensity, e.g. threat/reward).

    Decay keeps the memory from saturating:
      W *= (1 - decay)

    Retrieval: W · stimulus_query → conditioned emotion response
    """

    def __init__(self, d_sem: int, d_emotion: int = 16,
                 capacity: int = 512, decay: float = 0.9995):
        super().__init__()
        self.d_sem    = d_sem
        self.d_emotion = d_emotion
        self.decay    = decay

        # Slow weights (learnable parameter) — long-term conditioning
        self.W_lt = nn.Parameter(torch.zeros(d_emotion, d_sem) * 0.01)

        # Fast weights (buffer, not gradient) — rapid single-trial learning
        self.register_buffer("W_fast",
                             torch.zeros(d_emotion, d_sem))

        # Key projection: stimulus → associative key
        self.key_proj = nn.Linear(d_sem, d_sem, bias=False)
        nn.init.eye_(self.key_proj.weight)

        # Extinction gate: how much to suppress fast weights
        self.extinction_gate = nn.Sequential(
            nn.Linear(d_sem, d_emotion),
            nn.Sigmoid(),
        )

    def associate(self, stimulus: torch.Tensor,
                  emotion: torch.Tensor,
                  us_strength: torch.Tensor,
                  pfc_input: Optional[torch.Tensor] = None) -> None:
        """Hebbian update: pair stimulus with emotion weighted by US strength.

        stimulus:   (B, d_sem)
        emotion:    (B, d_emotion)
        us_strength:(B,) unconditioned stimulus intensity in [0, 1]
        pfc_input:  (B, d_sem) PFC signal for extinction
        """
        with torch.no_grad():
            key = F.normalize(self.key_proj(stimulus.detach()), dim=-1)
            emo = emotion.detach()

            # Extinction: PFC reduces fast-weight update
            lr_scale = us_strength.detach()
            if pfc_input is not None:
                ext = 1.0 - self.extinction_gate(pfc_input.detach()).mean(0)
                lr_scale = lr_scale * ext.mean()

            # Outer product: (B, d_emotion, 1) × (B, 1, d_sem) → (B, d_emotion, d_sem)
            delta = torch.einsum("be,bs->bes", emo, key)
            # Average across batch, scale by US strength
            delta = (delta * lr_scale.view(-1, 1, 1)).mean(0)

            # Update fast weights with decay
            self.W_fast = self.decay * self.W_fast + (1 - self.decay) * delta

    def recall(self, stimulus: torch.Tensor) -> torch.Tensor:
        """Retrieve conditioned emotion for a stimulus.

        stimulus: (B, d_sem)
        Returns: (B, d_emotion) conditioned emotional response
        """
        key = F.normalize(self.key_proj(stimulus), dim=-1)
        W = self.W_fast + self.W_lt
        return torch.einsum("es,bs->be", W, key)   # (B, d_emotion)


class Amygdala(BrainModule):
    """Full amygdala: valence/arousal encoding + Hebbian conditioning.

    Parameters
    ----------
    d_sem      : semantic dimension
    d_emotion  : emotional representation dimension (BLA output)
    n_nt       : number of neurotransmitters
    """

    def __init__(self, d_sem: int, d_emotion: int = 32, n_nt: int = 8):
        super().__init__()
        self.d_sem    = d_sem
        self.d_emotion = d_emotion

        # BLA: stimulus → emotional representation
        self.bla_encoder = nn.Sequential(
            nn.Linear(d_sem, d_sem // 2),
            nn.GELU(),
            nn.Linear(d_sem // 2, d_emotion),
            nn.Tanh(),
        )

        # Valence head: emotion rep → scalar [-1, +1]
        # -1 = maximum threat/aversion; +1 = maximum pleasure/safety
        self.valence_head = nn.Sequential(
            nn.Linear(d_emotion, d_emotion // 2),
            nn.GELU(),
            nn.Linear(d_emotion // 2, 1),
            nn.Tanh(),
        )

        # Arousal head: emotion rep → scalar [0, 1]
        # 0 = completely calm; 1 = maximally aroused
        self.arousal_head = nn.Sequential(
            nn.Linear(d_emotion, d_emotion // 2),
            nn.GELU(),
            nn.Linear(d_emotion // 2, 1),
            nn.Sigmoid(),
        )

        # Associative memory (BLA)
        self.associative = HebbianAssociativeMemory(d_sem, d_emotion)

        # CeA: drives NT release demands from emotional state
        # Maps emotion → NT modulation [DA, NE, 5HT, ACh, eCB, GABA, CRH, NPY]
        self.cea = nn.Sequential(
            nn.Linear(d_emotion * 2, n_nt),  # current + conditioned emotion
            nn.Sigmoid(),
        )

        # Emotion → thought coloring: how to tint the floating thought
        self.emotion_to_thought = nn.Linear(d_emotion, d_sem, bias=False)
        nn.init.zeros_(self.emotion_to_thought.weight)

        # Consolidation signal: how urgently hippocampus should consolidate this
        self.consolidation_head = nn.Sequential(
            nn.Linear(d_emotion, 1),
            nn.Sigmoid(),
        )

        # Emotional memory decay (EMA over ticks)
        self.register_buffer("ema_emotion", torch.zeros(d_emotion))
        self.ema_alpha = 0.2

    # ------------------------------------------------------------------

    def forward(self, stimulus: torch.Tensor,
                threat: torch.Tensor,
                reward: torch.Tensor,
                pfc_input: Optional[torch.Tensor] = None,
               ) -> dict:
        """
        stimulus:  (B, d_sem) current semantic representation
        threat:    (B,) threat scalar from SubconsciousCritic [0, 1]
        reward:    (B,) reward proxy [0, 1]
        pfc_input: (B, d_sem) PFC signal for extinction gating

        Returns dict with:
          emotion:         (B, d_emotion) — BLA emotional representation
          valence:         (B,)           — [-1, +1] emotional valence
          arousal:         (B,)           — [0, 1] arousal intensity
          nt_demand:       (B, n_nt)      — NT release demand from CeA
          thought_tint:    (B, d_sem)     — how to color the floating thought
          consolidation:   (B,)           — memory consolidation urgency
        """
        B = stimulus.shape[0]

        # 1. BLA: encode current stimulus into emotional representation
        bla_emotion = self.bla_encoder(stimulus)   # (B, d_emotion)

        # 2. Retrieve conditioned response from associative memory
        conditioned = self.associative.recall(stimulus)  # (B, d_emotion)

        # Combine: current + conditioned (CeA receives both)
        combined_emotion = bla_emotion + 0.4 * conditioned

        # 3. Valence and arousal
        valence = self.valence_head(combined_emotion).squeeze(-1)   # (B,)
        arousal = self.arousal_head(combined_emotion).squeeze(-1)   # (B,)

        # Bias valence: threat → negative, reward → positive
        valence = (valence + reward - threat).clamp(-1, 1)

        # 4. Update associative memory (Hebbian conditioning)
        # US = threat or reward (whichever is stronger)
        us_strength = torch.maximum(threat, reward)
        self.associative.associate(stimulus, bla_emotion, us_strength, pfc_input)

        # 5. CeA NT demands
        cat_emo = torch.cat([bla_emotion, conditioned], dim=-1)
        nt_demand = self.cea(cat_emo)   # (B, n_nt)
        # Scale NT demands by arousal
        nt_demand = nt_demand * arousal.unsqueeze(-1)

        # 6. EMA for smooth emotional tone
        with torch.no_grad():
            self.ema_emotion = ((1 - self.ema_alpha) * self.ema_emotion
                                + self.ema_alpha * combined_emotion.detach().mean(0))

        # 7. Thought tinting: emotion colors the floating thought
        # Positive valence pushes thought toward "positive" semantic region
        # Negative valence (threat) pushes toward "negative" semantic region
        emotion_for_tint = combined_emotion * valence.unsqueeze(-1)
        thought_tint = self.emotion_to_thought(emotion_for_tint)   # (B, d_sem)

        # 8. Consolidation urgency: high arousal + high |valence| → consolidate
        consolidation = self.consolidation_head(combined_emotion).squeeze(-1)
        consolidation = consolidation * arousal   # (B,)

        return {
            "emotion":       combined_emotion,
            "valence":       valence,
            "arousal":       arousal,
            "nt_demand":     nt_demand,
            "thought_tint":  thought_tint,
            "consolidation": consolidation,
        }
