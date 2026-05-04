"""Insula — Interoception and Gut Feelings for NeuroSLM.

The insula (insular cortex) is the brain's "body sense" region. It receives
signals from the viscera (heart, gut, lungs), maps them to conscious
awareness, and creates the felt quality of emotions.

Key biological functions:
  1. Interoception: awareness of internal body state (heart rate, hunger,
     temperature, pain). In humans, this is the felt sense of "being alive."
  2. Somatic marker hypothesis (Damasio): decisions are guided by "gut
     feelings" — bodily sensations that tag past choices as good or bad.
     The insula makes these somatic markers available to cognition.
  3. Empathy: the insula lets you *feel* others' pain/joy by simulating
     their bodily state internally.
  4. Risk/disgust: anterior insula fires when facing risky options and
     when processing morally repugnant content.
  5. Salience gating: insula + ACC form the Salience Network, deciding
     when to interrupt the DMN (default mode) and recruit the executive
     network (PFC).

For a language model, "interoception" maps to:
  - Model's own processing load (are we "strained"?)
  - Homeostatic states (energy usage proxy, recency of learning, etc.)
  - Semantic content about bodies, pain, pleasure, health → empathy activation
  - Social content → resonance response (felt reaction to described events)

What this adds:
  - Somatic marker signal that colors decisions with felt-quality
  - Empathic resonance when the model encounters social/embodied content
  - Fatigue/effort signal based on cognitive load
  - Disgust/acceptance signals for content filtering

References:
  Damasio (1994): Descartes' Error — somatic marker hypothesis
  Craig (2009): How do you feel — now? The anterior insula and human awareness
  Singer & Lamm (2009): The social neuroscience of empathy
  Menon & Uddin (2010): Saliency, switching, attention and control: the insula
"""
from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple

from .brain_module import BrainModule


class SomaticMarkerHead(nn.Module):
    """Computes somatic markers — felt valence tags for decisions.

    Somatic markers are body-state "tags" that signal whether a past
    choice led to good or bad outcomes. They gate decision-making without
    requiring explicit reasoning.
    """

    def __init__(self, d_sem: int, n_markers: int = 8):
        super().__init__()
        self.markers = nn.Sequential(
            nn.Linear(d_sem, d_sem // 2),
            nn.GELU(),
            nn.Linear(d_sem // 2, n_markers),
            nn.Tanh(),
        )
        # Marker → go/no-go gate for decisions
        self.gate = nn.Sequential(
            nn.Linear(n_markers, 1),
            nn.Tanh(),
        )

    def forward(self, rep: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        rep: (B, d_sem) or (B, T, d_sem)
        Returns: markers (B, n_markers), go_nogo (B,) in [-1, +1]
        """
        if rep.dim() == 3:
            rep = rep.mean(1)
        markers = self.markers(rep)
        go_nogo = self.gate(markers).squeeze(-1)
        return markers, go_nogo


class EmpathyResonance(nn.Module):
    """Computes empathic resonance — the model's felt reaction to others.

    When the model processes descriptions of others' experiences, the
    insula fires to simulate the *felt quality* of those experiences.
    This produces:
      - Empathy signal: how much resonance with the described experience
      - Valence shift: does the content push the model's felt state +/-?
    """

    def __init__(self, d_sem: int):
        super().__init__()
        # Detect social/embodied content
        self.social_detector = nn.Sequential(
            nn.Linear(d_sem, d_sem // 2),
            nn.GELU(),
            nn.Linear(d_sem // 2, 1),
            nn.Sigmoid(),
        )
        # Simulate others' emotional state from description
        self.simulation = nn.Sequential(
            nn.Linear(d_sem, d_sem),
            nn.GELU(),
            nn.Linear(d_sem, d_sem),
            nn.Tanh(),
        )
        # Resonance: how much to blend simulated state into own felt state
        self.resonance_gate = nn.Sequential(
            nn.Linear(d_sem * 2, 1),
            nn.Sigmoid(),
        )

    def forward(self, stimulus: torch.Tensor,
                own_state: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        stimulus:  (B, d_sem) the input representation
        own_state: (B, d_sem) current internal state

        Returns:
          empathy_state: (B, d_sem) simulated others' state blended in
          resonance:     (B,)       how much we resonated [0, 1]
        """
        social_weight = self.social_detector(stimulus)  # (B, 1)
        simulated     = self.simulation(stimulus)        # (B, d_sem)

        gate_inp = torch.cat([simulated, own_state], dim=-1)
        resonance = self.resonance_gate(gate_inp).squeeze(-1)  # (B,)

        # Blend simulated state proportionally to social content and resonance
        blend = social_weight.squeeze(-1) * resonance
        empathy_state = own_state + blend.unsqueeze(-1) * simulated * 0.3

        return empathy_state, resonance


class Insula(BrainModule):
    """Insular cortex: interoception, somatic markers, empathy, salience.

    Parameters
    ----------
    d_sem    : semantic dimension
    n_nt     : neurotransmitter count
    n_markers: somatic marker dimensions
    """

    def __init__(self, d_sem: int, n_nt: int = 8, n_markers: int = 8):
        super().__init__()
        self.d_sem = d_sem

        # Anterior insula: high-level interoceptive awareness
        self.anterior = nn.Sequential(
            nn.Linear(d_sem + n_markers, d_sem),
            nn.GELU(),
            nn.Linear(d_sem, d_sem),
            nn.LayerNorm(d_sem),
        )

        # Posterior insula: raw body signal processing
        # Maps NT state (proxy for physiological state) → interoceptive signal
        self.posterior = nn.Sequential(
            nn.Linear(n_nt, d_sem // 2),
            nn.GELU(),
            nn.Linear(d_sem // 2, d_sem // 2),
            nn.Sigmoid(),
        )

        # Somatic markers
        self.somatic = SomaticMarkerHead(d_sem, n_markers)

        # Empathy resonance
        self.empathy = EmpathyResonance(d_sem)

        # Salience gate: insula + ACC → interrupt DMN?
        self.salience_gate = nn.Sequential(
            nn.Linear(d_sem + d_sem // 2, 1),
            nn.Sigmoid(),
        )

        # Disgust/risk head: flags aversive content
        self.aversion_head = nn.Sequential(
            nn.Linear(d_sem, 1),
            nn.Sigmoid(),
        )

        # Effort/fatigue: maps computation history to felt effort
        self.fatigue_head = nn.Sequential(
            nn.Linear(n_nt + 1, 1),  # NT state + explicit effort scalar
            nn.Sigmoid(),
        )

        # Insula → NT modulation: pain → NE, pleasure → DA, disgust → 5HT↓
        self.nt_demand = nn.Sequential(
            nn.Linear(d_sem + n_markers, n_nt),
            nn.Sigmoid(),
        )

    # ------------------------------------------------------------------

    def forward(self, stimulus: torch.Tensor,
                nt_vec: torch.Tensor,
                floating_thought: Optional[torch.Tensor] = None,
                effort_proxy: Optional[torch.Tensor] = None,
               ) -> dict:
        """
        stimulus:       (B, d_sem) current semantic representation
        nt_vec:         (B, n_nt) neurotransmitter levels (proxy for body state)
        floating_thought: (B, d_sem) current thought for empathy computation
        effort_proxy:   (B,) current cognitive effort/load [0, 1]

        Returns dict with:
          interoceptive:   (B, d_sem)  — interoceptive representation
          somatic_markers: (B, n_markers)
          go_nogo:         (B,)        — decision gate in [-1, +1]
          empathy_state:   (B, d_sem)  — thought colored by empathic resonance
          resonance:       (B,)        — empathic resonance level
          salience:        (B,)        — interrupt DMN? [0, 1]
          aversion:        (B,)        — disgust/risk level [0, 1]
          fatigue:         (B,)        — effort/fatigue level [0, 1]
          nt_demand:       (B, n_nt)   — NT release demand
        """
        B, device = stimulus.shape[0], stimulus.device

        # Posterior insula: NT state → body sense
        body_sense = self.posterior(nt_vec)            # (B, d_sem//2)

        # Somatic markers
        markers, go_nogo = self.somatic(stimulus)      # (B, n_m), (B,)

        # Anterior insula: integrate body sense with semantic input
        ant_inp = torch.cat([stimulus, markers], dim=-1)
        insula_rep = self.anterior(ant_inp)             # (B, d_sem)

        # Empathy resonance (if floating_thought provided)
        thought = floating_thought if floating_thought is not None else stimulus
        empathy_state, resonance = self.empathy(stimulus, thought)

        # Salience gate
        sal_inp = torch.cat([insula_rep, body_sense], dim=-1)
        salience = self.salience_gate(sal_inp).squeeze(-1)     # (B,)

        # Aversion
        aversion = self.aversion_head(insula_rep).squeeze(-1)  # (B,)

        # Fatigue
        if effort_proxy is not None:
            fat_inp = torch.cat([nt_vec, effort_proxy.unsqueeze(-1)], dim=-1)
        else:
            fat_inp = torch.cat([nt_vec,
                                  torch.zeros(B, 1, device=device)], dim=-1)
        fatigue = self.fatigue_head(fat_inp).squeeze(-1)        # (B,)

        # NT demands
        nt_inp = torch.cat([insula_rep, markers], dim=-1)
        nt_demand = self.nt_demand(nt_inp)                      # (B, n_nt)

        return {
            "interoceptive":   insula_rep,
            "somatic_markers": markers,
            "go_nogo":         go_nogo,
            "empathy_state":   empathy_state,
            "resonance":       resonance,
            "salience":        salience,
            "aversion":        aversion,
            "fatigue":         fatigue,
            "nt_demand":       nt_demand,
        }
