"""NT-producing nuclei.

Each nucleus consumes some signals (novelty, reward, surprise, valence,
information-theoretic gain) and emits a release demand for its principal NT.
The actual release goes through `TransmitterSystem.release()`.

Mapping (simplified):
  VTA              -> Dopamine (DA)
  NucleusAccumbens -> reward integration; drives VTA via feedback;
                      models mesolimbic reward-prediction-error learning signal
  LocusCoeruleus   -> Norepinephrine (NE)
  RapheNuclei      -> Serotonin (5HT)
  BasalForebrain   -> Acetylcholine (ACh)

(Endocannabinoids — eCB — are produced postsynaptically by many regions and
are released by the *target* region itself in `Brain`, not by a dedicated
nucleus, modeling retrograde signaling.)
"""
from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F


class _NucleusBase(nn.Module):
    """Tiny MLP: feature vector → release demand in [0,1]."""
    def __init__(self, in_features: int, hidden: int = 16):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_features, hidden), nn.GELU(),
            nn.Linear(hidden, 1),
        )

    def demand(self, x: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(self.net(x)).squeeze(-1)  # (B,)


class VTA(_NucleusBase):
    """Ventral tegmental area: produces DA in response to reward-prediction
    error (RPE) and salience. Inputs:
      [rpe, salience, nacc_drive, valence]
    """
    def __init__(self):
        super().__init__(in_features=4)


class NucleusAccumbens(nn.Module):
    """NAcc: integrates novelty + reward + curiosity into a "wanting" signal,
    which drives VTA dopamine release. Also computes a *learning* signal that
    can be used to scale gradients (mesolimbic reinforcement).
    """
    def __init__(self):
        super().__init__()
        self.integrate = nn.Sequential(
            nn.Linear(4, 16), nn.GELU(),
            nn.Linear(16, 2),  # [drive_to_VTA, learning_gain]
        )

    def forward(self, novelty: torch.Tensor, reward: torch.Tensor,
                curiosity: torch.Tensor, ecb: torch.Tensor):
        """All inputs (B,). Returns (drive, learning_gain), both (B,) in [0,1].
        Endocannabinoid `ecb` is *disinhibitory* here (CB1 on local GABA neurons
        increases NAcc output) — modeled by adding it as a positive input."""
        x = torch.stack([novelty, reward, curiosity, ecb], dim=-1)
        out = torch.sigmoid(self.integrate(x))
        return out[..., 0], out[..., 1]


class LocusCoeruleus(_NucleusBase):
    """LC: NE. Inputs: [unexpected_uncertainty, arousal, novelty]."""
    def __init__(self):
        super().__init__(in_features=3)


class RapheNuclei(_NucleusBase):
    """Dorsal raphe: 5HT. Inputs: [recent_avg_reward, time_since_reward, mood]."""
    def __init__(self):
        super().__init__(in_features=3)


class BasalForebrain(_NucleusBase):
    """Nucleus basalis of Meynert: ACh. Inputs: [attention_demand, novelty, surprise]."""
    def __init__(self):
        super().__init__(in_features=3)


class SubstantiaNigra(nn.Module):
    """Substantia Nigra pars compacta (SNc) + pars reticulata (SNr).

    SNc: produces DA for the nigrostriatal pathway (motor planning / BG).
    SNr: produces GABA for thalamic gating (suppresses unwanted actions).

    Inputs to SNc: [motor_intent, bg_activity, d2_autoreceptor_feedback]
    Inputs to SNr: [bg_indirect_path, cortical_inhibition, gaba_tone]
    """
    def __init__(self):
        super().__init__()
        # SNc: DA release for dorsal striatum / BG direct pathway
        self.snc = nn.Sequential(
            nn.Linear(3, 16), nn.GELU(),
            nn.Linear(16, 1),
        )
        # SNr: GABA release to thalamus (inhibitory gate)
        self.snr = nn.Sequential(
            nn.Linear(3, 16), nn.GELU(),
            nn.Linear(16, 1),
        )

    def forward(self, motor_intent: torch.Tensor, bg_activity: torch.Tensor,
                d2_feedback: torch.Tensor, indirect_path: torch.Tensor,
                cortical_inhib: torch.Tensor, gaba_tone: torch.Tensor):
        """Returns (da_demand, gaba_demand) both (B,) in [0,1]."""
        snc_in = torch.stack([motor_intent, bg_activity, d2_feedback], dim=-1)
        snr_in = torch.stack([indirect_path, cortical_inhib, gaba_tone], dim=-1)
        da_demand = torch.sigmoid(self.snc(snc_in)).squeeze(-1)
        gaba_demand = torch.sigmoid(self.snr(snr_in)).squeeze(-1)
        return da_demand, gaba_demand


class PeriaqueductalGray(_NucleusBase):
    """PAG: opioid/eCB release for pain suppression and stress analgesia.
    Inputs: [threat_level, stress, arousal]."""
    def __init__(self):
        super().__init__(in_features=3)


class HypothalamicCRH(_NucleusBase):
    """Hypothalamic CRH neurons: stress axis (HPA).
    Drives NE/cortisol response to sustained threats.
    Inputs: [sustained_threat, uncertainty, metabolic_demand]."""
    def __init__(self):
        super().__init__(in_features=3)

