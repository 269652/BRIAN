"""Sensory cortices.

In this text-only prototype, the only modality is language. This module is a
thin wrapper that takes a comprehension embedding from the language cortex and
exposes it as a 'sensory token'. It is the integration point for adding vision
or audio later (each modality would own a SensoryEncoder subclass).

Topic classification (new):
  TopicClassifier classifies the semantic embedding into 4 domains:
    0 = DEFAULT   (general conversation)
    1 = MATH      (numerical / symbolic reasoning)
    2 = REASONING (relational / logical)
    3 = LANGUAGE  (linguistic / pragmatic)
  The topic probabilities gate the corresponding expert cortices
  and trigger typed vesicle synthesis in VesiclePool.
"""
from __future__ import annotations
from typing import Dict
import torch
import torch.nn as nn
import torch.nn.functional as F


class TextSensoryCortex(nn.Module):
    """Identity-ish wrapper; the language cortex already does the encoding work.

    Kept as a separate module so that:
      - swapping modalities later is clean
      - a salience/attention mask can be applied here (superior-colliculus analog)
    """
    def __init__(self, d_sem: int):
        super().__init__()
        self.salience = nn.Linear(d_sem, 1)
        self.proj = nn.Linear(d_sem, d_sem)

    def forward(self, sem: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        # sem: (B, d_sem)
        salience = torch.sigmoid(self.salience(sem))   # (B, 1)
        encoded = self.proj(sem) * salience
        return encoded, salience.squeeze(-1)


class SensoryFrameEncoder(nn.Module):
    """Maps a SensoryFrame numeric vector (6-dim) to a d_sem embedding.

    Used in the continuous sensory world loop: at each forward_lm step, one
    virtual-world frame is pulled from the environment stream, encoded here,
    and added as a grounding residual to the world model's latent state.

    Input channels (must match SensoryFrame.to_vec() order):
      valence, arousal, novelty, comfort, time_pressure, social_presence
    """
    N_CHANNELS = 6

    def __init__(self, d_sem: int):
        super().__init__()
        hidden = max(32, d_sem // 4)
        self.enc = nn.Sequential(
            nn.Linear(self.N_CHANNELS, hidden),
            nn.SiLU(),
            nn.Linear(hidden, d_sem),
        )
        # Zero-init output so encoder starts as a no-op; effect grows with training
        nn.init.zeros_(self.enc[2].weight)
        nn.init.zeros_(self.enc[2].bias)

    def encode_frame(self, frame_vec: list[float], device,
                     dtype=torch.float32) -> torch.Tensor:
        """frame_vec: list of 6 floats → (1, d_sem)"""
        t = torch.tensor(frame_vec, dtype=dtype, device=device).unsqueeze(0)
        return self.enc(t)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, N_CHANNELS) → (B, d_sem)"""
        return self.enc(x)


# Topic type constants (must stay in sync with neurochem/vesicles.py)
TOPIC_DEFAULT   = 0
TOPIC_MATH      = 1
TOPIC_REASONING = 2
TOPIC_LANGUAGE  = 3
N_TOPICS        = 4


# ─────────────────────────────────────────────────────────────────────────────
# Grid-world sensory VAE — compresses (vision, affordance, homeostatic) into
# the d_sem manifold consumed by the bowtie.
# ─────────────────────────────────────────────────────────────────────────────

class SensoryVAE(nn.Module):
    """Tiny VAE front-end for 10×10 grid frames + homeostatic vector.

    Architecture:
      • Vision: 2-conv encoder → flatten → linear → d_sem//2
      • Homeostatic (3-vec) → linear → d_sem//4
      • Affordance counts (n_rel-dim) → linear → d_sem//4
      • Concatenate → linear → (μ, log σ²) → d_sem
      • Reparameterise → z ∈ ℝ^{d_sem}
      • Decoder: linear → conv-transpose 2 → reconstruct vision channel

    Loss: reconstruction MSE + β·KL(N(μ, σ²) ‖ N(0, I)). The encoder is
    trained directly by the brain as part of the auxiliary loss block
    (gated by `_aux_w_scale`).

    The encoder output z is what gets injected into the bowtie's sensory
    cortex — same conceptual role as `SensoryFrameEncoder` for text
    frames, but with proper VAE structure so the latent is well-formed.
    """

    N_REL = 6          # see env/grid_world.py — above/below/left_of/right_of/next_to/on_top_of
    N_HOM = 3          # energy, hydration, integrity

    def __init__(self, d_sem: int, grid_size: int = 10,
                 n_block_types: int = 7, beta: float = 1.0):
        super().__init__()
        self.d_sem = d_sem
        self.grid_size = grid_size
        self.n_block_types = n_block_types
        self.beta = beta

        # Vision encoder — input (B, C=n_block_types, H, W)
        self.vision_enc = nn.Sequential(
            nn.Conv2d(n_block_types, 16, 3, padding=1),
            nn.SiLU(),
            nn.Conv2d(16, 32, 3, padding=1, stride=2),     # 10 → 5
            nn.SiLU(),
            nn.Flatten(),
            nn.Linear(32 * 5 * 5, d_sem // 2),
        )

        # Homeostatic encoder — 3-vec
        self.hom_enc = nn.Sequential(
            nn.Linear(self.N_HOM, max(8, d_sem // 16)),
            nn.SiLU(),
            nn.Linear(max(8, d_sem // 16), d_sem // 4),
        )

        # Affordance encoder — relation counts
        self.aff_enc = nn.Sequential(
            nn.Linear(self.N_REL, max(8, d_sem // 16)),
            nn.SiLU(),
            nn.Linear(max(8, d_sem // 16), d_sem // 4),
        )

        # Mix + project to (μ, log σ²)
        # vision (d_sem // 2) + hom (d_sem // 4) + aff (d_sem // 4) ≈ d_sem
        total = d_sem // 2 + d_sem // 4 + d_sem // 4
        # (compensate for integer rounding so the mix dim ≤ d_sem)
        self.mix = nn.Linear(total, d_sem)
        self.mu      = nn.Linear(d_sem, d_sem)
        self.logvar  = nn.Linear(d_sem, d_sem)

        # Vision-only decoder for reconstruction loss
        self.vision_dec = nn.Sequential(
            nn.Linear(d_sem, 32 * 5 * 5),
            nn.SiLU(),
            nn.Unflatten(1, (32, 5, 5)),
            nn.ConvTranspose2d(32, 16, 3, padding=1, stride=2,
                                 output_padding=1),       # 5 → 10
            nn.SiLU(),
            nn.Conv2d(16, n_block_types, 3, padding=1),
        )

    # ── encoding ─────────────────────────────────────────────────────────────

    def _affordance_counts(self,
                            affordance: Dict[str, list]) -> torch.Tensor:
        """Convert affordance-relation dict into the N_REL-vector of counts."""
        order = ("above", "below", "left_of", "right_of", "next_to", "on_top_of")
        return torch.tensor(
            [float(len(affordance.get(k, []))) for k in order],
            dtype=torch.float32)

    def encode_frame(self,
                      frame,              # GridFrame
                      device,
                      dtype=torch.float32) -> torch.Tensor:
        """Encode one ``GridFrame`` into the latent ``z`` ∈ ℝ^{d_sem}."""
        # Vision: (1, C, H, W)
        v = torch.as_tensor(frame.vision, dtype=dtype, device=device)
        if v.dim() == 3:
            # (H, W, C) → (C, H, W)
            v = v.permute(2, 0, 1)
        v = v.unsqueeze(0)
        # Homeostatic + affordance
        h = torch.as_tensor(frame.homeostatic, dtype=dtype, device=device).unsqueeze(0)
        a = self._affordance_counts(frame.affordance).to(device=device, dtype=dtype).unsqueeze(0)

        z, _mu, _lv = self.forward(v, h, a)
        return z

    # ── forward (training-time) ──────────────────────────────────────────────

    def forward(self,
                vision:      torch.Tensor,    # (B, C, H, W)
                homeostatic: torch.Tensor,    # (B, 3)
                affordance:  torch.Tensor,    # (B, N_REL)
                ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        v = self.vision_enc(vision)
        h = self.hom_enc(homeostatic)
        a = self.aff_enc(affordance)
        x = torch.cat([v, h, a], dim=-1)
        x = self.mix(x)
        mu = self.mu(x)
        lv = self.logvar(x).clamp(min=-10.0, max=10.0)
        # Reparameterise
        eps = torch.randn_like(mu)
        z = mu + (0.5 * lv).exp() * eps
        return z, mu, lv

    def reconstruction_loss(self,
                              vision:      torch.Tensor,
                              homeostatic: torch.Tensor,
                              affordance:  torch.Tensor,
                              ) -> torch.Tensor:
        """β-VAE loss = MSE(decoded, vision) + β · KL(N(μ,σ²) || N(0,I))."""
        z, mu, lv = self.forward(vision, homeostatic, affordance)
        recon = self.vision_dec(z)
        # Reconstruct only the vision channel (homeostat + affordance are
        # too low-dim to need decode; KL pressure shapes their part of z).
        rec_loss = F.mse_loss(recon, vision)
        kl = -0.5 * (1 + lv - mu.pow(2) - lv.exp()).mean()
        return rec_loss + self.beta * kl


class TopicClassifier(nn.Module):
    """Classifies a semantic embedding into topic domains for expert routing.

    Outputs a probability distribution over N_TOPICS topic types.
    Used by brain.py to:
      (a) gate MathCortex and ReasoningCortex (vesicle_gate argument)
      (b) synthesize typed vesicles (VesiclePool.synthesize_typed)

    Zero-init output so classification starts uniform (no routing bias).
    """

    def __init__(self, d_sem: int):
        super().__init__()
        hidden = max(64, d_sem // 2)
        self.net = nn.Sequential(
            nn.Linear(d_sem, hidden),
            nn.GELU(),
            nn.LayerNorm(hidden),
            nn.Linear(hidden, N_TOPICS),
        )
        # Zero-init output so classifier starts as uniform (no spurious routing)
        nn.init.zeros_(self.net[3].weight)
        nn.init.zeros_(self.net[3].bias)

    def forward(self, sem: torch.Tensor) -> torch.Tensor:
        """sem: (B, d_sem) → topic_probs: (B, N_TOPICS) summing to 1."""
        logits = self.net(sem)
        return F.softmax(logits, dim=-1)
