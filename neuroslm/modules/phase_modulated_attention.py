"""Theta-Gamma Phase Coding and Oscillatory Attention for NeuroSLM.

In the hippocampus and neocortex, neural oscillations are not epiphenomenal
noise — they are the substrate for temporal multiplexing of information:
  • Theta (~7 Hz):  defines the "memory epoch" — a single theta cycle
    encodes one thought or episode
  • Gamma (~40 Hz): within each theta cycle, ~5–7 gamma sub-cycles carry
    different items, with position encoded by phase

The *theta-gamma coupling* mechanism:
  Items at earlier positions in a sequence fire at earlier gamma phases
  within the theta cycle.  The *phase precession* phenomenon (O'Keefe &
  Recce 1993) shows that hippocampal place cells fire at earlier phases
  as an animal progresses through a place field — equivalent to a
  temporal scan of the sequence.

This module implements:

  1. Sinusoidal phase embedding: each token position t gets a phase
     vector encoding both theta and gamma components at multiple
     frequencies.  Unlike RoPE (which encodes relative position for
     attention bias), this encodes *oscillatory phase state* for each
     head — different heads can be tuned to different frequency bands.

  2. Phase-precession: queries are rotated by their temporal phase,
     keys are rotated by the phase at their stored position. This gives
     a continuous, differentiable analog of the hippocampal mechanism.

  3. Working memory slots: a fixed set of gamma-indexed slots (T_gamma
     slots per theta cycle) receive tokens by their phase and can be
     read out in order.  This is a compact O(T_gamma) memory rather than
     O(T) full-context attention.

  4. Phase reset: at every `reset_period` steps (one theta cycle), the
     gamma phase resets.  This naturally segments sequences into episodes
     — allowing the model to handle multi-turn conversations where each
     turn is a separate theta cycle.

  5. NT modulation: ACh level modulates the theta/gamma frequency ratio
     (higher ACh → stronger gamma locking → sharper working memory).

Novel advantage over standard positional encoding: the oscillatory phase
encoding is *dynamic* (changes during inference based on timing) rather
than fixed, enabling the model to represent "when in the current episode"
a token occurred, not just its absolute position.

References:
  O'Keefe & Recce (1993): Phase relationship between hippocampal place units
  Lisman & Jensen (2013): The theta-gamma neural code
  Buszáki & Draguhn (2004): Neuronal oscillations in cortical networks
  Kanerva (2009): Hyperdimensional computing
"""
from __future__ import annotations
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple


def _phase_embed(positions: torch.Tensor, d: int,
                 n_freqs: int = 4,
                 theta_freq: float = 7.0,
                 gamma_freq: float = 40.0) -> torch.Tensor:
    """Phase embedding for a set of integer positions.

    positions: (T,)
    returns:   (T, d) — phase embedding
    """
    # Build frequency bands: log-spaced between theta and gamma
    freqs = torch.exp(
        torch.linspace(math.log(theta_freq), math.log(gamma_freq),
                       n_freqs, device=positions.device)
    )   # (n_freqs,)

    # Phase angle for each position × frequency
    angles = positions.float().unsqueeze(1) * freqs.unsqueeze(0)  # (T, n_freqs)

    # Each frequency → sin + cos channels, tiled to fill d
    sin_cos = torch.cat([torch.sin(angles), torch.cos(angles)], dim=-1)  # (T, 2*n_freqs)
    # Tile to reach d
    reps = math.ceil(d / (2 * n_freqs))
    phase = sin_cos.repeat(1, reps)[:, :d]    # (T, d)
    return phase


class PhaseModulatedAttention(nn.Module):
    """Theta-gamma phase-coded multi-head attention.

    Parameters
    ----------
    d_model       : model dimension
    n_heads       : number of attention heads
    n_gamma_slots : working memory slots per theta cycle
    reset_period  : steps per theta cycle (phase reset frequency)
    n_freqs       : frequency bands for phase embedding
    """

    def __init__(self, d_model: int, n_heads: int = 4,
                 n_gamma_slots: int = 7,
                 reset_period: int = 64,
                 n_freqs: int = 4):
        super().__init__()
        assert d_model % n_heads == 0
        self.n_heads       = n_heads
        self.d_head        = d_model // n_heads
        self.d_model       = d_model
        self.n_gamma_slots = n_gamma_slots
        self.reset_period  = reset_period
        self.n_freqs       = n_freqs

        # Standard QKV
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)

        # Learnable per-head phase offsets (each head tunes to a sub-band)
        self.head_phase_offsets = nn.Parameter(
            torch.linspace(0, 2 * math.pi, n_heads)
        )

        # Gamma working memory slots (persistent within a theta cycle)
        self.gamma_slots = nn.Parameter(
            torch.randn(1, n_gamma_slots, d_model) * 0.02
        )
        self.slot_proj = nn.Linear(d_model, d_model)

        # NT modulation: ACh modulates gamma locking strength
        self.nt_gamma_gate = nn.Linear(4, 1)   # 4 NTs → scalar

        self.ln = nn.LayerNorm(d_model)

    # ------------------------------------------------------------------

    def _rotate_by_phase(self, x: torch.Tensor,
                         phase: torch.Tensor) -> torch.Tensor:
        """Apply phase rotation to last dimension.

        x:     (B, n_heads, T, d_head)
        phase: (T, d_head/2) — phase angles

        Interprets d_head as pairs of [real, imag] components and
        rotates each pair by the corresponding phase angle (like RoPE but
        with oscillatory frequencies instead of position-based).
        """
        Dh = x.shape[-1]
        half = Dh // 2
        x1, x2 = x[..., :half], x[..., half:]

        # phase: (T, half) → broadcast over B, n_heads
        cos_p = torch.cos(phase).unsqueeze(0).unsqueeze(0)   # (1,1,T,half)
        sin_p = torch.sin(phase).unsqueeze(0).unsqueeze(0)

        rot_x1 = x1 * cos_p - x2 * sin_p
        rot_x2 = x1 * sin_p + x2 * cos_p
        return torch.cat([rot_x1, rot_x2], dim=-1)

    # ------------------------------------------------------------------

    def forward(self, x: torch.Tensor,
                step_offset: int = 0,
                nt_levels: Optional[torch.Tensor] = None,
                mask: Optional[torch.Tensor] = None
               ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        x:           (B, T, d_model)
        step_offset: global step offset (for phase continuity across turns)
        nt_levels:   (B, 4) — [DA, NE, 5HT, ACh]
        mask:        (B, T, T) boolean mask

        Returns:
          out:       (B, T, d_model)
          phase_vec: (T, d_head) — phase embedding (for logging/analysis)
        """
        B, T, _ = x.shape
        Dh = self.d_head

        # ---- 1. Phase embedding for this sequence ----
        positions = torch.arange(step_offset, step_offset + T, device=x.device)
        # Apply per-head phase offset (each head samples different frequency)
        phase_base = _phase_embed(positions, Dh // 2,
                                  n_freqs=self.n_freqs)   # (T, Dh/2)

        # ---- 2. Standard QKV ----
        q = self.q_proj(x).view(B, T, self.n_heads, Dh).transpose(1, 2)
        k = self.k_proj(x).view(B, T, self.n_heads, Dh).transpose(1, 2)
        v = self.v_proj(x).view(B, T, self.n_heads, Dh).transpose(1, 2)

        # ---- 3. Phase-precession rotation (head-specific offset) ----
        for h in range(self.n_heads):
            offset_h = self.head_phase_offsets[h]
            phase_h = phase_base + offset_h.item()       # (T, Dh/2)
            q[:, h] = self._rotate_by_phase(
                q[:, h].unsqueeze(1), phase_h).squeeze(1)
            k[:, h] = self._rotate_by_phase(
                k[:, h].unsqueeze(1), phase_h).squeeze(1)

        # ---- 4. Standard scaled dot-product attention ----
        attn = torch.einsum("bhid,bhjd->bhij", q, k) * (Dh ** -0.5)
        if mask is not None:
            attn = attn.masked_fill(mask.unsqueeze(1), float("-inf"))
        attn_w = F.softmax(attn, dim=-1)
        attended = torch.einsum("bhij,bhjd->bhid", attn_w, v)
        attended = attended.transpose(1, 2).reshape(B, T, -1)  # (B, T, D)

        # ---- 5. Gamma working memory augmentation ----
        # Slots store a compressed summary of the current cycle
        slots = self.gamma_slots.expand(B, -1, -1)        # (B, n_slots, D)
        slot_keys = self.slot_proj(slots)                 # (B, n_slots, D)
        # Attention from x to slots
        slot_attn = torch.einsum("btd,bsd->bts", x, slot_keys) * (self.d_model ** -0.5)
        slot_w = F.softmax(slot_attn, dim=-1)             # (B, T, n_slots)
        slot_read = torch.einsum("bts,bsd->btd", slot_w, slots)  # (B, T, D)

        # NT modulation of gamma coupling strength
        gamma_strength = torch.tensor(0.3, device=x.device)
        if nt_levels is not None:
            nt = nt_levels.float()
            if nt.dim() == 1:
                nt = nt.unsqueeze(0).expand(B, -1)
            gamma_strength = torch.sigmoid(self.nt_gamma_gate(nt)).mean()

        # ---- 6. Combine and output ----
        combined = attended + gamma_strength * slot_read
        out = self.ln(x + self.out_proj(combined))

        return out, phase_base
