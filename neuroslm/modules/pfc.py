"""Prefrontal Cortex: NT-driven executive candidate selection.

Implements a biologically-grounded scoring algorithm where neurotransmitter
levels directly gate which class of candidates gets prioritised:

  Serotonin (5HT) high  → calming / safe / prosocial candidates
  Serotonin (5HT) low   → threat-rumination bias
  Norepinephrine (NE) high → urgent / threat-relevant / salient candidates
  Norepinephrine (NE) moderate → optimal attentional filtering (Yerkes-Dodson)
  Dopamine (DA) high    → reward-seeking / exploratory / novel candidates
  Dopamine (DA) low     → anhedonia / perseveration / conservative bias
  Acetylcholine (ACh) high → detail / focus / signal-to-noise improvement
  GABA high             → global inhibitory tone (down-weights all)
  Glutamate (Glu) high  → excitatory gain (up-weights all)

Architecture:
  - Transformer encoder integrates all candidates (GWS slots + hippo recalls + thought)
  - Four NT-specific projection heads score candidates on orthogonal dimensions:
      reward_proj   (scored by DA)
      safety_proj   (scored by 5HT)
      urgency_proj  (scored by NE, inverted-U shaped)
      focus_proj    (scored by ACh)
  - Final score = transformer logit + NT-weighted sum of dimension scores
  - Inhibitory gate (GABA-style lateral inhibition) suppresses weak candidates
  - Working memory: selected thought is blended into floating thought with DA-gated rate

References:
  Goldman-Rakic (1995) Cellular basis of working memory.
  Arnsten (2011) Catecholamine influences on PFC circuits.
  Arnsten (1998) Catecholamine modulation of PFC — inverted-U (NE).
  Cohen et al. (2002) Serotonin and decision-making.
"""
from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F

from .brain_module import BrainModule


class PrefrontalCortex(BrainModule):
    def __init__(self, d_sem: int, n_layers: int, n_heads: int):
        super().__init__()
        self.d_sem = d_sem

        # Transformer encoder: integrates slots + recalls + thought
        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_sem, nhead=n_heads,
            dim_feedforward=d_sem * 4,
            batch_first=True, activation="gelu", norm_first=True,
            dropout=0.0,
        )
        self.transformer = nn.TransformerEncoder(enc_layer, num_layers=n_layers)

        # Learnable selection query — the PFC's "executive pointer"
        self.select_query = nn.Parameter(torch.randn(1, 1, d_sem) * 0.02)

        # --- NT-specific scoring heads ---
        # Each maps a candidate (d_sem) → scalar score on one dimension
        self.reward_proj   = nn.Linear(d_sem, 1)   # DA domain
        self.safety_proj   = nn.Linear(d_sem, 1)   # 5HT domain
        self.urgency_proj  = nn.Linear(d_sem, 1)   # NE domain
        self.focus_proj    = nn.Linear(d_sem, 1)   # ACh domain

        # Base scoring: transformer CLS token → candidate affinity
        self.base_score    = nn.Linear(d_sem, 1)

        # Lateral inhibition gate (GABA-style: suppress non-winners)
        self.inhibition    = nn.Linear(d_sem, d_sem)

        # Replace gate: probability of overwriting floating thought
        self.replace_head  = nn.Linear(d_sem, 1)

        # Working-memory gating: DA-conditioned blend rate
        self.wm_gate       = nn.Linear(d_sem + 1, 1)  # d + DA_level → blend_rate

    # ------------------------------------------------------------------
    # NT-driven scoring
    # ------------------------------------------------------------------
    def _nt_scores(self, candidates: torch.Tensor,
                   nt_levels: dict) -> torch.Tensor:
        """Compute NT-weighted candidate scores.

        candidates: (B, K, d_sem)
        Returns:    (B, K) additive score bias
        """
        da  = nt_levels.get("DA",   0.5)
        ht  = nt_levels.get("5HT",  0.5)
        ne  = nt_levels.get("NE",   0.5)
        ach = nt_levels.get("ACh",  0.5)
        gaba = nt_levels.get("GABA", 0.5)
        glu  = nt_levels.get("Glu",  0.5)

        # Per-candidate dimension scores (B, K, 1) → squeeze → (B, K)
        r_s = self.reward_proj(candidates).squeeze(-1)   # reward dim
        s_s = self.safety_proj(candidates).squeeze(-1)   # safety dim
        u_s = self.urgency_proj(candidates).squeeze(-1)  # urgency dim
        f_s = self.focus_proj(candidates).squeeze(-1)    # focus dim

        # --- DA (dopamine) ---
        # High DA (D1-mediated): boost reward / novelty seeking
        # Low DA: boost conservative / default
        da_bias = 2.5 * (da - 0.5) * r_s        # [-1.25, +1.25] * reward_score

        # --- 5HT (serotonin) ---
        # High 5HT: safe/calming bias, strong threat suppression
        # Low 5HT: negative/rumination bias (- safety, + urgency)
        ht_bias = (2.5 * ht - 0.5) * s_s        # scaled safety bias

        # --- NE (norepinephrine) — inverted-U (Yerkes-Dodson) ---
        # Moderate NE (~0.5) = optimal attention; extreme = degraded
        # Formula: peak at NE=0.6, steep falloff for NE>0.8
        ne_urgency = ne * (1.5 - ne)             # inverted parabola, peak at 0.75
        ne_bias = 2.0 * ne_urgency * u_s         # urgency bias, peaks at moderate NE

        # When NE is very high (>0.75): threat filter — strongly suppress
        # non-urgent candidates (survival mode attention narrowing)
        if ne > 0.75:
            threat_suppression = -1.5 * (1.0 - u_s.clamp(-1, 1))
            ne_bias = ne_bias + threat_suppression * (ne - 0.75) / 0.25

        # --- ACh (acetylcholine) ---
        # High ACh: sharpen attention to detail / focused candidates
        ach_bias = 1.5 * ach * f_s

        # --- GABA: global inhibitory tone ---
        # Suppresses all candidates equally (reduces noise)
        gaba_bias = -0.4 * gaba

        # --- Glu: excitatory gain ---
        # Amplifies all scores (excitatory state)
        glu_gain = 1.0 + 0.3 * glu

        nt_score = (da_bias + ht_bias + ne_bias + ach_bias + gaba_bias) * glu_gain
        return nt_score   # (B, K)

    # ------------------------------------------------------------------
    # Forward pass
    # ------------------------------------------------------------------
    def forward(self, gws_slots: torch.Tensor,
                recalls: torch.Tensor,
                floating_thought: torch.Tensor,
                nt_levels: dict | None = None) -> tuple[torch.Tensor, torch.Tensor]:
        """
        gws_slots:       (B, S, d_sem)
        recalls:         (B, R, d_sem)
        floating_thought:(B, d_sem)
        nt_levels:       dict of NT scalars

        Returns:
            selected      (B, d_sem) — executive selection
            replace_gate  (B,)       — probability of replacing floating thought
        """
        if nt_levels is None:
            nt_levels = {}
        B = gws_slots.size(0)
        ft = floating_thought.unsqueeze(1)             # (B, 1, d)
        q  = self.select_query.expand(B, -1, -1)       # (B, 1, d)

        # Concatenate all inputs: [exec_query, gws_slots, recalls, floating_thought]
        x = torch.cat([q, gws_slots, recalls, ft], dim=1)   # (B, total_seq, d)
        y = self.transformer(x)                              # (B, total_seq, d)

        # Candidates = everything except the executive query token
        candidates = y[:, 1:]                                # (B, K, d)

        # Base scores from executive pointer attention
        base = self.base_score(candidates).squeeze(-1)       # (B, K)

        # NT-driven scoring
        nt_score = self._nt_scores(candidates, nt_levels)    # (B, K)

        total_score = base + nt_score                        # (B, K)

        # GABA-style lateral inhibition: suppress below-average candidates
        gaba = nt_levels.get("GABA", 0.5)
        if gaba > 0.4:
            threshold = total_score.mean(dim=-1, keepdim=True)
            suppression = torch.where(
                total_score < threshold,
                torch.sigmoid(self.inhibition(candidates)).mean(-1) * gaba * -0.5,
                torch.zeros_like(total_score))
            total_score = total_score + suppression

        # Select: softmax-weighted aggregation (soft winner-take-all)
        ne = nt_levels.get("NE", 0.5)
        temperature = max(0.3, 1.5 - ne)  # High NE → sharper selection
        weights = F.softmax(total_score / temperature, dim=-1)   # (B, K)
        selected = (weights.unsqueeze(-1) * candidates).sum(1)   # (B, d)

        # Working-memory gate: DA-conditioned blend rate for floating thought
        da  = nt_levels.get("DA", 0.5)
        da_t = torch.tensor([da], device=selected.device, dtype=selected.dtype)
        da_t = da_t.unsqueeze(0).expand(B, -1)
        wm_in = torch.cat([selected, da_t], dim=-1)
        replace_gate = torch.sigmoid(
            self.replace_head(selected) + self.wm_gate(wm_in) * 0.3
        ).squeeze(-1)   # (B,)

        return selected, replace_gate

    def _disabled_output(self, gws_slots, _recalls, floating_thought, *_, **__):
        B = gws_slots.size(0)
        return floating_thought, torch.zeros(B, device=gws_slots.device)
