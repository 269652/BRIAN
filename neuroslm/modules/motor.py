"""Motor cortex: turns a chosen BG action embedding into both:
  (a) a thought vector that conditions the language cortex's next-token logits
  (b) a discrete *motor action* drawn from a small action vocabulary

Action vocabulary (modeled loosely on goal-directed behavior + survival):
  SPEAK         — emit the next token (default during generation)
  REMAIN_SILENT — skip emitting; the brain keeps thinking
  RECALL        — query hippocampus more aggressively next tick
  PLAN          — invoke an extra forward-model rollout before acting
  FLEE          — survival action: bias all subsequent actions away from threat
"""
from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F


ACTION_NAMES = ("SPEAK", "REMAIN_SILENT", "RECALL", "PLAN", "FLEE")
N_ACTIONS = len(ACTION_NAMES)
ACTION_INDEX = {n: i for i, n in enumerate(ACTION_NAMES)}


class MotorCortex(nn.Module):
    def __init__(self, d_action: int, d_sem: int, d_hidden: int | None = None):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(d_action, d_sem),
            nn.GELU(),
            nn.Linear(d_sem, d_sem),
        )
        # Motor bias on language cortex hidden state (M1 → vocal pathway)
        # Zero-initialized so the model starts as a pure LM and learns to use
        # this channel only when it helps.
        d_hidden = d_hidden or d_sem
        self.to_lang_bias = nn.Linear(d_action, d_hidden)
        nn.init.zeros_(self.to_lang_bias.weight)
        nn.init.zeros_(self.to_lang_bias.bias)

        # Discrete-action head — picks SPEAK / REMAIN_SILENT / etc.
        self.action_head = nn.Linear(d_action, N_ACTIONS)
        # Bias SPEAK by default so an untrained model still emits tokens.
        with torch.no_grad():
            self.action_head.bias.zero_()
            self.action_head.bias[ACTION_INDEX["SPEAK"]] = 2.0

    def forward(self, action_emb: torch.Tensor,
                survival: torch.Tensor | None = None):
        """action_emb: (B, d_action). survival: optional (B,) bool — when True
        FLEE is forced and SPEAK is suppressed.
        Returns: (thought (B,d_sem), lang_bias (B,d_hidden),
                  action_idx (B,), action_logits (B,N), action_probs (B,N))
        """
        thought = self.proj(action_emb)
        lang_bias = self.to_lang_bias(action_emb)
        logits = self.action_head(action_emb)
        if survival is not None:
            mask = survival.unsqueeze(-1).float()
            override = torch.zeros_like(logits)
            override[:, ACTION_INDEX["FLEE"]] = 5.0
            override[:, ACTION_INDEX["REMAIN_SILENT"]] = 1.0
            override[:, ACTION_INDEX["SPEAK"]] = -5.0
            logits = logits * (1 - mask) + override * mask
        probs = F.softmax(logits, dim=-1)
        idx = probs.argmax(dim=-1)
        return thought, lang_bias, idx, logits, probs

    @staticmethod
    def name(idx) -> str:
        return ACTION_NAMES[int(idx)]
