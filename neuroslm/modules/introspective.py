"""Self-Reflective / Introspective Module for NeuroSLM

Periodically summarizes internal activations and feeds them back as context.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

class IntrospectiveModule(nn.Module):
    def __init__(self, d_model: int, summary_dim: int = 64):
        super().__init__()
        self.summary_proj = nn.Linear(d_model, summary_dim)
        self.feedback_proj = nn.Linear(summary_dim, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (B, T, d_model)
        """
        # Summarize activations (mean pooling)
        summary = self.summary_proj(x.mean(dim=1))  # (B, summary_dim)
        # Feed back as context
        feedback = self.feedback_proj(summary).unsqueeze(1)  # (B, 1, d_model)
        # Broadcast and add to input
        x = x + feedback
        return x, summary
