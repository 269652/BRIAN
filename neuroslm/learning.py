from __future__ import annotations
import torch
import torch.nn as nn


class LearningLayer(nn.Module):
    """A small trainable module that observes a few brain signals and
    produces a scalar learning-rate multiplier for the optimizer.

    This is a coarse "learn-to-learn" mechanism: instead of rewriting
    backprop, it learns to gate/scale learning online based on
    neuromodulatory state. It's intentionally lightweight so it can be
    trained together with the rest of the model.
    """
    def __init__(self, n_inputs: int = 8, hidden: int = 32, init_scale: float = 1.0):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_inputs, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden // 2),
            nn.ReLU(),
            nn.Linear(hidden // 2, 1),
        )
        # initialize last layer small so initial multiplier is near 1
        nn.init.zeros_(self.net[-1].weight)
        nn.init.constant_(self.net[-1].bias, float(init_scale - 1.0))

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """inputs: (B, n_inputs) or (n_inputs,) tensor of floats.
        Returns: scalar multiplier (float tensor) clipped to reasonable range.
        """
        if inputs.dim() == 1:
            x = inputs.unsqueeze(0)
        else:
            x = inputs
        out = self.net(x).squeeze(-1)
        # map to positive multiplier. use 1 + tanh(raw) for bounded, centered at 1
        mult = 1.0 + torch.tanh(out)
        # clamp to [0.1, 3.0] to avoid extreme learning rates
        mult = mult.clamp(0.1, 3.0)
        return mult
