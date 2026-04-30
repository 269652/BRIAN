"""Minimal LoRA helper: inject low-rank adapters into Linear modules.

This is a lightweight implementation intended for quick experiments. For
production use, consider using an established library (peft, bitsandbytes + adapters).
"""
from __future__ import annotations
import torch
import torch.nn as nn
import math
from typing import List


class LoRALayer(nn.Module):
    def __init__(self, orig: nn.Linear, r: int = 4, alpha: float = 16.0):
        super().__init__()
        self.orig = orig
        self.r = r
        self.alpha = alpha
        if r > 0:
            self.lora_a = nn.Parameter(torch.zeros((r, orig.in_features)))
            self.lora_b = nn.Parameter(torch.zeros((orig.out_features, r)))
            # scale
            self.scaling = alpha / r
            nn.init.kaiming_uniform_(self.lora_a, a=math.sqrt(5))
            nn.init.zeros_(self.lora_b)
        else:
            self.lora_a = None
            self.lora_b = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.orig(x)
        if self.r > 0:
            # LoRA delta: x @ A.T -> (B, r), B @ B.T -> (B, out)
            delta = (x @ self.lora_a.t()) @ self.lora_b.t()
            out = out + self.scaling * delta
        return out


def apply_lora_to_module(module: nn.Module, target_names: List[str] = None, r: int = 4, alpha: float = 16.0):
    """Recursively replace Linear modules named in `target_names` with LoRA-wrapped ones.
    If target_names is None, wrap all Linear layers.
    """
    for name, child in list(module.named_children()):
        if isinstance(child, nn.Linear) and (target_names is None or name in (target_names or [])):
            setattr(module, name, LoRALayer(child, r=r, alpha=alpha))
        else:
            apply_lora_to_module(child, target_names=target_names, r=r, alpha=alpha)
