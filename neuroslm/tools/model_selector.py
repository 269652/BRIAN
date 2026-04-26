"""Utilities to pick a feasible model/training strategy for a given GPU.

Goal: on a 14GB Colab GPU allow variable effective parameter sizes between
~400M and larger via quantization + LoRA. This module provides a heuristic
selector that maps available VRAM to a recommended strategy.
"""
from __future__ import annotations
import math
import torch


def detect_gpu_memory_gb(device: int = 0) -> float:
    try:
        if not torch.cuda.is_available():
            return 0.0
        prop = torch.cuda.get_device_properties(device)
        return prop.total_memory / (1024 ** 3)
    except Exception:
        return 0.0


def pick_strategy_for_vram(mem_gb: float) -> dict:
    """Return a dict with recommended strategy and approximate capacity.

    Strategies (heuristic):
      - dense: full dense model trained/fine-tuned in fp16/bf16. Feasible up to
        ~1.5B on 14GB without sharding.
      - quant_lora: use 4-bit or 8-bit quantization (bitsandbytes) + LoRA.
        Allows effective parameter counts up to several billion on 14GB.
      - offload: use quantization + CPU offload / accelerate to push beyond.

    The returned dict includes:
      - strategy: 'dense' | 'quant_lora' | 'quant_lora_offload'
      - target_params: approximate effective parameter scale (in billions)
      - notes: human-readable guidance
    """
    if mem_gb <= 0:
        return {"strategy": "cpu_only", "target_params": 0.05,
                "notes": "No CUDA: use tiny / LoRA on CPU or run in Colab."}

    # Heuristics tuned to Colab-like GPUs
    if mem_gb < 8:
        return {
            "strategy": "quant_lora",
            "target_params": 0.4,  # ~400M effective via quant+LoRA
            "notes": "Small GPUs: use 4/8-bit quantization + LoRA; limit batch size."
        }
    if mem_gb < 16:
        # 14GB Colab target
        return {
            "strategy": "quant_lora",
            "target_params": 2.0,  # effective up to ~2B with 4-bit + LoRA
            "notes": (
                "Use 4-bit quantization (bitsandbytes) + LoRA adapters. "
                "Dense models >1.5B will require quantization or offload."
            )
        }
    if mem_gb < 32:
        return {
            "strategy": "quant_lora_offload",
            "target_params": 7.0,
            "notes": "Use quantization + CPU/GPU offload to push to ~7B."
        }

    # Large GPUs
    return {
        "strategy": "quant_lora_offload",
        "target_params": 10.0,
        "notes": "Large GPU: can experiment up to ~10B with sharding & offload."
    }


def pick_for_current_gpu() -> dict:
    mem = detect_gpu_memory_gb()
    return {"vram_gb": mem, **pick_strategy_for_vram(mem)}


if __name__ == '__main__':
    print(pick_for_current_gpu())
