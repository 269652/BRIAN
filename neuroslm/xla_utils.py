"""XLA / TPU utility helpers for NeuroSLM.

Provides a unified device-selection layer so the rest of the codebase
never calls torch.cuda directly.  When torch_xla is present the code
runs on TPU; otherwise it falls back to CUDA or CPU transparently.

Usage
-----
from neuroslm.xla_utils import get_device, is_xla, mark_step, to_bfloat16

device = get_device()          # xla:0 | cuda | cpu
model  = to_bfloat16(model)    # bf16 for XLA/CUDA; fp32 on CPU
...
mark_step()                    # XLA graph flush (no-op on CUDA/CPU)
"""
from __future__ import annotations
import os
import torch
import torch.nn as nn

# ---------------------------------------------------------------------------
# XLA availability detection (import once, cache the result)
# ---------------------------------------------------------------------------
_XLA_AVAILABLE = False
_xla_device_fn = None  # callable() -> torch.device

try:
    # torch_xla 2.x new API
    import torch_xla  # type: ignore[import]
    import torch_xla.core.xla_model as xm  # type: ignore[import]
    _XLA_AVAILABLE = True

    # torch_xla ≥ 2.1 exposes torch_xla.device(); older builds only have xm.xla_device()
    if hasattr(torch_xla, "device"):
        _xla_device_fn = torch_xla.device
    else:
        _xla_device_fn = xm.xla_device  # type: ignore[assignment]

except ImportError:
    xm = None  # type: ignore[assignment]


def is_xla() -> bool:
    """True when running on a TPU via torch_xla."""
    return _XLA_AVAILABLE


def get_device() -> torch.device:
    """Return the best available device.

    Priority: XLA (TPU) > CUDA > CPU
    Respects the NEUROSLM_DEVICE environment variable when set.
    """
    env = os.environ.get("NEUROSLM_DEVICE", "").strip()
    if env:
        return torch.device(env)
    if _XLA_AVAILABLE and _xla_device_fn is not None:
        return _xla_device_fn()
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def mark_step() -> None:
    """Flush the XLA computation graph.

    On TPU this triggers the async execution of all pending XLA ops.
    On CUDA / CPU this is a no-op.  Call once per optimizer step.
    """
    if _XLA_AVAILABLE and xm is not None:
        xm.mark_step()


def optimizer_step(optimizer: torch.optim.Optimizer,
                   parameters=None, max_norm: float | None = None) -> float:
    """Perform a gradient-clipped optimizer step, XLA-aware.

    Returns the gradient norm (0.0 on XLA where clipping is done by xm).
    """
    if max_norm is not None:
        gnorm = torch.nn.utils.clip_grad_norm_(
            parameters or [], max_norm).item()
    else:
        gnorm = 0.0

    if _XLA_AVAILABLE and xm is not None:
        xm.optimizer_step(optimizer, barrier=False)
    else:
        optimizer.step()
    return gnorm


def should_use_gradient_checkpointing() -> bool:
    """Gradient checkpointing is recommended on both XLA and CUDA."""
    return _XLA_AVAILABLE or torch.cuda.is_available()


def native_dtype() -> torch.dtype:
    """bfloat16 for XLA (TPU native) and CUDA (Ampere+); fp32 for CPU."""
    if _XLA_AVAILABLE:
        return torch.bfloat16
    if torch.cuda.is_available():
        return torch.bfloat16
    return torch.float32


def to_bfloat16(model: nn.Module) -> nn.Module:
    """Cast model parameters to the native precision for the active backend.

    On XLA / CUDA: bfloat16  (half memory, same dynamic range as fp32).
    On CPU: fp32 (bfloat16 is slow on CPU).
    """
    dtype = native_dtype()
    if dtype != torch.float32:
        model = model.to(dtype)
    return model


# ---------------------------------------------------------------------------
# Data-loader shim.
#
# MpDeviceLoader requires a torch.utils.data.DataLoader — NOT a plain Python
# generator.  Our batch_iterator is a plain generator, so we always use the
# lightweight _IdentityLoader which just calls .to(device) per batch.
# On XLA the tensors are placed on the TPU via the same .to(xla_device) path.
# ---------------------------------------------------------------------------

class _IdentityLoader:
    """Wraps any iterable and moves each tensor batch to the target device."""
    def __init__(self, iterator, device):
        self._it = iterator
        self._device = device

    def __iter__(self):
        return self

    def __next__(self):
        batch = next(self._it)
        return batch.to(self._device)


def make_loader(iterator, device) -> _IdentityLoader:
    """Wrap an iterator so every batch lands on *device*.

    We use _IdentityLoader for both XLA and CUDA/CPU because MpDeviceLoader
    requires a torch.utils.data.DataLoader object (not a plain generator).
    The identity loader is simpler, equally correct, and avoids threading
    issues with the reconnecting stream iterator.
    """
    return _IdentityLoader(iterator, device)
