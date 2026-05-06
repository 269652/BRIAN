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
try:
    import torch_xla.core.xla_model as xm          # type: ignore[import]
    _XLA_AVAILABLE = True
except ImportError:
    xm = None                                        # type: ignore[assignment]
    _XLA_AVAILABLE = False


def is_xla() -> bool:
    """True when running on a TPU via torch_xla."""
    return _XLA_AVAILABLE


def get_device() -> str | torch.device:
    """Return the best available device.

    Priority: XLA (TPU) > CUDA > CPU
    Respects the NEUROSLM_DEVICE environment variable when set.
    """
    env = os.environ.get("NEUROSLM_DEVICE", "").strip()
    if env:
        return torch.device(env)
    if _XLA_AVAILABLE:
        return xm.xla_device()
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def mark_step() -> None:
    """Flush the XLA computation graph.

    On TPU this triggers the async execution of all pending XLA ops.
    On CUDA / CPU this is a no-op.  Call once per optimizer step.
    """
    if _XLA_AVAILABLE:
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

    if _XLA_AVAILABLE:
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
# Data-loader shim — wraps a plain iterator with XLA's ParallelLoader when
# running on TPU so that the data pipeline shadows compute on all cores.
# ---------------------------------------------------------------------------

class _IdentityLoader:
    """Thin wrapper that makes a plain iterator look like an XLA loader."""
    def __init__(self, iterator, device):
        self._it = iterator
        self._device = device

    def __iter__(self):
        for batch in self._it:
            yield batch.to(self._device)


def make_loader(iterator, device) -> _IdentityLoader:
    """Wrap an iterator for the given device.

    On XLA: uses MpDeviceLoader so the host pre-fetches data onto TPU cores
    in parallel with computation, hiding H2D transfer latency.
    On CUDA/CPU: returns a lightweight wrapper that calls .to(device) per batch.
    """
    if _XLA_AVAILABLE:
        from torch_xla.distributed.parallel_loader import MpDeviceLoader  # type: ignore
        return MpDeviceLoader(iterator, device)
    return _IdentityLoader(iterator, device)
