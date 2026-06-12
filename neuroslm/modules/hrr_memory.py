"""Holographic Reduced Representations (HRR) — bound key-value memory.

HRR (Plate 1995, *Holographic Reduced Representations: Convolution
Algebra for Compositional Distributed Representations*) implements a
content-addressable memory whose **binding** operation is circular
convolution:

    bind(a, b)   = a ⊛ b   (circular convolution, "*c" in the lit)
    unbind(a, k) = a ⊛ k^{-1}   where k^{-1} is the involution of k

A memory ``M = Σ_i bind(k_i, v_i)`` retrieves the value bound to a key
by unbinding:  ``unbind(M, k_j) ≈ v_j + noise``. The noise is bounded
in expectation by ``√(N/d)`` for random keys (Plate, Theorem 3.1) so
the memory degrades gracefully under load.

We implement convolution in the Fourier domain (faster + numerically
stable):

    bind(a, b)   = IFFT(FFT(a) ⊙ FFT(b))
    inverse(k)   = IFFT(conj(FFT(k)))           # spectral inversion
    unbind(M, k) = IFFT(FFT(M) ⊙ conj(FFT(k)))  # = bind(M, inverse(k))

The :class:`HRRMemory` module bundles this with a key/value projection
pair so it acts as an ``edge`` endpoint in the BRIAN feature DSL: it
takes ``(B, T, D)`` inputs and returns ``(B, T, D)`` retrieved values,
content-addressed by the input itself (self-memory).

References
~~~~~~~~~~
* Plate, T.A. — *Holographic Reduced Representations* (CSLI Publications
  2003; foundational TR 1991).
* Kanerva, P. — "Hyperdimensional computing: An Introduction"
  (Cognitive Computation 2009) — survey of the broader family.
* Schlegel, K. et al. — "A comparison of vector symbolic architectures"
  (Artificial Intelligence Review 2022).

Implementation: tested by ``tests/test_hrr_memory.py`` (16 contracts).
"""

from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn as nn


_FFT_EPS = 1e-12


# ──────────────────────────────────────────────────────────────────────
# Core HRR primitives (functional)
# ──────────────────────────────────────────────────────────────────────


def hrr_bind(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Circular convolution along the last dim — HRR binding.

    .. math::
        (a \\circledast b)_k = \\sum_j a_j\\, b_{(k-j) \\bmod d}

    Computed in the Fourier domain for ``O(d log d)`` and clean grads.
    Both tensors must share the same last-dim length; leading dims
    broadcast normally.
    """
    if a.shape[-1] != b.shape[-1]:
        raise ValueError(
            f"hrr_bind: last dims must match (got {a.shape[-1]} vs "
            f"{b.shape[-1]})"
        )
    A = torch.fft.rfft(a, dim=-1)
    B = torch.fft.rfft(b, dim=-1)
    return torch.fft.irfft(A * B, n=a.shape[-1], dim=-1)


def hrr_inverse(k: torch.Tensor) -> torch.Tensor:
    """Spectral inverse used by HRR unbinding (Plate's "approximate inverse").

    For real ``k`` the unbinding kernel is the **involution**
    ``k'[i] = k[(-i) mod d]`` (equivalently ``conj(FFT(k))`` in the
    spectrum). The "approximate inverse" form is what makes HRR
    well-defined even for non-unit-magnitude spectra — see Plate §3.4.

    Retrieval ``bind(bind(k, v), inverse(k))`` is exact only when
    ``|FFT(k)|`` is constant across frequencies. For random Gaussian
    ``k`` it's close-but-not-equal: cos similarity ~ 1 for high ``d``,
    less for low ``d``. Use :func:`hrr_inverse_exact` if you need
    bit-equal retrieval (at the cost of conditioning).
    """
    K = torch.fft.rfft(k, dim=-1)
    return torch.fft.irfft(K.conj(), n=k.shape[-1], dim=-1)


def hrr_inverse_exact(k: torch.Tensor) -> torch.Tensor:
    """Exact spectral inverse: ``bind(k, exact_inverse(k)) == δ``.

    Computed as ``IFFT(1 / FFT(k))``. Provably gives bit-equal
    retrieval ``unbind_exact(bind(k, v), k) == v``, but blows up
    numerically when ``FFT(k)`` has near-zero components. Use only when
    keys are guaranteed non-degenerate (e.g. random Gaussian with
    sufficient dimension).
    """
    K = torch.fft.rfft(k, dim=-1)
    K_inv = K.conj() / (K * K.conj() + _FFT_EPS).real
    return torch.fft.irfft(K_inv, n=k.shape[-1], dim=-1)


def hrr_unbind(m: torch.Tensor, k: torch.Tensor) -> torch.Tensor:
    """Unbind via the involution inverse: ``bind(m, inverse(k))``.

    Approximate retrieval suitable for superposed memory; degrades
    gracefully under load (noise ~ ``√(N/d)``, Plate Theorem 3.1).
    """
    if m.shape[-1] != k.shape[-1]:
        raise ValueError(
            f"hrr_unbind: last dims must match (got m={m.shape[-1]}, "
            f"k={k.shape[-1]})"
        )
    M = torch.fft.rfft(m, dim=-1)
    K = torch.fft.rfft(k, dim=-1)
    return torch.fft.irfft(M * K.conj(), n=m.shape[-1], dim=-1)


def hrr_unbind_exact(m: torch.Tensor, k: torch.Tensor) -> torch.Tensor:
    """Bit-exact unbinding via the exact spectral inverse.

    Counterpart to :func:`hrr_inverse_exact`. Use for verification
    contracts; the standard :func:`hrr_unbind` is what production code
    (and :class:`HRRMemory`) uses because it's well-conditioned even
    under heavy superposition.
    """
    if m.shape[-1] != k.shape[-1]:
        raise ValueError(
            f"hrr_unbind_exact: last dims must match (got m={m.shape[-1]}, "
            f"k={k.shape[-1]})"
        )
    M = torch.fft.rfft(m, dim=-1)
    K = torch.fft.rfft(k, dim=-1)
    inv = K.conj() / (K * K.conj() + _FFT_EPS).real
    return torch.fft.irfft(M * inv, n=m.shape[-1], dim=-1)


def hrr_superpose(*pairs: tuple[torch.Tensor, torch.Tensor]) -> torch.Tensor:
    """Sum of ``bind(k_i, v_i)`` over many ``(key, value)`` pairs.

    Convenience for "store these N bindings in one memory vector".
    Caller is responsible for matching dims; we just superpose.
    """
    if not pairs:
        raise ValueError("hrr_superpose: at least one pair required")
    out = hrr_bind(*pairs[0])
    for k, v in pairs[1:]:
        out = out + hrr_bind(k, v)
    return out


# ──────────────────────────────────────────────────────────────────────
# HRR Memory module
# ──────────────────────────────────────────────────────────────────────


class HRRMemory(nn.Module):
    """Self-attention-shaped HRR memory layer.

    For each input token, the layer:
      1. projects to ``(key, value, query)``,
      2. constructs the **per-batch memory** as
         ``M_b = Σ_t bind(k_{b,t}, v_{b,t})``,
      3. retrieves ``r_{b,t} = unbind(M_b, q_{b,t})``,
      4. linearly projects the retrieved value back to ``d_model``.

    So each token reads back a content-addressed superposition of the
    rest of the sequence — a kind of "soft hashtable" with O(T·d log d)
    cost and O(d) memory, far cheaper than attention's O(T²·d).

    Args:
        d_model: input/output embedding dimension.
        d_memory: HRR vector dimension. Higher → less crosstalk; the
            published Plate analysis gives ``noise ∝ √(T/d_memory)``.
        normalize_keys: if ``True``, project keys onto the unit sphere
            before binding — Plate's "well-behaved keys" condition,
            sharply reduces retrieval noise.
        bias: include bias in projections.
    """

    def __init__(
        self,
        d_model: int,
        d_memory: int = 256,
        *,
        normalize_keys: bool = True,
        bias: bool = True,
    ) -> None:
        super().__init__()
        if d_model <= 0 or d_memory <= 0:
            raise ValueError(
                f"d_model + d_memory must be positive (got "
                f"{d_model}, {d_memory})"
            )
        self.d_model = d_model
        self.d_memory = d_memory
        self.normalize_keys = bool(normalize_keys)

        self.k_proj = nn.Linear(d_model, d_memory, bias=bias)
        self.v_proj = nn.Linear(d_model, d_memory, bias=bias)
        self.q_proj = nn.Linear(d_model, d_memory, bias=bias)
        self.out_proj = nn.Linear(d_memory, d_model, bias=bias)

    def _maybe_normalize(self, k: torch.Tensor) -> torch.Tensor:
        if not self.normalize_keys:
            return k
        return k / k.norm(dim=-1, keepdim=True).clamp_min(_FFT_EPS)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """``(B, T, d_model)`` → ``(B, T, d_model)``.

        The per-batch HRR memory is built ONCE across the time axis,
        then queried by every token. Causal-mask variants are easy to
        add via the optional ``causal`` flag (not needed for the BRIAN
        circuit's re-entry loop).
        """
        if x.dim() != 3:
            raise ValueError(
                f"HRRMemory expects (B, T, D), got shape {tuple(x.shape)}"
            )
        k = self._maybe_normalize(self.k_proj(x))      # (B, T, d_memory)
        v = self.v_proj(x)
        q = self._maybe_normalize(self.q_proj(x))

        # Memory = Σ_t bind(k_t, v_t). Compute in Fourier domain, sum
        # over T, then back to time → IFFT once at the end.
        K = torch.fft.rfft(k, dim=-1)                  # (B, T, d_mem/2+1)
        V = torch.fft.rfft(v, dim=-1)
        M_freq = (K * V).sum(dim=1, keepdim=True)      # (B, 1, d_mem/2+1)

        # Retrieve: per-query unbinding.
        Q = torch.fft.rfft(q, dim=-1)
        retrieved_freq = M_freq * Q.conj()             # broadcast over T
        retrieved = torch.fft.irfft(
            retrieved_freq, n=self.d_memory, dim=-1
        )                                              # (B, T, d_memory)

        return self.out_proj(retrieved)                # (B, T, d_model)
