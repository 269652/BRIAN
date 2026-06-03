# -*- coding: utf-8 -*-
"""C6 — Phase-amplitude coupling (PAC) binding probe.

The current `osc[δ θ γ]` log column is a softmax over the FFT-power
bands of a single scalar activation-magnitude signal — a noise process
that has shown no signal across the 2040-step baseline trace. To
recover the *binding hypothesis* (gamma envelope locking to theta phase
is the neural correlate of feature binding, Lisman 2013), we compute
the Tort modulation index over the same buffer:

    1. Bandpass-filter the signal (via FFT masks) into theta (slow)
       and gamma (fast) bands.
    2. Extract instantaneous phase φ_t from the theta-band signal
       (analytic-signal arg via FFT) and amplitude A_t from the
       gamma-band envelope (|analytic|).
    3. Bin φ_t into N phase bins; compute mean A within each bin.
    4. Normalise to a probability P(j); compute Tort MI:
           PAC = (H_max - H(P)) / H_max ∈ [0, 1]

PAC = 0 ⇔ amplitude is uniform across phase bins (no coupling).
PAC ≈ 1 ⇔ amplitude is concentrated in one bin (sharp coupling).

Telemetry: `pac`, `pac_pref_phase`.

If PAC rises monotonically with Φ over training, the binding
hypothesis is empirically supported in this architecture; Phase 8 can
then add a tiny reward.
"""
from __future__ import annotations
import math
from collections import deque
from typing import Dict, Optional, Tuple

import torch


def _bandpass_analytic(signal: torch.Tensor,
                       lo_frac: float,
                       hi_frac: float
                       ) -> Tuple[torch.Tensor, torch.Tensor]:
    """Return (phase, amplitude) of the analytic signal of `signal`
    restricted to the frequency band [lo_frac, hi_frac] (as fractions
    of the Nyquist).

    Uses an FFT mask + inverse FFT to construct the band-limited
    analytic signal (Hilbert-style: zero negative frequencies, double
    positives, then iFFT). All operations float32.
    """
    x = signal.float()
    x = x - x.mean()
    N = x.numel()
    if N < 4:
        zero = torch.zeros_like(x)
        return zero, zero
    X = torch.fft.fft(x)
    # Analytic-signal construction: zero negative freqs, double positives.
    H = torch.zeros(N, dtype=X.dtype)
    if N % 2 == 0:
        H[0] = 1.0
        H[N // 2] = 1.0
        H[1:N // 2] = 2.0
    else:
        H[0] = 1.0
        H[1:(N + 1) // 2] = 2.0
    # Band mask in normalised frequency [0, 1] (fraction of Nyquist).
    freqs = torch.fft.fftfreq(N) * 2.0           # in units of Nyquist
    freqs_abs = freqs.abs()
    band = ((freqs_abs >= lo_frac) & (freqs_abs <= hi_frac)).to(X.dtype)
    Y = X * H * band
    z = torch.fft.ifft(Y)
    phase = torch.atan2(z.imag, z.real)
    amplitude = z.abs()
    return phase.real, amplitude.real


class PACBindingProbe:
    """Tort modulation index over a rolling 1-D signal buffer.

    Theta and gamma bands are specified as fractions of the Nyquist —
    by default `theta=(0.05, 0.2)` and `gamma=(0.4, 0.95)`. The
    observer feeds a scalar per training step (the same one the legacy
    `OscillationTracker` consumes), and PAC is computed on the buffer.

    Parameters
    ----------
    window : int
        Buffer length. Default 128.
    n_phase_bins : int
        Number of phase bins for the Tort MI. Default 18.
    theta : (lo, hi)
        Theta band as fractions of Nyquist.
    gamma : (lo, hi)
        Gamma band as fractions of Nyquist.
    """

    def __init__(self,
                 window: int = 128,
                 n_phase_bins: int = 18,
                 theta: Tuple[float, float] = (0.05, 0.2),
                 gamma: Tuple[float, float] = (0.4, 0.95)):
        if window < 16:
            raise ValueError("window must be >= 16")
        if n_phase_bins < 4:
            raise ValueError("n_phase_bins must be >= 4")
        self.window = int(window)
        self.n_phase_bins = int(n_phase_bins)
        self.theta = (float(theta[0]), float(theta[1]))
        self.gamma = (float(gamma[0]), float(gamma[1]))
        self._buf: deque = deque(maxlen=self.window)
        self._last = {"pac": 0.0, "pac_pref_phase": 0.0}

    def observe(self, scalar: float) -> None:
        self._buf.append(float(scalar))

    def compute(self) -> Dict[str, float]:
        if len(self._buf) < max(16, self.window // 4):
            return dict(self._last)
        sig = torch.tensor(list(self._buf), dtype=torch.float32)
        phase, _ = _bandpass_analytic(sig, *self.theta)
        _, amp = _bandpass_analytic(sig, *self.gamma)
        if phase.numel() == 0 or amp.numel() == 0:
            return dict(self._last)

        # Bin phase ∈ [-π, π] into n_phase_bins.
        bins = self.n_phase_bins
        edges = torch.linspace(-math.pi, math.pi, bins + 1)
        # bucketize: returns indices in [1, bins]
        idx = torch.bucketize(phase, edges[1:-1])           # (N,) ∈ [0, bins-1]
        idx = idx.clamp(0, bins - 1)
        sums = torch.zeros(bins, dtype=torch.float32)
        counts = torch.zeros(bins, dtype=torch.float32)
        sums.scatter_add_(0, idx, amp)
        counts.scatter_add_(0, idx, torch.ones_like(amp))
        # Mean amplitude per bin (fall back to 0 for empty bins).
        mean_amp = sums / counts.clamp_min(1.0)
        total = mean_amp.sum()
        if total <= 1e-12:
            return dict(self._last)
        P = mean_amp / total
        # Tort modulation index.
        H_max = math.log(bins)
        # Avoid log(0): mask zeros.
        P_nz = P[P > 0]
        H = float(-(P_nz * P_nz.log()).sum().item()) if P_nz.numel() else H_max
        pac = max(0.0, min(1.0, (H_max - H) / H_max))
        # Preferred phase: argmax bin centre.
        pref_idx = int(mean_amp.argmax().item())
        bin_centres = 0.5 * (edges[:-1] + edges[1:])
        pref_phase = float(bin_centres[pref_idx].item())
        self._last = {"pac": float(pac), "pac_pref_phase": pref_phase}
        return dict(self._last)

    def stats(self) -> Dict[str, float]:
        return dict(self._last)
