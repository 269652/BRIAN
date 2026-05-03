"""Neural Oscillation Metrics — multi-band spectral analysis of neural activity.

Tracks oscillatory activity across biologically-inspired frequency bands,
computed from the hidden state dynamics of the model. These are analogous
to EEG bands measured in neuroscience:

  Delta  (0.5-4 Hz):  Deep processing, consolidation
  Theta  (4-8 Hz):    Memory encoding/retrieval, navigation
  Alpha  (8-12 Hz):   Idle/inhibition, top-down control
  Beta   (12-30 Hz):  Motor planning, active maintenance
  Gamma  (30-100 Hz): Binding, conscious perception, attention

In our model, "frequency" is measured in ticks (forward passes), not
real time. We compute spectral power from the temporal evolution of
module activations using a sliding window FFT.

Scientific basis:
  - Buzsáki & Draguhn (2004): Neuronal oscillations in cortical networks
  - Jensen & Mazaheri (2010): Alpha as pulsed inhibition
  - Fries (2015): Rhythms for cognition — communication through coherence
"""
from __future__ import annotations
import torch
import torch.nn as nn
import math
from dataclasses import dataclass, field


@dataclass
class OscillationSnapshot:
    """Single timepoint of oscillation measurements."""
    delta: float = 0.0     # slow: consolidation, deep processing
    theta: float = 0.0     # memory encoding/retrieval
    alpha: float = 0.0     # idle/inhibition
    beta: float = 0.0      # active maintenance, motor prep
    gamma: float = 0.0     # binding, attention, conscious access

    # Cross-frequency coupling
    theta_gamma_coupling: float = 0.0   # memory-attention interaction
    alpha_beta_ratio: float = 0.0       # inhibition vs activation balance

    # Global metrics
    dominant_band: str = 'alpha'
    spectral_entropy: float = 0.0  # uniformity of spectral power
    neural_synchrony: float = 0.0  # cross-region phase alignment

    def as_dict(self) -> dict:
        return {
            'delta': self.delta,
            'theta': self.theta,
            'alpha': self.alpha,
            'beta': self.beta,
            'gamma': self.gamma,
            'theta_gamma_coupling': self.theta_gamma_coupling,
            'alpha_beta_ratio': self.alpha_beta_ratio,
            'dominant_band': self.dominant_band,
            'spectral_entropy': self.spectral_entropy,
            'neural_synchrony': self.neural_synchrony,
        }

    def format(self) -> str:
        return (f"δ={self.delta:.3f} θ={self.theta:.3f} α={self.alpha:.3f} "
                f"β={self.beta:.3f} γ={self.gamma:.3f} | "
                f"θ-γ={self.theta_gamma_coupling:.3f} "
                f"sync={self.neural_synchrony:.3f} "
                f"dom={self.dominant_band}")


class NeuralOscillationTracker(nn.Module):
    """Tracks multi-band neural oscillations from hidden state dynamics.

    Records the temporal evolution of activations and computes spectral
    power in each frequency band using a sliding window FFT.
    """

    def __init__(self, d_model: int, n_regions: int = 8,
                 window_size: int = 64, hop_size: int = 1):
        super().__init__()
        self.d_model = d_model
        self.n_regions = n_regions
        self.window_size = window_size
        self.hop_size = hop_size

        # Circular buffer of region activations over time
        # Shape: (n_regions, window_size, d_model)
        self.register_buffer(
            '_activity_buffer',
            torch.zeros(n_regions, window_size, d_model))
        self.register_buffer('_write_idx', torch.zeros(1, dtype=torch.long))
        self.register_buffer('_total_ticks', torch.zeros(1, dtype=torch.long))

        # Band definitions (in normalized frequency, 0-0.5)
        # Since our "sample rate" is 1 tick, Nyquist = 0.5
        # We map brain frequencies to tick-frequencies:
        #   delta ≈ slowest variations (bins 1-2)
        #   theta ≈ slow oscillations (bins 2-4)
        #   alpha ≈ medium (bins 4-6)
        #   beta  ≈ medium-fast (bins 6-12)
        #   gamma ≈ fast (bins 12+)
        # These map to FFT bin indices for a window of size W
        self.band_bins = {}  # set in _compute_band_bins

        # Region name mapping
        self._region_names: list[str] = []

        # History of snapshots for trend analysis
        self._history: list[OscillationSnapshot] = []
        self._max_history = 1000

    def register_regions(self, names: list[str]):
        """Register region names for tracking."""
        self._region_names = names[:self.n_regions]

    @torch.no_grad()
    def record(self, region_idx: int, activation: torch.Tensor):
        """Record a region's activation at the current tick.

        Args:
            region_idx: index of the region (0..n_regions-1)
            activation: (B, D) or (D,) — mean-pooled activation
        """
        if region_idx >= self.n_regions:
            return
        if activation.dim() > 1:
            activation = activation.mean(0)
        # Truncate/pad to d_model
        d = min(activation.shape[0], self.d_model)
        idx = int(self._write_idx.item()) % self.window_size
        self._activity_buffer[region_idx, idx, :d] = activation[:d].detach()

    @torch.no_grad()
    def tick(self):
        """Advance the tick counter (call once per forward pass)."""
        self._write_idx += 1
        self._total_ticks += 1

    @torch.no_grad()
    def compute_spectrum(self) -> OscillationSnapshot:
        """Compute the current oscillation spectrum across all regions.

        Returns an OscillationSnapshot with power in each band.
        """
        W = self.window_size
        n_ticks = int(self._total_ticks.item())

        if n_ticks < 4:
            # Not enough data yet
            snap = OscillationSnapshot()
            self._history.append(snap)
            return snap

        # Use however many ticks we have (up to window_size)
        effective_w = min(n_ticks, W)

        # Compute per-region spectral power
        # We analyze the mean activation magnitude over time
        band_powers = {'delta': [], 'theta': [], 'alpha': [],
                       'beta': [], 'gamma': []}

        for r in range(min(len(self._region_names), self.n_regions)):
            # Get the activity trace: (effective_w, d_model)
            idx = int(self._write_idx.item())
            if n_ticks >= W:
                # Full window: circular buffer order
                start = idx % W
                trace = torch.cat([
                    self._activity_buffer[r, start:],
                    self._activity_buffer[r, :start]
                ], dim=0)[:effective_w]
            else:
                trace = self._activity_buffer[r, :effective_w]

            # Compute RMS activation per tick: (effective_w,)
            rms = trace.pow(2).mean(dim=-1).sqrt()

            # Detrend (remove mean)
            rms = rms - rms.mean()

            if effective_w < 4:
                continue

            # FFT
            spectrum = torch.fft.rfft(rms)
            power = spectrum.abs().pow(2)

            # Band extraction (map to FFT bins)
            n_bins = power.shape[0]
            # delta: bins 1-2, theta: 2-4, alpha: 4-8, beta: 8-16, gamma: 16+
            def band_power(low_bin, high_bin):
                lo = max(1, min(low_bin, n_bins - 1))
                hi = max(lo + 1, min(high_bin, n_bins))
                return power[lo:hi].mean().item() if hi > lo else 0.0

            band_powers['delta'].append(band_power(1, 3))
            band_powers['theta'].append(band_power(3, 5))
            band_powers['alpha'].append(band_power(5, 9))
            band_powers['beta'].append(band_power(9, 17))
            band_powers['gamma'].append(band_power(17, n_bins))

        # Average across regions
        def avg(lst):
            return sum(lst) / max(len(lst), 1)

        delta = avg(band_powers['delta'])
        theta = avg(band_powers['theta'])
        alpha = avg(band_powers['alpha'])
        beta = avg(band_powers['beta'])
        gamma = avg(band_powers['gamma'])

        # Normalize to [0, 1] range
        total = delta + theta + alpha + beta + gamma + 1e-8
        delta /= total
        theta /= total
        alpha /= total
        beta /= total
        gamma /= total

        # Cross-frequency coupling: theta-gamma
        # High coupling = memory-attention interaction
        theta_gamma = min(1.0, 2.0 * (theta * gamma) ** 0.5)

        # Alpha-beta ratio: inhibition vs activation
        alpha_beta = alpha / (beta + 1e-8)

        # Dominant band
        bands = {'delta': delta, 'theta': theta, 'alpha': alpha,
                 'beta': beta, 'gamma': gamma}
        dominant = max(bands, key=bands.get)

        # Spectral entropy (how uniform is the distribution)
        probs = torch.tensor([delta, theta, alpha, beta, gamma])
        probs = probs / probs.sum().clamp(min=1e-8)
        spectral_entropy = -(probs * (probs + 1e-8).log()).sum().item()
        max_entropy = math.log(5)
        spectral_entropy /= max_entropy  # normalize to [0, 1]

        # Neural synchrony: cross-region phase coherence
        synchrony = self._compute_synchrony()

        snap = OscillationSnapshot(
            delta=delta, theta=theta, alpha=alpha,
            beta=beta, gamma=gamma,
            theta_gamma_coupling=theta_gamma,
            alpha_beta_ratio=alpha_beta,
            dominant_band=dominant,
            spectral_entropy=spectral_entropy,
            neural_synchrony=synchrony,
        )

        if len(self._history) >= self._max_history:
            self._history = self._history[-self._max_history // 2:]
        self._history.append(snap)

        return snap

    @torch.no_grad()
    def _compute_synchrony(self) -> float:
        """Compute cross-region phase synchrony (PLV-inspired).

        Phase-locking value between region pairs, averaged.
        """
        n_active = min(len(self._region_names), self.n_regions)
        if n_active < 2:
            return 0.0

        n_ticks = int(self._total_ticks.item())
        effective_w = min(n_ticks, self.window_size)
        if effective_w < 4:
            return 0.0

        # Get RMS traces per region
        traces = []
        idx = int(self._write_idx.item())
        for r in range(n_active):
            if n_ticks >= self.window_size:
                start = idx % self.window_size
                trace = torch.cat([
                    self._activity_buffer[r, start:],
                    self._activity_buffer[r, :start]
                ], dim=0)[:effective_w]
            else:
                trace = self._activity_buffer[r, :effective_w]
            rms = trace.pow(2).mean(dim=-1).sqrt()
            traces.append(rms)

        # Pairwise correlation as synchrony proxy
        correlations = []
        for i in range(len(traces)):
            for j in range(i + 1, len(traces)):
                a = traces[i] - traces[i].mean()
                b = traces[j] - traces[j].mean()
                denom = (a.pow(2).sum() * b.pow(2).sum()).sqrt()
                if denom > 1e-8:
                    corr = (a * b).sum() / denom
                    correlations.append(abs(corr.item()))

        return sum(correlations) / max(len(correlations), 1)

    def trend(self, last_n: int = 50) -> dict:
        """Get trends in oscillation bands over recent history."""
        if len(self._history) < 2:
            return {}
        recent = self._history[-last_n:]
        n = len(recent)
        half = n // 2
        first_half = recent[:half]
        second_half = recent[half:]

        def avg_band(snaps, band):
            return sum(getattr(s, band) for s in snaps) / max(len(snaps), 1)

        trends = {}
        for band in ['delta', 'theta', 'alpha', 'beta', 'gamma']:
            early = avg_band(first_half, band)
            late = avg_band(second_half, band)
            trends[f'{band}_trend'] = late - early
        return trends
