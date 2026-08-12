# -*- coding: utf-8 -*-
"""Sensory cortices (§15): raw multi-modal signal -> latent embedding,
never text. Three modalities, one convention each: a real, checkable
front-end (exact DSP for audio, a real frozen pretrained vision tower
for images, a real MLP for kinematic/force state) feeding a frozen
linear projection into a caller-chosen ``output_dim`` — the SAME
dimensionality convention :class:`~neuroslm.cognition.runtime.
CognitiveRuntime`'s text ``embed_fn`` already uses, so a percept's
``content_vec`` competes in the SAME cosine-similarity recall space as
the mind's own thoughts (:mod:`neuroslm.memory.episodic`) without any
caption ever being generated.

Where a modality has no pretrained general-purpose encoder to lean on
(acoustic — no HF audio model is bundled; §15 design note), the
front-end is exact, well-established DSP (log-mel spectrogram) and
only the final feature-mixing head is a FROZEN, untrained random
projection — a real, citable technique in its own right (random
convolutional/random-feature encoders: Saxe et al. 2011, "On Random
Weights and Unsupervised Feature Learning"; Rahimi & Recht 2007,
"Random Features for Large-Scale Kernel Machines") rather than a
learned one. This is a documented, deliberate placeholder for a
pretrained audio encoder — not a stub: every call performs real,
correct, checkable math end to end.
"""
import math
from typing import Any, Dict, List, Optional, Sequence, Union

import torch
import torch.nn as nn
import torch.nn.functional as F


# ── Mechanism: log-mel spectrogram (exact DSP, no learned parameters) ─

def _hz_to_mel(freq: float) -> float:
    return 2595.0 * math.log10(1.0 + freq / 700.0)


def _mel_to_hz(mel: float) -> float:
    return 700.0 * (10.0 ** (mel / 2595.0) - 1.0)


def _mel_filterbank(sample_rate: int, n_fft: int, n_mels: int,
                    f_min: float = 0.0,
                    f_max: Optional[float] = None) -> torch.Tensor:
    """Triangular mel filterbank, HTK mel scale (``2595·log10(1+f/700)``)
    — the same formula behind MFCCs since Davis & Mermelstein (1980).
    Returns ``(n_mels, n_fft // 2 + 1)``."""
    if f_max is None:
        f_max = sample_rate / 2.0
    n_freqs = n_fft // 2 + 1
    mel_pts = torch.linspace(_hz_to_mel(f_min), _hz_to_mel(f_max), n_mels + 2)
    hz_pts = torch.tensor([_mel_to_hz(float(m)) for m in mel_pts])
    bin_pts = torch.floor((n_fft + 1) * hz_pts / sample_rate).long()

    fb = torch.zeros(n_mels, n_freqs)
    for m in range(1, n_mels + 1):
        left, center, right = (int(bin_pts[m - 1]), int(bin_pts[m]),
                               int(bin_pts[m + 1]))
        if center > left:
            for k in range(max(left, 0), min(center, n_freqs)):
                fb[m - 1, k] = (k - left) / (center - left)
        if right > center:
            for k in range(max(center, 0), min(right, n_freqs)):
                fb[m - 1, k] = (right - k) / (right - center)
    return fb


def log_mel_spectrogram(waveform: Union[Sequence[float], torch.Tensor],
                        sample_rate: int = 16000, n_fft: int = 400,
                        hop_length: int = 160, n_mels: int = 40,
                        eps: float = 1e-10) -> torch.Tensor:
    """Real, checkable acoustic front-end: STFT power spectrogram ->
    mel filterbank -> log compression. Returns ``(frames, n_mels)``.

    No learned parameters — this IS the algorithm, not a stand-in for
    one; :class:`AcousticCortex` is the only caller in production but
    the function is exercised directly by
    ``tests/sensory/test_cortices.py::TestLogMelSpectrogram`` to pin
    the DSP math independent of the frozen head that consumes it.
    """
    wav = torch.as_tensor(waveform, dtype=torch.float32).reshape(-1)
    if wav.numel() < n_fft:
        wav = F.pad(wav, (0, n_fft - wav.numel()))
    window = torch.hann_window(n_fft, dtype=wav.dtype)
    spec = torch.stft(wav, n_fft=n_fft, hop_length=hop_length,
                      window=window, center=True, return_complex=True)
    power = spec.abs() ** 2  # (freq, frames)
    fb = _mel_filterbank(sample_rate, n_fft, n_mels).to(power.dtype)
    mel = fb @ power  # (n_mels, frames)
    return torch.log(mel + eps).transpose(0, 1).contiguous()


class AcousticCortex:
    """Waveform -> latent embedding. Exact mel-spectrogram DSP front
    end feeding a frozen 1-D conv head (see module docstring re: the
    random-feature-head design decision — swap in a pretrained audio
    encoder later via the same call signature)."""

    def __init__(self, output_dim: int = 512, n_mels: int = 40,
                n_fft: int = 400, hop_length: int = 160,
                head_channels: int = 64, seed: int = 0):
        self.output_dim = output_dim
        self.n_mels = n_mels
        self.n_fft = n_fft
        self.hop_length = hop_length
        with torch.random.fork_rng():
            torch.manual_seed(seed)
            self.head = nn.Sequential(
                nn.Conv1d(n_mels, head_channels, kernel_size=5, padding=2),
                nn.ReLU(),
                nn.AdaptiveAvgPool1d(1),
            )
            self.proj = nn.Linear(head_channels, output_dim, bias=False)
        for p in list(self.head.parameters()) + list(self.proj.parameters()):
            p.requires_grad_(False)
        self.head.eval()
        self.proj.eval()

    @torch.no_grad()
    def __call__(self, waveform: Union[Sequence[float], torch.Tensor],
                sample_rate: int = 16000) -> List[float]:
        mel = log_mel_spectrogram(waveform, sample_rate, self.n_fft,
                                  self.hop_length, self.n_mels)  # (T, mel)
        x = mel.transpose(0, 1).unsqueeze(0)  # (1, mel, T)
        feat = self.head(x).squeeze(-1)  # (1, head_channels)
        return self.proj(feat).squeeze(0).tolist()


class ProprioceptiveCortex:
    """Joint kinematic/force state -> latent embedding via a frozen
    MLP. Accepts either a ``{"positions": [...], "velocities": [...],
    "forces": [...]}`` dict (Isaac Sim's ``Articulation`` API shape) or
    a flat sequence (treated as positions). Each channel is
    fixed-padded/truncated to ``max_dof`` so a fixed-size articulation
    (a 7-DOF arm, a 12-DOF quadruped, ...) always produces a
    fixed-length input to the encoder."""

    def __init__(self, max_dof: int = 32, output_dim: int = 512,
                seed: int = 0):
        self.max_dof = max_dof
        self.output_dim = output_dim
        in_dim = max_dof * 3  # positions | velocities | forces
        with torch.random.fork_rng():
            torch.manual_seed(seed)
            self.net = nn.Sequential(
                nn.Linear(in_dim, 128),
                nn.Tanh(),
                nn.Linear(128, output_dim),
            )
        for p in self.net.parameters():
            p.requires_grad_(False)
        self.net.eval()

    def _fit(self, seq: Sequence[float]) -> List[float]:
        seq = list(seq)[: self.max_dof]
        return seq + [0.0] * (self.max_dof - len(seq))

    def _to_vector(self, state: Union[Dict[str, Sequence[float]],
                                     Sequence[float]]) -> List[float]:
        if isinstance(state, dict):
            positions = state.get("positions") or []
            velocities = state.get("velocities") or []
            forces = state.get("forces") or []
        else:
            positions, velocities, forces = list(state), [], []
        return self._fit(positions) + self._fit(velocities) + self._fit(forces)

    @torch.no_grad()
    def __call__(self, state: Union[Dict[str, Sequence[float]],
                                    Sequence[float]]) -> List[float]:
        x = torch.tensor(self._to_vector(state), dtype=torch.float32)
        return self.net(x.unsqueeze(0)).squeeze(0).tolist()


class VisualCortex:
    """Image -> latent embedding via a frozen pretrained CLIP vision
    tower (``transformers``, already a project dependency — no new
    library) plus a frozen linear projection into ``output_dim``. The
    vision tower's OWN embedding space is real, pretrained, semantic —
    unlike the acoustic head, no random-feature caveat applies here.

    ``model_factory``/``processor_factory`` are the same injection
    points ``build_runtime_from_hf_lm`` uses for its LM: production
    leaves them unset (real network load from HF Hub); tests inject
    deterministic doubles so the suite never touches the network.
    """

    def __init__(self, model_id: str = "openai/clip-vit-base-patch32",
                output_dim: int = 512, seed: int = 0, device: str = "cpu",
                model_factory: Optional[Any] = None,
                processor_factory: Optional[Any] = None):
        self.output_dim = output_dim
        self.device = device

        if model_factory is not None:
            self.model = model_factory()
        else:
            from transformers import CLIPVisionModelWithProjection
            self.model = CLIPVisionModelWithProjection.from_pretrained(model_id)
        self.model.eval()
        if hasattr(self.model, "to"):
            self.model = self.model.to(device)

        if processor_factory is not None:
            self.processor = processor_factory()
        else:
            from transformers import CLIPImageProcessor
            self.processor = CLIPImageProcessor.from_pretrained(model_id)

        hidden = self._forward_embeds(self._probe_image()).shape[-1]
        with torch.random.fork_rng():
            torch.manual_seed(seed)
            self.proj = nn.Linear(int(hidden), output_dim, bias=False)
        for p in self.proj.parameters():
            p.requires_grad_(False)
        self.proj.eval()

    def _probe_image(self):
        import numpy as np
        return np.zeros((8, 8, 3), dtype="uint8")

    @torch.no_grad()
    def _forward_embeds(self, image) -> torch.Tensor:
        inputs = self.processor(images=image, return_tensors="pt")
        pixel_values = inputs["pixel_values"].to(self.device)
        out = self.model(pixel_values=pixel_values)
        embeds = getattr(out, "image_embeds", None)
        if embeds is None:
            embeds = getattr(out, "pooler_output", None)
        return embeds.squeeze(0).float()

    @torch.no_grad()
    def __call__(self, image) -> List[float]:
        embeds = self._forward_embeds(image)
        return self.proj(embeds).tolist()
