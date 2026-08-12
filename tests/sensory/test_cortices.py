# -*- coding: utf-8 -*-
"""Sensory cortices: visual / acoustic / proprioceptive raw-signal ->
latent-embedding encoders (§15). Contracts pin the actual math (DSP
correctness, determinism, discriminability) — not shape-only checks.

Dependency-injected exactly like the rest of the cognition layer
(``build_runtime_from_hf_lm``'s ``model_factory``/``tokenizer_factory``
pattern): production wires a real frozen CLIP vision tower via
``transformers``; tests inject a deterministic fake vision backend so
the suite never touches the network. The mel-spectrogram DSP and the
proprioceptive encoder have no network dependency at all — real math,
tested directly.
"""
import math
import random

import pytest
import torch

from neuroslm.sensory.cortices import (
    AcousticCortex,
    ProprioceptiveCortex,
    VisualCortex,
    log_mel_spectrogram,
)


# ── Mechanism: log-mel spectrogram DSP (no dependency injection — this
#    is real, self-contained, checkable math) ─────────────────────────

class TestLogMelSpectrogram:
    def _sine(self, freq, sr=16000, dur=0.5, amp=1.0):
        n = int(sr * dur)
        t = torch.arange(n, dtype=torch.float32) / sr
        return amp * torch.sin(2 * math.pi * freq * t)

    def test_output_shape_is_frames_by_n_mels(self):
        wav = self._sine(440.0)
        mel = log_mel_spectrogram(wav, sample_rate=16000, n_fft=400,
                                  hop_length=160, n_mels=40)
        assert mel.shape[1] == 40
        assert mel.shape[0] > 1

    def test_silence_has_lower_energy_than_tone(self):
        silence = torch.zeros(8000)
        tone = self._sine(440.0)
        mel_silence = log_mel_spectrogram(silence, 16000, 400, 160, 40)
        mel_tone = log_mel_spectrogram(tone, 16000, 400, 160, 40)
        assert float(mel_tone.mean()) > float(mel_silence.mean())

    def test_low_freq_tone_activates_lower_mel_bins_than_high_freq(self):
        # A real, checkable DSP property: the mel scale is monotonic in
        # frequency, so a 200 Hz tone's peak-energy bin index must be
        # LOWER than a 6000 Hz tone's, for the SAME filterbank.
        low = self._sine(200.0)
        high = self._sine(6000.0)
        mel_low = log_mel_spectrogram(low, 16000, 400, 160, 40)
        mel_high = log_mel_spectrogram(high, 16000, 400, 160, 40)
        peak_low = int(mel_low.mean(dim=0).argmax())
        peak_high = int(mel_high.mean(dim=0).argmax())
        assert peak_low < peak_high

    def test_louder_tone_has_higher_energy_same_frequency(self):
        quiet = self._sine(440.0, amp=0.1)
        loud = self._sine(440.0, amp=1.0)
        mel_quiet = log_mel_spectrogram(quiet, 16000, 400, 160, 40)
        mel_loud = log_mel_spectrogram(loud, 16000, 400, 160, 40)
        assert float(mel_loud.mean()) > float(mel_quiet.mean())

    def test_finite_on_silence_no_log_zero_nan(self):
        mel = log_mel_spectrogram(torch.zeros(4000), 16000, 400, 160, 40)
        assert torch.isfinite(mel).all()


# ── AcousticCortex: DSP front-end + frozen conv head -> latent ───────

class TestAcousticCortex:
    def _sine(self, freq, sr=16000, dur=0.5):
        n = int(sr * dur)
        t = torch.arange(n, dtype=torch.float32) / sr
        return torch.sin(2 * math.pi * freq * t)

    def test_output_dim_matches_config(self):
        cortex = AcousticCortex(output_dim=64, seed=0)
        vec = cortex(self._sine(440.0).tolist(), sample_rate=16000)
        assert len(vec) == 64

    def test_deterministic_same_input_same_output(self):
        cortex = AcousticCortex(output_dim=32, seed=1)
        wav = self._sine(440.0).tolist()
        a = cortex(wav, sample_rate=16000)
        b = cortex(wav, sample_rate=16000)
        assert a == b

    def test_distinct_tones_produce_distinct_embeddings(self):
        cortex = AcousticCortex(output_dim=32, seed=2)
        a = cortex(self._sine(220.0).tolist(), sample_rate=16000)
        b = cortex(self._sine(4000.0).tolist(), sample_rate=16000)
        assert a != b

    def test_finite_output(self):
        cortex = AcousticCortex(output_dim=16, seed=3)
        vec = cortex(self._sine(1000.0).tolist(), sample_rate=16000)
        assert all(math.isfinite(x) for x in vec)

    def test_silence_does_not_crash_and_is_finite(self):
        cortex = AcousticCortex(output_dim=16, seed=4)
        vec = cortex([0.0] * 4000, sample_rate=16000)
        assert len(vec) == 16
        assert all(math.isfinite(x) for x in vec)


# ── ProprioceptiveCortex: joint kinematic/force state -> latent ──────

class TestProprioceptiveCortex:
    def test_output_dim_matches_config(self):
        cortex = ProprioceptiveCortex(max_dof=12, output_dim=48, seed=0)
        vec = cortex({"positions": [0.1] * 6, "velocities": [0.0] * 6})
        assert len(vec) == 48

    def test_deterministic_same_state_same_output(self):
        cortex = ProprioceptiveCortex(max_dof=8, output_dim=16, seed=1)
        state = {"positions": [0.2, -0.1, 0.5], "velocities": [0.0, 0.1, 0.0]}
        a = cortex(state)
        b = cortex(state)
        assert a == b

    def test_distinct_states_produce_distinct_embeddings(self):
        cortex = ProprioceptiveCortex(max_dof=8, output_dim=16, seed=2)
        a = cortex({"positions": [0.0, 0.0], "velocities": [0.0, 0.0]})
        b = cortex({"positions": [1.5, -2.0], "velocities": [0.3, 0.1]})
        assert a != b

    def test_handles_fewer_joints_than_max_dof(self):
        cortex = ProprioceptiveCortex(max_dof=32, output_dim=16, seed=3)
        vec = cortex({"positions": [0.1, 0.2], "velocities": [0.0, 0.0]})
        assert len(vec) == 16
        assert all(math.isfinite(x) for x in vec)

    def test_handles_more_joints_than_max_dof_by_truncating(self):
        cortex = ProprioceptiveCortex(max_dof=4, output_dim=16, seed=4)
        vec = cortex({"positions": [0.1] * 10, "velocities": [0.0] * 10})
        assert len(vec) == 16
        assert all(math.isfinite(x) for x in vec)

    def test_accepts_flat_sequence_as_well_as_dict(self):
        cortex = ProprioceptiveCortex(max_dof=8, output_dim=16, seed=5)
        vec = cortex([0.1, 0.2, 0.3, 0.0, 0.0, 0.0])
        assert len(vec) == 16

    def test_forces_channel_shifts_embedding(self):
        cortex = ProprioceptiveCortex(max_dof=8, output_dim=16, seed=6)
        base = {"positions": [0.0, 0.0], "velocities": [0.0, 0.0]}
        with_force = {"positions": [0.0, 0.0], "velocities": [0.0, 0.0],
                      "forces": [5.0, -5.0]}
        assert cortex(base) != cortex(with_force)


# ── VisualCortex: frozen pretrained vision tower -> projected latent
#    (dependency-injected — no network in tests) ─────────────────────

class _FakeCLIPOutput:
    def __init__(self, embeds):
        self.image_embeds = embeds


class _FakeCLIPVisionModel:
    """Deterministic stand-in for a real CLIPVisionModelWithProjection:
    output is a fixed function of the input pixel content (mean per
    channel), so distinct images produce distinct embeddings and
    identical images produce identical ones — exactly the properties a
    real frozen vision tower has, without any network or GPU."""

    def __init__(self, hidden=8):
        self.hidden = hidden

    def eval(self):
        return self

    def to(self, device):
        return self

    def __call__(self, pixel_values):
        # pixel_values: (B, C, H, W)
        b = pixel_values.shape[0]
        means = pixel_values.mean(dim=(2, 3))  # (B, C)
        reps = means.repeat(1, (self.hidden // means.shape[1]) + 1)
        return _FakeCLIPOutput(reps[:, : self.hidden])


class _FakeCLIPProcessor:
    def __call__(self, images, return_tensors="pt"):
        import numpy as np
        from PIL import Image
        if isinstance(images, Image.Image):
            arr = np.asarray(images.convert("RGB"), dtype="float32") / 255.0
        else:
            arr = np.asarray(images, dtype="float32")
            if arr.max() > 1.5:
                arr = arr / 255.0
        t = torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0)
        return {"pixel_values": t}


class TestVisualCortex:
    def _rt(self, output_dim=32, seed=0):
        return VisualCortex(
            output_dim=output_dim, seed=seed,
            model_factory=lambda: _FakeCLIPVisionModel(hidden=8),
            processor_factory=lambda: _FakeCLIPProcessor(),
        )

    def _image(self, fill):
        import numpy as np
        return (np.ones((32, 32, 3), dtype="uint8") * fill)

    def test_output_dim_matches_config(self):
        cortex = self._rt(output_dim=32)
        vec = cortex(self._image(128))
        assert len(vec) == 32

    def test_deterministic_same_image_same_output(self):
        cortex = self._rt()
        img = self._image(100)
        assert cortex(img) == cortex(img)

    def test_distinct_images_produce_distinct_embeddings(self):
        cortex = self._rt()
        dark = self._image(10)
        bright = self._image(240)
        assert cortex(dark) != cortex(bright)

    def test_accepts_pil_image_input(self):
        from PIL import Image
        cortex = self._rt()
        img = Image.fromarray(self._image(200))
        vec = cortex(img)
        assert len(vec) == 32
        assert all(math.isfinite(x) for x in vec)

    def test_injection_points_present(self):
        import inspect
        sig = inspect.signature(VisualCortex.__init__)
        assert "model_factory" in sig.parameters
        assert "processor_factory" in sig.parameters
