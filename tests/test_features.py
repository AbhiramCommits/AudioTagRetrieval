"""Tests for log-mel feature extraction."""

import torch

from audiotag.features.melspec import log_mel


def _sine(seconds: float = 2.0, sr: int = 16000, freq: float = 440.0, channels: int = 1):
    t = torch.arange(int(sr * seconds), dtype=torch.float32) / sr
    wave = (0.5 * torch.sin(2 * torch.pi * freq * t)).unsqueeze(0)
    return wave.repeat(channels, 1)


def test_log_mel_shape():
    waveform = _sine()
    mel = log_mel(waveform, 16000)
    assert mel.dim() == 3
    assert mel.shape[0] == 1
    assert mel.shape[1] == 128
    assert mel.shape[2] == 1 + waveform.shape[-1] // 512
    assert torch.isfinite(mel).all()


def test_log_mel_normalization():
    mel = log_mel(_sine(seconds=3.0), 16000)
    assert abs(mel.mean().item()) < 1e-4
    assert abs(mel.std().item() - 1.0) < 1e-3


def test_log_mel_resamples_and_mono_mixes():
    waveform = _sine(seconds=1.0, sr=44100, channels=2)
    mel = log_mel(waveform, 44100)
    assert mel.shape[0] == 1
    assert mel.shape[1] == 128
    assert torch.isfinite(mel).all()
