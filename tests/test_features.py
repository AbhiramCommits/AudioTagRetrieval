"""Tests for log-mel feature extraction."""

import pytest
import torch

from audiotag.features.melspec import crop_or_pad, log_mel


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


@pytest.mark.parametrize("sr", [44100, 22050])
def test_log_mel_resamples(sr):
    seconds = 1.0
    waveform = _sine(seconds=seconds, sr=sr)
    mel = log_mel(waveform, sr)
    assert mel.shape[0] == 1
    assert mel.shape[1] == 128
    expected_frames = 1 + int(16000 * seconds) // 512
    assert mel.shape[2] == expected_frames
    assert torch.isfinite(mel).all()


def test_log_mel_mono_mixes_stereo():
    waveform = _sine(seconds=1.0, sr=16000, channels=2)
    mel = log_mel(waveform, 16000)
    mono_mel = log_mel(waveform.mean(dim=0, keepdim=True), 16000)
    assert mel.shape == mono_mel.shape
    assert torch.allclose(mel, mono_mel, atol=1e-6)


def test_crop_or_pad_pads_short_clip():
    mel = torch.randn(1, 128, 100)
    out = crop_or_pad(mel, 313)
    assert out.shape == (1, 128, 313)
    assert torch.equal(out[..., :100], mel)
    assert (out[..., 100:] == 0).all()


def test_crop_or_pad_center_crops_long_clip():
    mel = torch.randn(1, 128, 400)
    out = crop_or_pad(mel, 313)
    assert out.shape == (1, 128, 313)
    start = (400 - 313) // 2
    assert torch.equal(out, mel[..., start : start + 313])


def test_crop_or_pad_keeps_exact_length():
    mel = torch.randn(1, 128, 313)
    out = crop_or_pad(mel, 313)
    assert torch.equal(out, mel)
