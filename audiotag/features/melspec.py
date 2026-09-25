"""Log-mel spectrogram feature extraction with torchaudio."""

import torchaudio
from torch import Tensor


def log_mel(
    waveform: Tensor,
    sr: int,
    *,
    target_sr: int = 16000,
    n_fft: int = 1024,
    win_length: int = 1024,
    hop_length: int = 512,
    n_mels: int = 128,
    eps: float = 1e-5,
) -> Tensor:
    """Compute a per-clip mean/std-normalized log-mel spectrogram.

    Args:
        waveform: Float tensor of shape ``[channels, samples]`` (or ``[samples]``).
        sr: Sample rate of ``waveform``.

    Returns:
        Tensor of shape ``[1, n_mels, T]`` with zero mean and unit variance
        per clip.
    """
    if waveform.dim() > 1:
        waveform = waveform.mean(dim=0, keepdim=True)  # mono-mix
    if sr != target_sr:
        waveform = torchaudio.functional.resample(waveform, sr, target_sr)

    mel = torchaudio.transforms.MelSpectrogram(
        sample_rate=target_sr,
        n_fft=n_fft,
        win_length=win_length,
        hop_length=hop_length,
        n_mels=n_mels,
    )(waveform)
    db = torchaudio.transforms.AmplitudeToDB()(mel)

    mean = db.mean()
    std = db.std()
    return (db - mean) / (std + eps)
