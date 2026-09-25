"""Central configuration for the audio-tag-retrieval pipeline.

All values can be overridden with environment variables prefixed by ``AUDIOTAG_``
(e.g. ``AUDIOTAG_SAMPLE_RATE=22050``) or via a ``.env`` file.
"""

from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


class Config(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="AUDIOTAG_", env_file=".env", extra="ignore")

    data_dir: Path = Path("data")

    # Audio / spectrogram parameters.
    sample_rate: int = 16000
    n_mels: int = 128
    n_fft: int = 1024
    win_length: int = 1024
    hop_length: int = 512
    clip_seconds: int = 10

    # Training / retrieval.
    batch_size: int = 32
    lr: float = 1e-3
    epochs: int = 50
    top_k: int = 20
    num_workers: int = 4
    n_tags: int = 50
    seed: int = 0

    @property
    def raw_dir(self) -> Path:
        return self.data_dir / "raw"

    @property
    def processed_dir(self) -> Path:
        return self.data_dir / "processed"

    @property
    def mels_dir(self) -> Path:
        return self.processed_dir / "mels"

    @property
    def clip_samples(self) -> int:
        return self.sample_rate * self.clip_seconds

    @property
    def n_frames(self) -> int:
        return self.clip_samples // self.hop_length + 1

    def ensure_dirs(self) -> None:
        for directory in (self.raw_dir, self.processed_dir, self.mels_dir):
            directory.mkdir(parents=True, exist_ok=True)
