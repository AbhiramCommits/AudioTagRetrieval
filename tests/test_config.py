"""Tests for pydantic-settings configuration."""

from pathlib import Path

from audiotag.config import Config


def test_defaults():
    cfg = Config()
    assert cfg.sample_rate == 16000
    assert cfg.n_mels == 128
    assert cfg.n_fft == 1024
    assert cfg.win_length == 1024
    assert cfg.hop_length == 512
    assert cfg.clip_seconds == 10
    assert cfg.batch_size == 32
    assert cfg.lr == 1e-3
    assert cfg.top_k == 20
    assert cfg.n_tags == 50


def test_derived_paths_and_frames(tmp_path):
    cfg = Config(data_dir=tmp_path / "data")
    assert cfg.raw_dir == tmp_path / "data" / "raw"
    assert cfg.processed_dir == tmp_path / "data" / "processed"
    assert cfg.mels_dir == tmp_path / "data" / "processed" / "mels"
    assert cfg.clip_samples == 160000
    assert cfg.n_frames == 313


def test_env_override(monkeypatch):
    monkeypatch.setenv("AUDIOTAG_SAMPLE_RATE", "22050")
    monkeypatch.setenv("AUDIOTAG_TOP_K", "42")
    cfg = Config()
    assert cfg.sample_rate == 22050
    assert cfg.top_k == 42


def test_ensure_dirs(tmp_path):
    cfg = Config(data_dir=tmp_path / "data")
    cfg.ensure_dirs()
    assert Path(cfg.raw_dir).is_dir()
    assert Path(cfg.processed_dir).is_dir()
    assert Path(cfg.mels_dir).is_dir()
