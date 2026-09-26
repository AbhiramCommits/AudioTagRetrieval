"""Tests for the labels pipeline (vocabulary + manifests) on synthetic CSVs."""

import csv
import json
from pathlib import Path

import numpy as np
import pandas as pd

from audiotag.data import labels as labels_mod


def _write_annotations(tmp_path: Path, dirs=("0", "c", "f"), n_per=3):
    raw = tmp_path / "data" / "raw"
    tag_cols = ["vocals", "vocal", "beat", "beats", "piano"]
    clip_ids = []
    for d in dirs:
        (raw / d).mkdir(parents=True, exist_ok=True)
        for i in range(n_per):
            cid = f"{d}/artist{i}-00-01-0{i}"
            clip_ids.append(cid)
            (raw / d / f"artist{i}-00-01-0{i}.mp3").touch()
    with open(raw / "annotations_final.csv", "w", newline="") as fh:
        writer = csv.writer(fh, delimiter="\t")
        writer.writerow(["clip_id"] + tag_cols + ["mp3_path"])
        for j, cid in enumerate(clip_ids):
            labels = [1 if (j + k) % 3 == 0 else 0 for k in range(len(tag_cols))]
            writer.writerow([j] + labels + [cid + ".mp3"])
    return raw, tag_cols, clip_ids


def test_sniff_sep(tmp_path):
    raw, _, _ = _write_annotations(tmp_path)
    assert labels_mod._sniff_sep(raw / "annotations_final.csv") == "\t"


def test_tag_columns_detection(tmp_path):
    raw, tag_cols, _ = _write_annotations(tmp_path)
    df = pd.read_csv(raw / "annotations_final.csv", sep="\t")
    detected = labels_mod.tag_columns(df)
    assert detected == tag_cols


def test_merge_synonyms(tmp_path):
    raw, _, _ = _write_annotations(tmp_path)
    df = pd.read_csv(raw / "annotations_final.csv", sep="\t")
    merged = labels_mod.merge_synonyms(df, labels_mod.tag_columns(df))
    assert "vocal" in merged and "vocals" not in merged
    assert "beat" in merged and "beats" not in merged
    assert "piano" in merged


def test_labels_pipeline_end_to_end(tmp_path):
    raw, _, _ = _write_annotations(tmp_path)
    data_dir = tmp_path / "data"

    assert labels_mod.main(["--data-dir", str(data_dir)]) == 0

    tags_payload = json.loads((data_dir / "processed" / "tags.json").read_text())
    assert set(tags_payload["tags"]) == {"vocal", "beat", "piano"}

    splits = {"train": ["0"], "val": ["c"], "test": ["f"]}
    for split, dirs in splits.items():
        df = pd.read_parquet(data_dir / "processed" / f"{split}.parquet")
        assert len(df) == 3
        assert set(df["clip_id"].str.split("/", n=1).str[0]) == set(dirs)
        label_vec = df["labels"].iloc[0]
        assert len(label_vec) == 3
        assert label_vec.dtype == np.uint8
        assert all(df["mp3_path"].str.endswith(".mp3"))


def test_labels_pipeline_drops_missing_files(tmp_path):
    raw, _, _ = _write_annotations(tmp_path, n_per=1)
    # delete one clip so the manifest must drop it
    (raw / "c" / "artist0-00-01-00.mp3").unlink()
    data_dir = tmp_path / "data"
    labels_mod.main(["--data-dir", str(data_dir)])
    df = pd.read_parquet(data_dir / "processed" / "val.parquet")
    assert len(df) == 0
    train = pd.read_parquet(data_dir / "processed" / "train.parquet")
    assert len(train) == 1


def test_labels_missing_annotations_errors(tmp_path):
    assert labels_mod.main(["--data-dir", str(tmp_path / "data")]) == 1
