"""Build the top-50 tag vocabulary and train/val/test manifests for MagnaTagATune.

Reads ``annotations_final.csv``, ranks tags by positive count, merges known
synonym pairs, keeps the top ``n_tags``, and writes:

- ``data/processed/tags.json``: tag list + per-tag positive counts
- ``data/processed/{train,val,test}.parquet``: clip_id, mp3_path, labels
  (50-dim uint8 vector)

Uses the standard MagnaTagATune folder split: directories 0-b train, c
validate, d-f test.

Usage:
    python -m audiotag.data.labels
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from tqdm import tqdm

TRAIN_DIRS = set("0123456789ab")
VAL_DIRS = {"c"}
TEST_DIRS = set("def")

# canonical tag -> alias tags merged into it (counts summed, labels OR-ed)
SYNONYMS: dict[str, tuple[str, ...]] = {
    "beat": ("beats",),
    "vocal": ("vocals",),
    "female vocal": ("female vocals",),
    "male vocal": ("male vocals",),
    "no vocal": ("no vocals",),
    "no beat": ("no beats",),
    "string": ("strings",),
    "guitar": ("guitars",),
    "drum": ("drums",),
    "synth": ("synthesizer",),
}


def _sniff_sep(path: Path) -> str:
    with open(path) as fh:
        header = fh.readline()
    return "\t" if header.count("\t") > header.count(",") else ","


def tag_columns(df: pd.DataFrame) -> list[str]:
    """Return columns that look like binary 0/1 tag annotations."""
    cols = []
    for col in df.columns:
        series = df[col]
        if series.dtype.kind in "iufb":
            values = set(series.dropna().unique())
            if values <= {0, 1}:
                cols.append(col)
    return cols


def merge_synonyms(df: pd.DataFrame, tag_cols: list[str]) -> list[str]:
    """Merge synonym columns into their canonical tag; return final column order."""
    present = list(tag_cols)
    for canonical, aliases in SYNONYMS.items():
        for alias in aliases:
            if alias not in df.columns:
                continue
            if canonical in df.columns:
                df[canonical] = df[canonical] | df[alias]
            else:
                df[canonical] = df[alias]
                present.append(canonical)
            if alias in present:
                present.remove(alias)
            df.drop(columns=[alias], inplace=True)
    return [col for col in present if col in df.columns]


def split_dir(clip_id: str) -> str:
    return clip_id.split("/", 1)[0]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", default="data", help="base data directory (default: data)")
    parser.add_argument("--n-tags", type=int, default=50, help="tag vocabulary size (default: 50)")
    parser.add_argument(
        "--include-missing",
        action="store_true",
        help="keep clips whose mp3 file is not on disk (e.g. subset runs)",
    )
    args = parser.parse_args(argv)

    data_dir = Path(args.data_dir)
    raw_dir = data_dir / "raw"
    processed_dir = data_dir / "processed"
    processed_dir.mkdir(parents=True, exist_ok=True)

    annotations = raw_dir / "annotations_final.csv"
    if not annotations.exists():
        print(f"error: {annotations} not found; run 'python -m audiotag.data.download' first")
        return 1

    df = pd.read_csv(annotations, sep=_sniff_sep(annotations))
    if "mp3_path" in df.columns:
        df["mp3_path"] = df["mp3_path"].astype(str)
        df["clip_id"] = df["mp3_path"].str.replace(r"\.mp3$", "", regex=True)
    else:
        id_col = "clip_id" if "clip_id" in df.columns else df.columns[0]
        df["clip_id"] = df[id_col].astype(str)
        df["mp3_path"] = df["clip_id"].map(
            lambda cid: cid if cid.endswith(".mp3") else cid + ".mp3"
        )

    tag_cols = merge_synonyms(df, tag_columns(df))
    print(f"found {len(tag_cols)} tags after synonym merging")

    counts = df[tag_cols].sum().sort_values(ascending=False)
    top_tags = counts.head(args.n_tags).index.tolist()

    df["mp3_path"] = df["clip_id"].map(lambda cid: cid if cid.endswith(".mp3") else cid + ".mp3")

    missing = [p for p in tqdm(df["mp3_path"], desc="checking files") if not (raw_dir / p).exists()]
    if missing and not args.include_missing:
        print(f"warn: {len(missing)} clips missing on disk; dropping them from manifests")
        df = df[~df["mp3_path"].isin(set(missing))]

    tags_payload = {
        "tags": top_tags,
        "positive_counts": {tag: int(counts[tag]) for tag in top_tags},
    }
    with open(processed_dir / "tags.json", "w") as fh:
        json.dump(tags_payload, fh, indent=2)
    print(f"wrote {processed_dir / 'tags.json'}")

    splits = {"train": TRAIN_DIRS, "val": VAL_DIRS, "test": TEST_DIRS}
    for split, dirs in splits.items():
        sub = df[df["clip_id"].map(split_dir).isin(dirs)].sort_values("clip_id")
        labels = sub[top_tags].to_numpy(dtype=np.uint8)
        table = pa.table(
            {
                "clip_id": pa.array(sub["clip_id"].tolist()),
                "mp3_path": pa.array(sub["mp3_path"].tolist()),
                "labels": pa.array(labels.tolist(), type=pa.list_(pa.uint8())),
            }
        )
        path = processed_dir / f"{split}.parquet"
        pq.write_table(table, path)
        print(f"wrote {path} ({len(sub)} clips)")

    return 0


if __name__ == "__main__":
    sys.exit(main())
