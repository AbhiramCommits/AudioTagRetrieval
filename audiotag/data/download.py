"""Download and extract the MagnaTagATune dataset into ``data/raw/``.

Idempotent: files already present on disk with the expected remote size are
skipped, as is extraction once the marker file matches. On fetch failures a
manual-download fallback URL is printed.

Usage:
    python -m audiotag.data.download            # full dataset
    python -m audiotag.data.download --subset 500
"""

from __future__ import annotations

import argparse
import shutil
import sys
import zipfile
from pathlib import Path

import requests
from tqdm import tqdm

BASE_URLS = [
    "https://mirg.city.ac.uk/codeapps/the-magnatagatune-dataset",
    "http://mi.soi.city.ac.uk/datasets/magnatagatune",
]

PART_NAMES = ["mp3.zip.001", "mp3.zip.002", "mp3.zip.003"]
ANNOTATIONS_NAME = "annotations_final.csv"
ZIP_NAME = "mp3.zip"
EXTRACTED_MARKER = ".extracted"

CHUNK_SIZE = 1 << 20  # 1 MiB


class DownloadError(RuntimeError):
    """Raised when a remote file could not be fetched."""


def _remote_size(session: requests.Session, url: str) -> int | None:
    try:
        resp = session.head(url, timeout=30)
        if resp.status_code == 200 and "content-length" in resp.headers:
            return int(resp.headers["content-length"])
    except requests.RequestException:
        pass
    return None


def fetch_file(session: requests.Session, url: str, dst: Path, *, force: bool = False) -> None:
    """Download ``url`` to ``dst`` unless it already exists with the right size."""
    expected = _remote_size(session, url)
    if dst.exists() and not force:
        actual = dst.stat().st_size
        if expected is None or actual == expected:
            print(f"skip {dst.name} (already present, {actual / 1e6:.1f} MB)")
            return
        print(f"warn: {dst.name} size mismatch ({actual} != {expected}), re-downloading")

    dst.parent.mkdir(parents=True, exist_ok=True)
    tmp = dst.with_name(dst.name + ".part")
    try:
        with session.get(url, stream=True, timeout=120) as resp:
            resp.raise_for_status()
            total = int(resp.headers.get("content-length", 0))
            with open(tmp, "wb") as fh, tqdm(
                total=total, unit="B", unit_scale=True, desc=dst.name
            ) as bar:
                for chunk in resp.iter_content(chunk_size=CHUNK_SIZE):
                    fh.write(chunk)
                    bar.update(len(chunk))
        tmp.replace(dst)
    except (requests.RequestException, OSError) as exc:
        tmp.unlink(missing_ok=True)
        raise DownloadError(f"failed to fetch {url}: {exc}") from exc


def _manual_download_hint(name: str, dst: Path) -> None:
    print(
        f"\nMANUAL DOWNLOAD NEEDED: could not fetch '{name}' automatically.\n"
        f"  Fetch it yourself, e.g. from {BASE_URLS[0]}/{name}\n"
        f"  (or one of: {' '.join(f'{b}/{name}' for b in BASE_URLS)})\n"
        f"  and place it at {dst}, then re-run this command.\n",
        file=sys.stderr,
    )


def _concat_parts(raw: Path, *, force: bool) -> None:
    combined = raw / ZIP_NAME
    if combined.exists() and not force:
        return
    parts = [raw / name for name in PART_NAMES]
    if not all(part.exists() for part in parts):
        print("error: missing mp3.zip parts, cannot concatenate", file=sys.stderr)
        raise SystemExit(1)
    tmp = combined.with_name(combined.name + ".part")
    print("concatenating parts into mp3.zip ...")
    with open(tmp, "wb") as out:
        for part in parts:
            with open(part, "rb") as fh:
                shutil.copyfileobj(fh, out)
    tmp.replace(combined)


def _extract(raw: Path, subset: int | None, *, force: bool) -> None:
    marker = raw / EXTRACTED_MARKER
    desired = str(subset) if subset else "all"
    if marker.exists() and marker.read_text().strip() == desired and not force:
        print("skip extraction (already extracted)")
        return

    print(f"extracting mp3s to {raw} ...")
    with zipfile.ZipFile(raw / ZIP_NAME) as zf:
        entries = sorted(name for name in zf.namelist() if name.lower().endswith(".mp3"))
        if subset:
            entries = entries[:subset]
        for name in tqdm(entries, desc="extract"):
            zf.extract(name, raw)
    marker.write_text(desired)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", default="data", help="base data directory (default: data)")
    parser.add_argument(
        "--subset", type=int, default=None, help="keep only the first N clips (fast local runs)"
    )
    parser.add_argument("--force", action="store_true", help="re-download / re-extract everything")
    args = parser.parse_args(argv)

    raw = Path(args.data_dir) / "raw"
    raw.mkdir(parents=True, exist_ok=True)

    with requests.Session() as session:
        for name in PART_NAMES + [ANNOTATIONS_NAME]:
            for base in BASE_URLS:
                url = f"{base.rstrip('/')}/{name}"
                try:
                    fetch_file(session, url, raw / name, force=args.force)
                    break
                except DownloadError as exc:
                    print(f"warn: {exc}", file=sys.stderr)
            else:
                _manual_download_hint(name, raw / name)
                return 1

    _concat_parts(raw, force=args.force)
    _extract(raw, args.subset, force=args.force)

    n_clips = sum(1 for _ in raw.rglob("*.mp3"))
    print(f"ready: {n_clips} clips in {raw}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
