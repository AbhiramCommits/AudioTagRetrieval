"""Download and extract the MagnaTagATune dataset into ``data/raw/``.

Sources are tried in order (HuggingFace mirror first, then the City University
mirror). Idempotent: files already present on disk with the expected remote
size are skipped, as is extraction once the marker file matches. On fetch
failures a manual-download fallback URL is printed.

Usage:
    python -m audiotag.data.download            # full dataset
    python -m audiotag.data.download --subset 2000
"""

from __future__ import annotations

import argparse
import random
import shutil
import sys
import zipfile
from collections import defaultdict
from pathlib import Path

import requests
from tqdm import tqdm

# HuggingFace hosts the concatenated mp3.zip; the City mirror hosts 3 parts.
BASE_URLS = [
    "https://huggingface.co/datasets/confit/magnatagatune/resolve/main",
    "https://mirg.city.ac.uk/datasets/magnatagatune",
]

PART_NAMES = ["mp3.zip.001", "mp3.zip.002", "mp3.zip.003"]
ANNOTATIONS_NAME = "annotations_final.csv"
ZIP_NAME = "mp3.zip"
EXTRACTED_MARKER = ".extracted"

CHUNK_SIZE = 1 << 20  # 1 MiB


class DownloadError(RuntimeError):
    """Raised when a remote file could not be fetched."""


def _remote_size(session: requests.Session, url: str) -> int | None:
    """Determine remote size via a 1-byte ranged GET (follows redirects)."""
    try:
        with session.get(
            url, headers={"Range": "bytes=0-0"}, timeout=(15, 30), stream=True
        ) as resp:
            if resp.status_code == 206 and "content-range" in resp.headers:
                return int(resp.headers["content-range"].split("/")[-1])
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
        if expected is None and actual > 0:
            print(f"skip {dst.name} (already present, {actual / 1e6:.1f} MB, size unchecked)")
            return
        if expected is not None and actual == expected:
            print(f"skip {dst.name} (already present, {actual / 1e6:.1f} MB)")
            return
        print(f"warn: {dst.name} size mismatch ({actual} != {expected}), re-downloading")

    dst.parent.mkdir(parents=True, exist_ok=True)
    tmp = dst.with_name(dst.name + ".part")
    try:
        with session.get(url, stream=True, timeout=(15, 120)) as resp:
            resp.raise_for_status()
            total = int(resp.headers.get("content-length", 0))
            with (
                open(tmp, "wb") as fh,
                tqdm(total=total, unit="B", unit_scale=True, desc=dst.name) as bar,
            ):
                for chunk in resp.iter_content(chunk_size=CHUNK_SIZE):
                    fh.write(chunk)
                    bar.update(len(chunk))
        tmp.replace(dst)
    except (requests.RequestException, OSError) as exc:
        tmp.unlink(missing_ok=True)
        raise DownloadError(f"failed to fetch {url}: {exc}") from exc


def _manual_download_hint(name: str, dst: Path) -> None:
    urls = " ".join(f"{base}/{name}" for base in BASE_URLS)
    print(
        f"\nMANUAL DOWNLOAD NEEDED: could not fetch '{name}' automatically.\n"
        f"  Fetch it yourself, e.g. from {urls}\n"
        f"  and place it at {dst}, then re-run this command.\n",
        file=sys.stderr,
    )


def _download_annotations(raw: Path, session: requests.Session, *, force: bool) -> bool:
    for base in BASE_URLS:
        url = f"{base.rstrip('/')}/{ANNOTATIONS_NAME}"
        try:
            fetch_file(session, url, raw / ANNOTATIONS_NAME, force=force)
            return True
        except DownloadError as exc:
            print(f"warn: {exc}", file=sys.stderr)
    _manual_download_hint(ANNOTATIONS_NAME, raw / ANNOTATIONS_NAME)
    return False


def _download_mp3s(raw: Path, session: requests.Session, *, force: bool) -> bool:
    if not force and (raw / ZIP_NAME).exists():
        print("skip mp3.zip (already present)")
        return True
    for base in BASE_URLS:
        url = f"{base.rstrip('/')}/{ZIP_NAME}"
        try:
            fetch_file(session, url, raw / ZIP_NAME, force=force)
            return True
        except DownloadError as exc:
            print(f"warn: {exc}", file=sys.stderr)
    for base in BASE_URLS:
        try:
            for name in PART_NAMES:
                fetch_file(session, f"{base.rstrip('/')}/{name}", raw / name, force=force)
            _concat_parts(raw, force=force)
            return True
        except DownloadError as exc:
            print(f"warn: {exc}", file=sys.stderr)
    _manual_download_hint(ZIP_NAME, raw / ZIP_NAME)
    return False


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


def _select_entries(entries: list[str], subset: int | None) -> list[str]:
    """Deterministic stratified sample across directories (keeps splits usable)."""
    if subset is None or subset >= len(entries):
        return entries
    rng = random.Random(42)
    by_dir: dict[str, list[str]] = defaultdict(list)
    for name in entries:
        by_dir[name.split("/", 1)[0]].append(name)
    dirs = sorted(by_dir)

    counts = {d: int(subset * len(by_dir[d]) / len(entries)) for d in dirs}
    remaining = subset - sum(counts.values())
    fracs = {d: subset * len(by_dir[d]) / len(entries) - counts[d] for d in dirs}
    for d in sorted(dirs, key=lambda d: -fracs[d])[:remaining]:
        counts[d] += 1

    chosen: list[str] = []
    for d in dirs:
        chosen.extend(rng.sample(by_dir[d], min(counts[d], len(by_dir[d]))))
    return sorted(chosen)


def _extract(raw: Path, subset: int | None, *, force: bool) -> None:
    marker = raw / EXTRACTED_MARKER
    desired = str(subset) if subset else "all"
    if marker.exists() and marker.read_text().strip() == desired and not force:
        print("skip extraction (already extracted)")
        return

    print(f"extracting mp3s to {raw} ...")
    with zipfile.ZipFile(raw / ZIP_NAME) as zf:
        entries = sorted(name for name in zf.namelist() if name.lower().endswith(".mp3"))
        selected = _select_entries(entries, subset)
        for name in tqdm(selected, desc="extract"):
            zf.extract(name, raw)
    marker.write_text(desired)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", default="data", help="base data directory (default: data)")
    parser.add_argument(
        "--subset", type=int, default=None, help="keep only N clips (fast local runs)"
    )
    parser.add_argument("--force", action="store_true", help="re-download / re-extract everything")
    args = parser.parse_args(argv)

    raw = Path(args.data_dir) / "raw"
    raw.mkdir(parents=True, exist_ok=True)

    with requests.Session() as session:
        if not _download_mp3s(raw, session, force=args.force):
            return 1
        if not _download_annotations(raw, session, force=args.force):
            return 1

    _extract(raw, args.subset, force=args.force)

    n_clips = sum(1 for _ in raw.rglob("*.mp3"))
    print(f"ready: {n_clips} clips in {raw}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
