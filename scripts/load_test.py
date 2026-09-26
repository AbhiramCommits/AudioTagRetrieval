#!/usr/bin/env python
"""Async load generator for the serving API.

Posts real audio clips to /analyze with configurable concurrency and
duration, with a warmup phase excluded from the stats, and reports
p50/p95/p99 latency, throughput (QPS), error rate, and mean per-stage
timings (decode/mel/forward/faiss) from the X-Stage-* response headers.

Usage:
    python scripts/load_test.py --concurrency 8 --duration 30 --warmup 5
    python scripts/load_test.py --concurrency 32 --duration 30 --out reports/serving_benchmark.jsonl
"""

from __future__ import annotations

import argparse
import asyncio
import json
import random
import sys
import time
from pathlib import Path

import httpx

STAGE_HEADERS = ["decode", "mel", "forward", "faiss"]


def percentile(sorted_values: list[float], pct: float) -> float:
    if not sorted_values:
        return 0.0
    idx = min(len(sorted_values) - 1, int(pct / 100.0 * len(sorted_values)))
    return sorted_values[idx]


async def load(
    url: str, concurrency: int, duration: float, warmup: float, audio_files: list[Path]
) -> dict:
    latencies: list[float] = []
    errors = 0
    requests = 0
    stage_sums = {name: 0.0 for name in STAGE_HEADERS}
    stage_count = 0

    async def worker(client: httpx.AsyncClient, stop_at: float, measure: bool):
        nonlocal latencies, errors, requests, stage_sums, stage_count
        rng = random.Random()
        while time.monotonic() < stop_at:
            path = rng.choice(audio_files)
            content = path.read_bytes()
            start = time.perf_counter()
            try:
                response = await client.post(
                    f"{url}/analyze",
                    files={"file": (path.name, content, "audio/mpeg")},
                    timeout=60.0,
                )
                latency = (time.perf_counter() - start) * 1000.0
                if measure and response.status_code == 200:
                    latencies.append(latency)
                    requests += 1
                    for name in STAGE_HEADERS:
                        value = response.headers.get(f"X-Stage-{name.title()}-Ms")
                        if value is not None:
                            stage_sums[name] += float(value)
                    stage_count += 1
                elif measure:
                    errors += 1
                    requests += 1
            except httpx.HTTPError:
                if measure:
                    errors += 1
                    requests += 1

    async with httpx.AsyncClient(limits=httpx.Limits(max_connections=concurrency + 8)) as client:
        # Warmup phase: exercised but not measured.
        warmup_until = time.monotonic() + warmup
        await asyncio.gather(*[worker(client, warmup_until, False) for _ in range(concurrency)])

        stop_at = time.monotonic() + duration
        t0 = time.perf_counter()
        await asyncio.gather(*[worker(client, stop_at, True) for _ in range(concurrency)])
        measured = time.perf_counter() - t0

    latencies.sort()
    qps = requests / measured if measured > 0 else 0.0
    return {
        "concurrency": concurrency,
        "duration": duration,
        "warmup": warmup,
        "requests": requests,
        "errors": errors,
        "error_rate": errors / requests if requests else 0.0,
        "qps": qps,
        "p50_ms": percentile(latencies, 50),
        "p95_ms": percentile(latencies, 95),
        "p99_ms": percentile(latencies, 99),
        "stages_ms": {name: stage_sums[name] / stage_count for name in STAGE_HEADERS}
        if stage_count
        else {},
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default="http://127.0.0.1:8000")
    parser.add_argument("--concurrency", type=int, default=1)
    parser.add_argument("--duration", type=float, default=30.0)
    parser.add_argument("--warmup", type=float, default=5.0)
    parser.add_argument("--audio-dir", default="data/raw")
    parser.add_argument("--max-audio", type=int, default=200)
    parser.add_argument("--out", default=None, help="append a JSON record to this file")
    args = parser.parse_args(argv)

    audio_files = sorted(Path(args.audio_dir).rglob("*.mp3"))[: args.max_audio]
    if not audio_files:
        print(f"error: no mp3s found under {args.audio_dir}", file=sys.stderr)
        return 1
    print(
        f"load test: {len(audio_files)} clips, concurrency={args.concurrency}, "
        f"duration={args.duration}s, warmup={args.warmup}s",
        flush=True,
    )

    result = asyncio.run(load(args.url, args.concurrency, args.duration, args.warmup, audio_files))

    print(
        f"requests={result['requests']} errors={result['errors']} "
        f"error_rate={result['error_rate'] * 100:.2f}% qps={result['qps']:.2f}"
    )
    print(
        f"latency ms: p50={result['p50_ms']:.1f} p95={result['p95_ms']:.1f} "
        f"p99={result['p99_ms']:.1f}"
    )
    if result["stages_ms"]:
        stage_str = " ".join(f"{name}={result['stages_ms'][name]:.1f}ms" for name in STAGE_HEADERS)
        print(f"stage means: {stage_str}")
    else:
        print("stage means: (no successful responses)")

    if args.out:
        with open(args.out, "a") as fh:
            fh.write(json.dumps(result) + "\n")
        print(f"appended result to {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
