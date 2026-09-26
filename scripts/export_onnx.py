#!/usr/bin/env python
"""Export the trained tagger to ONNX and benchmark it against PyTorch.

Exports a wrapper that computes the logits and the embedding in a single
pass (dynamic batch and time axes), verifies ONNXRuntime output matches
PyTorch within 1e-4, and writes a latency comparison to
``reports/onnx_comparison.md``.

Usage:
    python scripts/export_onnx.py
    python scripts/export_onnx.py --model cnn --runs 100
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from audiotag.config import Config  # noqa: E402
from audiotag.models import build_model  # noqa: E402


class ExportWrapper(nn.Module):
    """Runs the conv stack once and returns (logits, embedding)."""

    def __init__(self, model: nn.Module) -> None:
        super().__init__()
        self.model = model

    def forward(self, mel: torch.Tensor):
        embedding = self.model.embed(mel)
        logits = self.model.head(embedding)
        return logits, embedding


def bench_torch(wrapper: nn.Module, x: torch.Tensor, runs: int) -> float:
    with torch.inference_mode():
        for _ in range(5):
            wrapper(x)
        start = time.perf_counter()
        for _ in range(runs):
            wrapper(x)
    return (time.perf_counter() - start) * 1000.0 / runs


def bench_ort(session, x: np.ndarray, runs: int) -> float:
    for _ in range(5):
        session.run(None, {"mel": x})
    start = time.perf_counter()
    for _ in range(runs):
        session.run(None, {"mel": x})
    return (time.perf_counter() - start) * 1000.0 / runs


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=["cnn", "transformer"], default="cnn")
    parser.add_argument("--checkpoint", default=None, help="default: artifacts/{model}/best.pt")
    parser.add_argument("--out", default=None, help="default: artifacts/{model}/model.onnx")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--runs", type=int, default=50)
    args = parser.parse_args(argv)

    import onnx
    import onnxruntime as ort

    cfg = Config()
    checkpoint = (
        Path(args.checkpoint) if args.checkpoint else Path("artifacts") / args.model / "best.pt"
    )
    if not checkpoint.exists():
        print(f"error: checkpoint {checkpoint} not found; run training first")
        return 1
    out_path = Path(args.out) if args.out else Path("artifacts") / args.model / "model.onnx"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    ckpt = torch.load(checkpoint, map_location="cpu", weights_only=False)
    if ckpt["model"] != "cnn":
        print(
            "error: ONNX export only supports the CNN tagger; the transformer export "
            "trips an ONNXRuntime LayerNormalization buffer-reuse bug under dynamic "
            "batch sizes",
            file=sys.stderr,
        )
        return 1
    model = build_model(ckpt["model"], n_classes=len(ckpt["tags"])).to(args.device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    wrapper = ExportWrapper(model).to(args.device).eval()

    dummy = torch.randn(1, 1, cfg.n_mels, cfg.n_frames, device=args.device)
    torch.onnx.export(
        wrapper,
        dummy,
        str(out_path),
        input_names=["mel"],
        output_names=["logits", "embedding"],
        dynamic_axes={
            "mel": {0: "batch", 3: "time"},
            "logits": {0: "batch"},
            "embedding": {0: "batch"},
        },
        opset_version=17,
    )
    onnx.checker.check_model(onnx.load(out_path))

    session = ort.InferenceSession(str(out_path), providers=["CPUExecutionProvider"])
    print(f"exported {out_path} ({out_path.stat().st_size / 1e6:.2f} MB)")

    # Verification across batch sizes and time lengths.
    max_diff = 0.0
    for batch, frames in ((1, cfg.n_frames), (2, cfg.n_frames), (1, 200), (3, 150)):
        x = torch.randn(batch, 1, cfg.n_mels, frames, device=args.device)
        with torch.inference_mode():
            logits_t, emb_t = wrapper(x)
        logits_o, emb_o = session.run(None, {"mel": x.cpu().numpy()})
        max_diff = max(
            max_diff,
            float((logits_t.cpu() - torch.from_numpy(logits_o)).abs().max()),
            float((emb_t.cpu() - torch.from_numpy(emb_o)).abs().max()),
        )
    print(f"max |torch - onnxruntime| over shapes: {max_diff:.2e}")
    if max_diff >= 1e-4:
        print("error: outputs diverge beyond 1e-4", file=sys.stderr)
        return 1

    # Single-clip latency, both engines.
    torch.set_num_threads(2)
    x = torch.randn(1, 1, cfg.n_mels, cfg.n_frames)
    torch_ms = bench_torch(wrapper, x, args.runs)
    ort_ms = bench_ort(session, x.numpy(), args.runs)

    report = f"""# ONNX vs PyTorch

Single-clip inference (`[1, 1, 128, 313]` mel), {args.model} model, CPU,
mean over {args.runs} runs after warmup (torch with 2 intra-op threads,
onnxruntime {ort.__version__} CPU provider).

| engine | mean latency (ms) | artifact |
|---|---|---|
| PyTorch (torch.inference_mode) | {torch_ms:.2f} | {checkpoint} |
| ONNX Runtime | {ort_ms:.2f} | {out_path} |

Max absolute output difference across batch sizes 1-3 and time lengths
150-{cfg.n_frames} frames: {max_diff:.2e} (< 1e-4).

The API serves either engine via `AUDIOTAG_BACKEND=torch|onnx`.
"""
    reports_dir = Path("reports")
    reports_dir.mkdir(parents=True, exist_ok=True)
    (reports_dir / "onnx_comparison.md").write_text(report)
    print(report)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
