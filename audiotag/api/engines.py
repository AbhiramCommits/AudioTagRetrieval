"""Inference engines: PyTorch or ONNX Runtime, behind one interface."""

from __future__ import annotations

import numpy as np
import torch
from torch import nn


class TorchEngine:
    """Runs the tagger under torch.inference_mode()."""

    def __init__(self, model: nn.Module, device: torch.device) -> None:
        self.model = model.to(device).eval()
        self.device = device

    def run(self, mel: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        with torch.inference_mode():
            embedding = self.model.embed(mel)
            logits = self.model.head(embedding)
        return logits, embedding


class OnnxEngine:
    """Runs an exported ONNX graph (outputs: logits, embedding)."""

    def __init__(self, session) -> None:
        self.session = session

    def run(self, mel: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        logits, embedding = self.session.run(
            ["logits", "embedding"], {"mel": mel.detach().cpu().numpy().astype(np.float32)}
        )
        return torch.from_numpy(logits), torch.from_numpy(embedding)


def build_engine(backend: str, model: nn.Module, device: torch.device, onnx_path=None):
    """Instantiate the requested engine (``torch`` or ``onnx``)."""
    if backend == "onnx":
        import onnxruntime as ort

        if onnx_path is None:
            raise RuntimeError("onnx backend requires an exported model path")
        session = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
        return OnnxEngine(session)
    if backend == "torch":
        return TorchEngine(model, device)
    raise RuntimeError(f"unknown engine backend {backend!r} (expected 'torch' or 'onnx')")
