"""Tests for the torch/onnx inference engines."""

import numpy as np
import pytest
import torch
from torch import nn

from audiotag.api.engines import OnnxEngine, TorchEngine, build_engine
from tests.helpers import TinyEmbedModel


def test_torch_engine_run_shapes():
    engine = TorchEngine(TinyEmbedModel(n_classes=5, dim=8), torch.device("cpu"))
    mel = torch.randn(3, 1, 128, 313)
    logits, embedding = engine.run(mel)
    assert logits.shape == (3, 5)
    assert embedding.shape == (3, 8)


def test_build_engine_defaults_to_torch():
    engine = build_engine("torch", TinyEmbedModel(5, 8), torch.device("cpu"))
    assert isinstance(engine, TorchEngine)


def test_build_engine_rejects_unknown():
    with pytest.raises(RuntimeError, match="unknown"):
        build_engine("tflite", TinyEmbedModel(5, 8), torch.device("cpu"))


def test_onnx_engine_matches_torch(tmp_path):
    pytest.importorskip("onnxruntime")

    class Wrapper(nn.Module):
        def __init__(self):
            super().__init__()
            self.embed_layer = nn.Linear(8, 8)
            self.head_layer = nn.Linear(8, 5)

        def forward(self, x):
            emb = self.embed_layer(x)
            return self.head_layer(emb), emb

    wrapper = Wrapper().eval()
    path = tmp_path / "tiny.onnx"
    torch.onnx.export(
        wrapper,
        torch.randn(2, 8),
        str(path),
        input_names=["mel"],
        output_names=["logits", "embedding"],
        dynamic_axes={"mel": {0: "batch"}, "logits": {0: "batch"}, "embedding": {0: "batch"}},
        opset_version=17,
    )

    engine = OnnxEngine(
        build_session := __import__("onnxruntime").InferenceSession(
            str(path), providers=["CPUExecutionProvider"]
        )
    )
    del build_session

    x = torch.randn(4, 8)
    with torch.inference_mode():
        logits_t, emb_t = wrapper(x)
    logits_o, emb_o = engine.run(x)
    assert logits_o.shape == (4, 5)
    assert emb_o.shape == (4, 8)
    assert np.abs(logits_t.numpy() - logits_o.numpy()).max() < 1e-5
    assert np.abs(emb_t.numpy() - emb_o.numpy()).max() < 1e-5
