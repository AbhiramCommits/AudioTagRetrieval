"""Tests for the CNN and transformer tagging models."""

import pytest
import torch

from audiotag.models import AudioTransformerTagger, ShortChunkCNN, build_model

BATCH = 4
T_FRAMES = 313


@pytest.fixture(params=["cnn", "transformer"])
def model(request):
    return build_model(request.param, n_classes=50)


def test_forward_and_embed_shapes(model):
    x = torch.randn(BATCH, 1, 128, T_FRAMES)
    embed = model.embed(x)
    logits = model(x)
    assert logits.shape == (BATCH, 50)
    if isinstance(model, ShortChunkCNN):
        assert embed.shape == (BATCH, 1024)
    else:
        assert embed.shape == (BATCH, model.d_model)


def test_gradients_flow(model):
    x = torch.randn(BATCH, 1, 128, T_FRAMES)
    loss = model(x).sum()
    loss.backward()
    params = list(model.parameters())
    grads = [p.grad for p in params if p.grad is not None]
    assert grads
    assert all(g.abs().sum() > 0 for g in grads)


def test_transformer_rejects_too_many_patches():
    model = AudioTransformerTagger(n_classes=50, max_frames=128)
    x = torch.randn(1, 1, 128, 1024)  # 8 x 64 = 512 patches > max 8*8
    with pytest.raises(ValueError, match="max is"):
        model.embed(x)


def test_deterministic_with_fixed_seed():
    x = torch.randn(2, 1, 128, T_FRAMES)
    outputs = []
    for _ in range(2):
        torch.manual_seed(42)
        model = ShortChunkCNN(n_classes=50)
        outputs.append(model.embed(x).detach())
    assert torch.equal(outputs[0], outputs[1])


def test_build_model_rejects_unknown():
    with pytest.raises(ValueError, match="unknown model"):
        build_model("lstm", 50)
