"""Model zoo for audio tagging."""

from audiotag.models.cnn import ShortChunkCNN
from audiotag.models.transformer import AudioTransformerTagger

__all__ = ["AudioTransformerTagger", "ShortChunkCNN", "build_model"]


def build_model(name: str, n_classes: int = 50):
    """Instantiate a tagging model by name (``cnn`` or ``transformer``)."""
    name = name.lower()
    if name == "cnn":
        return ShortChunkCNN(n_classes=n_classes)
    if name == "transformer":
        return AudioTransformerTagger(n_classes=n_classes)
    raise ValueError(f"unknown model {name!r} (expected 'cnn' or 'transformer')")
