"""
Lightweight model loader / predictor for unit tests.
Real project can swap DummyModel out for a real checkpoint later.
"""
from __future__ import annotations
import os
import hashlib
from typing import Tuple


class DummyModel:
    """Very small deterministic model used only for unit tests."""
    def __init__(self, ckpt_path: str):
        self.ckpt_path = ckpt_path

    def predict(self, text: str) -> Tuple[str, float]:
        # deterministic pseudo-prediction based on hash of text
        h = int(hashlib.sha256(text.encode()).hexdigest(), 16)
        labels = ["joy", "sadness", "anger", "neutral"]
        label = labels[h % len(labels)]
        confidence = round(((h % 100) / 100), 2)  # 0.00 … 0.99
        return label, confidence


def load_model(weights_path: str) -> DummyModel:
    """
    Raise FileNotFoundError if checkpoint is missing, else return model.
    """
    if not os.path.exists(weights_path):
        raise FileNotFoundError(f"Checkpoint not found: {weights_path}")
    return DummyModel(weights_path)


def predict(model: DummyModel, text: str):
    """
    Convenience wrapper so higher code can call predict(model, text).
    """
    return model.predict(text)
