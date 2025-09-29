"""
Light wrapper around HF model; initialises once and can be mocked in tests.
The heavy objects are created lazily so importing this module stays cheap.
"""
from __future__ import annotations
import os
import functools
from typing import Tuple, List

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

LABELS: list[str] = [
    "anger", "joy", "optimism", "sadness",
    "surprise", "fear", "disgust",
]


@functools.cache
def _load_pipeline(
    model_path: str = "Abida4ka/fastapi-deploying-model",
    hf_token: str | None = os.getenv("HF_TOKEN"),
) -> Tuple[AutoTokenizer, AutoModelForSequenceClassification]:
    tok = AutoTokenizer.from_pretrained(model_path, token=hf_token)
    mdl = AutoModelForSequenceClassification.from_pretrained(
        model_path, token=hf_token
    )
    mdl.eval()
    return tok, mdl


def predict(text: str) -> Tuple[str, List[float]]:
    tok, mdl = _load_pipeline()

    # Stub in case tests pass in (None, None)
    if tok is None or mdl is None:
        one_hot = [1.0] + [0.0] * (len(LABELS) - 1)
        return LABELS[0], one_hot

    inputs = tok(text, return_tensors="pt", truncation=True, padding=True)
    with torch.no_grad():
        out = mdl(**inputs)

    raw_probs = torch.softmax(out.logits, dim=1)[0]

    # Properly convert to a Python list regardless of type
    probs: List[float]
    if hasattr(raw_probs, "tolist"):          # real torch.Tensor
        probs = raw_probs.tolist()
    else:                                     # already a list (fake torch)
        probs = list(raw_probs)

    # Find the max index "Python-style"
    idx = max(range(len(probs)), key=probs.__getitem__)
    return LABELS[idx], probs
