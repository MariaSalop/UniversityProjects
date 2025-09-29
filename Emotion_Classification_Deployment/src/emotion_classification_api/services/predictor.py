"""
Core prediction logic extracted from FastAPI.py.
This module can be tested independently of HTTP.
"""

import os
from functools import lru_cache
from typing import List

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

# English: these are the emotion labels, same as in FastAPI
LABELS: List[str] = [
    "anger", "joy", "optimism", "sadness",
    "surprise", "fear", "disgust",
]


@lru_cache(maxsize=1)
def _load_pipeline(
    model_path: str = os.getenv("MODEL_PATH", "Abida4ka/fastapi-deploying-model"),
    hf_token: str | None = os.getenv("HF_TOKEN"),
):
    """Lazy-load tokenizer and model once per process."""
    tokenizer = AutoTokenizer.from_pretrained(model_path, token=hf_token)
    model = AutoModelForSequenceClassification.from_pretrained(model_path, token=hf_token)
    model.eval()
    return tokenizer, model


def predict_text(text: str):
    """
    Given a piece of text, return a dict with:
      - predicted_class: index of the label
      - predicted_label: label string
      - probabilities: list of floats
    """
    tok, mdl = _load_pipeline()
    inputs = tok(text, return_tensors="pt", truncation=True, padding=True)
    with torch.no_grad():
        outputs = mdl(**inputs)

    # Support both real tensors and lists from fake stubs
    raw = torch.softmax(outputs.logits, dim=1)[0]
    if hasattr(raw, "tolist"):
        probs = raw.tolist()
    else:
        probs = list(raw)

    # Determine the class index with the highest probability
    idx = max(range(len(probs)), key=probs.__getitem__)
    return {
        "predicted_class": idx,
        "predicted_label": LABELS[idx],
        "probabilities": probs,
    }
