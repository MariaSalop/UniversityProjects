import io
import subprocess
import pandas as pd
import pytest
import torch
from fastapi.testclient import TestClient

# import our modules
from emotion_classification_api.services.predictor import predict_text, LABELS
from emotion_classification_api.services.csv_processor import process_text_csv
from emotion_classification_api.services.transcriber import (
    transcribe_audio_bytes,
    transcribe_youtube,
)
from emotion_classification_api.FastAPI_services import app

client = TestClient(app)


# ─── Section 1: Predictor tests ─────────────────────────────────────────────
class DummyTok:
    def __call__(self, text, **kw):
        return {"input_ids": [[0]], "attention_mask": [[1]]}


class DummyMdl:
    def __call__(self, **inputs):
        return type("Res", (), {"logits": torch.tensor([[0.3, 0.7]])})

    def eval(self): return self


@pytest.fixture(autouse=True)
def fake_pipeline(monkeypatch):
    """Stub out HF loader so no real model is fetched."""
    monkeypatch.setenv("MODEL_PATH", "dummy")
    monkeypatch.setattr(
        "emotion_classification_api.services.predictor._load_pipeline",
        lambda: (DummyTok(), DummyMdl())
    )


def test_predict_text_returns_expected():
    """Test core predict_text logic."""
    res = predict_text("hello")
    assert res["predicted_class"] == 1
    assert res["predicted_label"] == LABELS[1]
    assert pytest.approx(res["probabilities"]) == [0.3, 0.7]


# ─── Section 2: CSV processor tests ──────────────────────────────────────────

@pytest.fixture(autouse=True)
def fake_predict_for_csv(monkeypatch):
    """Make predict_text() return uppercase label."""
    monkeypatch.setattr(
        "emotion_classification_api.services.csv_processor.predict_text",
        lambda txt: {
            "predicted_label": txt.upper(),
            "predicted_class": 0,
            "probabilities": [],
        }
    )


def test_process_text_csv_success():
    """Ensure CSV with text column is processed correctly."""
    df = pd.DataFrame({"text": ["a", "b", ""]})
    buf = io.StringIO(df.to_csv(index=False))
    out_csv = process_text_csv(buf)
    lines = out_csv.strip().splitlines()
    assert lines[0] == "text,predicted_label"
    assert "a,A" in lines[1]
    assert "b,B" in lines[2]


def test_process_text_csv_missing_column():
    """Missing 'text' column should raise ValueError."""
    buf = io.StringIO("foo,bar\nx,y")
    with pytest.raises(ValueError):
        process_text_csv(buf)


# ─── Section 3: Transcriber tests ────────────────────────────────────────────

def test_transcribe_audio_bytes(monkeypatch):
    """transcribe_audio_bytes should return model text."""
    fake = {"text": "audio text"}

    class FakeModel:
        def transcribe(self, path): return fake
    monkeypatch.setattr(
        "emotion_classification_api.services.transcriber.whisper.load_model",
        lambda *a, **k: FakeModel()
    )
    out = transcribe_audio_bytes(b"RIFF")
    assert out == "audio text"


def test_transcribe_youtube(monkeypatch, tmp_path):
    """transcribe_youtube should invoke yt-dlp and return model text."""
    fake = {"text": "yt text"}

    class FakeModel:
        def transcribe(self, path): return fake
    # stub subprocess and whisper
    monkeypatch.setattr(subprocess, "run", lambda *a, **k: None)
    monkeypatch.setattr(
        "emotion_classification_api.services.transcriber.whisper.load_model",
        lambda *a, **k: FakeModel()
    )
    out = transcribe_youtube("https://youtu.be/dQw4w9WgXcQ")
    assert out == "yt text"


# ─── Section 4: HTTP smoke test ───────────────────────────────────────────────

def test_predict_http_smoke(monkeypatch):
    """Simple FastAPI predict endpoint check."""
    # stub internal logic
    monkeypatch.setattr(
        "emotion_classification_api.services.predictor.predict_text",
        lambda txt: {
            "predicted_class": 0,
            "predicted_label": "anger",
            "probabilities": [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        }
    )
    r = client.post("/predict", json={"text": "hey"})
    assert r.status_code == 200
    js = r.json()
    assert js["predicted_label"] == "anger"
    assert len(js["probabilities"]) == 7
