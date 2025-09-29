# tests/local_api/test_local_api_endpoints.py

import io
import subprocess
import pytest
from fastapi.testclient import TestClient

from emotion_classification_api.FastAPI_services import app

client = TestClient(app)


def test_root():
    """GET / should return a greeting."""
    r = client.get("/")
    assert r.status_code == 200
    assert r.json() == {"message": "Emotion Classification API is running."}


def test_predict_http(monkeypatch):
    """POST /predict returns what predictor.predict_text returns."""
    fake = {
        "text": "foo",
        "predicted_class": 2,
        "predicted_label": "sadness",
        "probabilities": [0.0, 0.0, 1.0],
    }
    monkeypatch.setattr(
        "emotion_classification_api.services.predictor.predict_text",
        lambda t: fake
    )
    r = client.post("/predict", json={"text": "ignored"})
    assert r.status_code == 200
    assert r.json() == fake


def test_upload_csv_success(monkeypatch):
    """POST /upload-csv returns CSV through StreamingResponse."""
    csv_out = "text,predicted_label\nhello,JOY\n"
    monkeypatch.setattr(
        "emotion_classification_api.services.csv_processor.process_text_csv",
        lambda f: csv_out
    )
    body = io.BytesIO(b"text\nhello\n")
    r = client.post(
        "/upload-csv",
        files={"file": ("in.csv", body, "text/csv")}
    )
    assert r.status_code == 200
    assert r.headers["content-type"].startswith("text/csv")
    assert r.headers["content-disposition"] == "attachment; filename=predicted.csv"
    assert r.content.decode() == csv_out


@pytest.mark.parametrize("exc, code, msg", [
    (ValueError("no text"), 400, "no text"),
    (RuntimeError("oops"), 500, "Internal error"),
])
def test_upload_csv_errors(monkeypatch, exc, code, msg):
    """POST /upload-csv raises 400 on ValueError and 500 on other exceptions."""
    monkeypatch.setattr(
        "emotion_classification_api.services.csv_processor.process_text_csv",
        lambda f: (_ for _ in ()).throw(exc)
    )
    body = io.BytesIO(b"text\nhello\n")
    r = client.post(
        "/upload-csv",
        files={"file": ("in.csv", body, "text/csv")}
    )
    assert r.status_code == code
    # in 400, detail = str(exc); in 500, detail = "Internal error"
    assert msg in (r.json().get("detail") or "")


def test_youtube_transcript_success(monkeypatch):
    """POST /youtube-transcript returns JSON with transcript."""
    monkeypatch.setattr(
        "emotion_classification_api.services.transcriber.transcribe_youtube",
        lambda url: "YT OK"
    )
    r = client.post("/youtube-transcript", json={"url": "any"})
    assert r.status_code == 200
    assert r.json() == {"transcript": "YT OK"}


def test_youtube_transcript_download_error(monkeypatch):
    """If transcribe_youtube raises CalledProcessError → 400."""
    def fail(url):
        raise subprocess.CalledProcessError(1, cmd="yt-dlp")
    monkeypatch.setattr(
        "emotion_classification_api.services.transcriber.transcribe_youtube",
        fail
    )
    r = client.post("/youtube-transcript", json={"url": "any"})
    assert r.status_code == 400
    assert "Failed to download audio" in r.json()["detail"]


def test_youtube_transcript_other_error(monkeypatch):
    """If transcribe_youtube raises any other exception → 500."""
    monkeypatch.setattr(
        "emotion_classification_api.services.transcriber.transcribe_youtube",
        lambda url: (_ for _ in ()).throw(Exception("fail"))
    )
    r = client.post("/youtube-transcript", json={"url": "any"})
    assert r.status_code == 500
    assert "Failed to transcribe video" in r.json()["detail"]


def test_audio_transcript_success(monkeypatch):
    """POST /upload-audio-transcript returns transcript."""
    monkeypatch.setattr(
        "emotion_classification_api.services.transcriber.transcribe_audio_bytes",
        lambda b: "AUDIO OK"
    )
    body = io.BytesIO(b"WAVDATA")
    r = client.post(
        "/upload-audio-transcript",
        files={"file": ("in.wav", body, "audio/wav")}
    )
    assert r.status_code == 200
    assert r.json() == {"transcript": "AUDIO OK"}


def test_audio_transcript_error(monkeypatch):
    """If transcribe_audio_bytes fails → 500."""
    monkeypatch.setattr(
        "emotion_classification_api.services.transcriber.transcribe_audio_bytes",
        lambda b: (_ for _ in ()).throw(Exception("bad"))
    )
    body = io.BytesIO(b"WAVDATA")
    r = client.post(
        "/upload-audio-transcript",
        files={"file": ("in.wav", body, "audio/wav")}
    )
    assert r.status_code == 500
    assert "Failed to transcribe audio" in r.json()["detail"]
