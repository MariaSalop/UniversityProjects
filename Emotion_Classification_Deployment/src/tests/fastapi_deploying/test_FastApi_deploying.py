import io
import textwrap
import pathlib


def test_root_and_health(fastapi_client):
    r1 = fastapi_client.get("/")
    r2 = fastapi_client.get("/health")
    assert r1.status_code == 200 and "running" in r1.json()["message"].lower()
    assert r2.status_code == 200 and r2.json() == {"status": "healthy"}


def test_predict_endpoint(fastapi_client):
    resp = fastapi_client.post("/predict", json={"text": "I am happy"})
    js = resp.json()
    assert resp.status_code == 200
    assert set(js) >= {"text", "predicted_label", "probabilities"}
    assert len(js["probabilities"]) == 7


def test_upload_csv_endpoint(fastapi_client, tmp_path):
    csv_content = textwrap.dedent("""\
        text
        Hello world
        I am tired
    """)
    csv_bytes = io.BytesIO(csv_content.encode())
    resp = fastapi_client.post(
        "/upload-csv",
        files={"file": ("sample.csv", csv_bytes, "text/csv")},
    )
    assert resp.status_code == 200
    out_path = pathlib.Path(resp.json()["file_path"])
    assert out_path.name == "csv_predictions.csv"


def test_youtube_stub(fastapi_client, monkeypatch):
    monkeypatch.setattr("subprocess.run", lambda *a, **k: None)
    import nltk
    monkeypatch.setattr(nltk, "sent_tokenize", lambda txt: ["first sentence", "second"])
    resp = fastapi_client.post(
        "/youtube-transcript",
        params={"url": "https://youtu.be/dQw4w9WgXcQ"},
    )
    assert resp.status_code == 200
    assert resp.json()["file_path"].endswith("youtube_transcript.csv")
