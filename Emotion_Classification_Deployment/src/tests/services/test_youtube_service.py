# from FastApi_deploying.services import youtube_service
from FastApi_deploying.services import youtube_service
import subprocess


def test_download_and_transcribe_invokes_subprocess(monkeypatch, tmp_path):
    called = {}

    # stub subprocess.run so we capture the command
    def fake_run(cmd, check):
        called["cmd"] = cmd
    monkeypatch.setattr(subprocess, "run", fake_run)

    # dummy TemporaryDirectory context manager
    class DummyTempDir:
        def __enter__(self):
            # return our tmp_path as "working dir"
            return tmp_path

        def __exit__(self, exc_type, exc_val, exc_tb):
            # no suppression
            return False

    # patch only youtube_service.tempfile.TemporaryDirectory,
    # leaving real tempfile untouched
    monkeypatch.setattr(
        youtube_service.tempfile,
        "TemporaryDirectory",
        lambda: DummyTempDir()
    )

    # stub whisper.load_model → fake model that returns {"text": ...}
    class FakeModel:
        def transcribe(self, path):
            return {"text": "ok"}
    monkeypatch.setattr(
        youtube_service.whisper,
        "load_model",
        lambda *a, **k: FakeModel()
    )

    url = "https://youtu.be/dQw4w9WgXcQ"
    text = youtube_service.download_and_transcribe(url)

    # assert that yt-dlp was invoked and we got "ok" back
    assert "yt-dlp" in " ".join(called["cmd"])
    assert text == "ok"
