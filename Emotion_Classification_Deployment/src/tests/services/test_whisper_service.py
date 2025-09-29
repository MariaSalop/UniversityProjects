from FastApi_deploying.services import whisper_service


def test_transcribe_audio(monkeypatch):
    fake_result = {"text": "hello world"}

    class _FakeModel:
        def transcribe(self, f):
            return fake_result

    monkeypatch.setattr(whisper_service.whisper, "load_model",
                        lambda name="base": _FakeModel())

    text = whisper_service.transcribe_audio(b"RIFF....")
    assert text == "hello world"
