"""Whisper transcription kept in a tiny helper for easy mocking."""
import tempfile
import whisper


def transcribe_audio(blob: bytes) -> str:
    with tempfile.NamedTemporaryFile(suffix=".wav") as tmp:
        tmp.write(blob)
        tmp.flush()
        model = whisper.load_model("base")
        return model.transcribe(tmp.name)["text"]
