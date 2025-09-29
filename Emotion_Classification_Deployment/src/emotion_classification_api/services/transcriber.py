"""
Extracted file-transcription logic for both YouTube and audio-upload.
"""

import tempfile
import subprocess
from pathlib import Path
import whisper
import nltk


def transcribe_audio_bytes(blob: bytes) -> str:
    """Write blob to temp .wav, run whisper, return raw text."""
    nltk.download('punkt', quiet=True)

    with tempfile.NamedTemporaryFile(suffix=".wav") as tmp:
        tmp.write(blob)
        tmp.flush()
        model = whisper.load_model("base")
        return model.transcribe(tmp.name)["text"]


def transcribe_youtube(url: str) -> str:
    """Download & extract audio via yt-dlp, run whisper, return raw text."""
    nltk.download('punkt', quiet=True)

    with tempfile.TemporaryDirectory() as d:
        mp3 = Path(d) / "audio.mp3"
        subprocess.run(
            ["yt-dlp", "-x", "--audio-format", "mp3", "-o", str(mp3), url],
            check=True,
        )
        model = whisper.load_model("base")
        return model.transcribe(str(mp3))["text"]
