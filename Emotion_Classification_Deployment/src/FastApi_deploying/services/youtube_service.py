"""
Download a YouTube video, extract the audio track, transcribe it with Whisper
and return the transcript.

The function is tiny on purpose – in unit-tests we monkey-patch
`subprocess.run` and `whisper.load_model`, so **no real network calls** and
no heavy models are pulled in CI.
"""

from __future__ import annotations
import logging
import subprocess
import tempfile
from pathlib import Path

import whisper     # mocked in tests

log = logging.getLogger(__name__)   # module-level logger


def download_and_transcribe(url: str) -> str:
    """Return transcript text for the given YouTube *url*."""
    log.info("Start processing YouTube URL: %s", url)

    # 1. Download & convert to mp3 (yt-dlp does all the heavy lifting)
    with tempfile.TemporaryDirectory() as tmpdir:
        mp3_path = Path(tmpdir, "audio.mp3")
        cmd = [
            "yt-dlp", "-x", "--audio-format", "mp3",
            "-o", str(mp3_path), url,
        ]
        log.info("Running command: %s", " ".join(cmd))
        subprocess.run(cmd, check=True)
        log.info("Download finished, saved to %s", mp3_path)

        # 2. Transcribe with Whisper
        model = whisper.load_model("base")
        log.info("Whisper model loaded, starting transcription")
        transcript = model.transcribe(str(mp3_path))["text"]

    log.info("Transcription completed (%.0f chars)", len(transcript))
    return transcript
