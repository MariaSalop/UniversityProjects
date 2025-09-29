"""
Endpoints that create CSV transcripts from user-supplied data:
  • POST /upload-csv                – plain CSV with a 'text' column
  • POST /youtube-transcript        – YouTube URL  → transcript CSV
  • POST /upload-audio-transcript   – audio bytes  → transcript CSV
"""

from __future__ import annotations

import csv
import io
import logging
import os
import tempfile
from pathlib import Path

import nltk
from fastapi import APIRouter, HTTPException, UploadFile, File

from ..services import youtube_service, whisper_service

router = APIRouter()
log = logging.getLogger(__name__)

# -------- output directory (redirected to tmp in tests) --------
_APP_OUT = Path(os.getenv("APP_OUTPUT_DIR", tempfile.gettempdir())) / "app_output"


def _ensure_out_dir() -> Path:
    """Make sure the output directory exists and is writable."""
    try:
        _APP_OUT.mkdir(parents=True, exist_ok=True)
        return _APP_OUT
    except (OSError, PermissionError):
        # read-only FS inside CI container – fall back to a private tmp dir
        return Path(tempfile.mkdtemp())


# --------------------- helpers ---------------------------------
def _save_csv(text: str, filename: str) -> str:
    """Tokenise into sentences → write one per line → return file path."""
    out_dir = _ensure_out_dir()
    buf = io.StringIO()
    writer = csv.writer(buf)
    for sentence in nltk.sent_tokenize(text):
        writer.writerow([sentence.strip()])
    dest = out_dir / filename
    dest.write_text(buf.getvalue(), encoding="utf-8")
    return str(dest)


# --------------------- endpoints -------------------------------

@router.post("/upload-csv")
async def upload_csv(file: UploadFile = File(...)):
    """
    Accept a CSV file with a `text` column, append a fake `prediction`
    column (real code could call model_service.predict) and save to disk.
    """
    try:
        import pandas as pd  # local import keeps startup fast
        df = pd.read_csv(file.file)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Bad CSV: {exc}") from exc

    if "text" not in df.columns:
        raise HTTPException(status_code=400, detail="Missing 'text' column")

    df["prediction"] = "neutral"
    dest = _ensure_out_dir() / "csv_predictions.csv"
    df.to_csv(dest, index=False)
    return {"file_path": str(dest)}


@router.post("/youtube-transcript")
async def youtube_transcript(url: str):
    """
    Download YouTube audio, transcribe with Whisper, save transcript CSV.
    """
    text = youtube_service.download_and_transcribe(url)
    path = _save_csv(text, "youtube_transcript.csv")
    return {"file_path": path}


@router.post("/upload-audio-transcript")
async def upload_audio_transcript(file: UploadFile = File(...)):
    """
    Receive arbitrary audio (wav/mp3...), transcribe with Whisper,
    save transcript CSV.
    """
    text = whisper_service.transcribe_audio(await file.read())
    path = _save_csv(text, "uploaded_audio_transcript.csv")
    return {"file_path": path}
