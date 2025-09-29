import logging
import subprocess

from fastapi import FastAPI, UploadFile, File, HTTPException, Body
from fastapi.responses import StreamingResponse

from pydantic import BaseModel

import emotion_classification_api.services.predictor as predictor_module
import emotion_classification_api.services.csv_processor as csv_processor
import emotion_classification_api.services.transcriber as transcriber

# ---------------------- Logging Setup -----------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# --------------------- FastAPI App -------------------------

app = FastAPI()
logger.info("Local Emotion Classification API initialized.")


# --------------------- Request Models ---------------------
class TextInput(BaseModel):
    text: str


# --------------------- Endpoints --------------------------

@app.get("/")
async def root():
    """Health-check root."""
    return {"message": "Emotion Classification API is running."}


@app.post("/predict")
async def predict_endpoint(input: TextInput):
    """
    Predict endpoint delegates to predictor.predict_text.
    """
    return predictor_module.predict_text(input.text)


@app.post("/upload-csv")
async def upload_csv_endpoint(file: UploadFile = File(...)):
    """
    CSV upload: uses process_text_csv to get back a CSV string,
    then streams it back with proper headers.
    """
    try:
        csv_str = csv_processor.process_text_csv(file.file)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error("Error processing CSV: %s", e)
        raise HTTPException(status_code=500, detail="Internal error")

    return StreamingResponse(
        iter([csv_str]),
        media_type="text/csv",
        headers={"Content-Disposition": "attachment; filename=predicted.csv"},
    )


@app.post("/youtube-transcript")
async def youtube_transcript_endpoint(url: str = Body(..., embed=True)):
    """
    Download & transcribe YouTube video, return raw text in JSON.
    """
    try:
        transcript = transcriber.transcribe_youtube(url)
    except subprocess.CalledProcessError as e:
        logger.error("yt-dlp error: %s", e)
        raise HTTPException(status_code=400, detail="Failed to download audio")
    except Exception as e:
        logger.error("Transcription error: %s", e)
        raise HTTPException(status_code=500, detail="Failed to transcribe video")
    return {"transcript": transcript}


@app.post("/upload-audio-transcript")
async def audio_transcript_endpoint(file: UploadFile = File(...)):
    """
    Transcribe uploaded audio file, return raw text.
    """
    blob = await file.read()
    try:
        transcript = transcriber.transcribe_audio_bytes(blob)
    except Exception as e:
        logger.error("Audio transcription error: %s", e)
        raise HTTPException(status_code=500, detail="Failed to transcribe audio")
    return {"transcript": transcript}
