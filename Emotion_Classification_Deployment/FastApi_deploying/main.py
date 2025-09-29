import logging
import torch
from fastapi import FastAPI, UploadFile, File, Form
from fastapi.responses import FileResponse
from pydantic import BaseModel
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from datetime import datetime
import pandas as pd
import whisper
import tempfile
import subprocess
import io
import os
import csv
import nltk
from azureml.core import Workspace, Datastore, Dataset
from azureml.core.authentication import ServicePrincipalAuthentication

# ------------------------ Setup Logging ------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# ------------------ Feedback CSV upload function ------------------
def upload_feedback_csv_to_azureml(feedback_type="good", local_file="csv_predictions.csv"):
    auth = ServicePrincipalAuthentication(
        tenant_id="0a33589b-0036-4fe8-a829-3ed0926af886",
        service_principal_id="a2230f31-0fda-428d-8c5c-ec79e91a49f5",
        service_principal_password=""
    )
    ws = Workspace(
        subscription_id="0a94de80-6d3b-49f2-b3e9-ec5818862801",
        resource_group="buas-y2",
        workspace_name="NLP8-2025",
        auth=auth
    )
    datastore = Datastore.get(ws, "workspaceblobstore")
    target_path = f"feedback/{feedback_type}/"
    remote_path = os.path.join(target_path, os.path.basename(local_file))
    datastore.upload_files(
        files=[local_file],
        target_path=target_path,
        overwrite=True,
        show_progress=True
    )
    # Register as dataset asset
    Dataset.File.from_files((datastore, remote_path)).register(
        workspace=ws,
        name=f"feedback_{feedback_type}_csv",
        description=f"{feedback_type.capitalize()} feedback CSV from user submission.",
        create_new_version=True
    )

# ------------------ Load model and tokenizer ------------------
def load_model(model_path="Abida4ka/fastapi-deploying-model", hf_token=os.getenv("HF_TOKEN")):
    logger.info("Loading model and tokenizer from Hugging Face repo: %s", model_path)
    tokenizer = AutoTokenizer.from_pretrained(model_path, token=hf_token)
    model = AutoModelForSequenceClassification.from_pretrained(model_path, token=hf_token)
    logger.info("Model and tokenizer loaded successfully.")
    return model, tokenizer

model, tokenizer = load_model()

# ----------------------- Input Schema -------------------------
class TextInput(BaseModel):
    text: str

# ---------------------- Initialize App ------------------------
app = FastAPI()
logger.info("Emotion Classification API initialized.")

# ------------------------ Endpoints ---------------------------

@app.get("/")
async def root():
    logger.info("Root endpoint accessed.")
    return {"message": "Emotion Classification API is running"}

@app.get("/health")
async def health():
    return {"status": "healthy"}

@app.post("/predict")
async def predict_emotion(input_text: TextInput):
    logger.info("Prediction request received for text: %s", input_text.text)
    inputs = tokenizer(
        input_text.text,
        return_tensors="pt",
        truncation=True,
        padding=True
    )
    with torch.no_grad():
        outputs = model(**inputs)
    probabilities = torch.softmax(outputs.logits, dim=1)[0]
    labels = [
        "anger", "joy", "optimism", "sadness",
        "surprise", "fear", "disgust"
    ]
    predicted_class = torch.argmax(probabilities).item()
    predicted_label = labels[predicted_class]
    logger.info(
        "Predicted label: %s with class index %d",
        predicted_label,
        predicted_class
    )
    predictions = [
        {"label": label, "score": float(prob)}
        for label, prob in zip(labels, probabilities)
    ]
    return {
        "text": input_text.text,
        "predicted_class": predicted_class,
        "predicted_label": predicted_label,
        "emotions": predictions
    }

@app.post("/upload-csv")
async def upload_csv(file: UploadFile = File(...)):
    logger.info("CSV upload request received: %s", file.filename)
    try:
        df = pd.read_csv(file.file)
    except Exception as e:
        logger.error("Error reading CSV: %s", str(e))
        return {"error": "Failed to read CSV file."}
    labels = [
        "anger", "joy", "optimism", "sadness",
        "surprise", "fear", "disgust"
    ]
    if "text" not in df.columns:
        logger.warning("CSV missing 'text' column.")
        return {"error": "CSV must contain a 'text' column."}
    df = df[["text"]]
    output = io.StringIO()
    writer = csv.writer(output)
    writer.writerow(["text", "predicted_label"])
    for sentence in df["text"]:
        sentence = sentence.strip()
        if not sentence:
            continue
        logger.info("Processing text: %s", sentence)
        try:
            inputs = tokenizer(
                sentence,
                return_tensors="pt",
                truncation=True,
                padding=True
            )
            with torch.no_grad():
                outputs = model(**inputs)
            probs = torch.softmax(outputs.logits, dim=1)[0]
            predicted_class = torch.argmax(probs).item()
            predicted_label = labels[predicted_class]
            writer.writerow([sentence, predicted_label])
            logger.info("Predicted label: %s", predicted_label)
        except Exception as e:
            logger.error("Error processing sentence: %s | %s", sentence, str(e))
    output_dir = "/app/output"
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, "csv_predictions.csv")
    with open(output_path, "w", newline="", encoding="utf-8") as f:
        f.write(output.getvalue())
    logger.info("CSV saved to: %s", output_path)
    return {
        "message": "CSV saved successfully.",
        "file_path": output_path
    }

# ------------------- Feedback CSV Endpoint ----------------------
@app.post("/feedback-csv")
async def feedback_csv(
    feedback: str = Form(...),   # "good" or "bad"
    csv_file: UploadFile = File(...)
):
    logger.info("Received feedback-csv: %s", feedback)
    local_path = f"/tmp/{csv_file.filename}"
    with open(local_path, "wb") as f:
        f.write(await csv_file.read())
    try:
        upload_feedback_csv_to_azureml(feedback_type=feedback, local_file=local_path)
    except Exception as e:
        logger.error(f"Failed to upload feedback CSV: {e}")
        return {"status": "error", "detail": str(e)}
    finally:
        if os.path.exists(local_path):
            os.remove(local_path)
    return {"status": "uploaded", "feedback": feedback}

# ------------------- Download Endpoints ----------------------
@app.get("/download-csv")
async def download_csv():
    path = "/app/output/csv_predictions.csv"
    if os.path.exists(path):
        return FileResponse(path, filename="csv_predictions.csv", media_type="text/csv")
    return {"error": "CSV file not found."}

# ------------------ Transcript Endpoints ---------------------
nltk.download('punkt')

@app.post("/youtube-transcript")
async def generate_transcript_csv(url: str):
    logger.info("Received YouTube URL for transcript: %s", url)
    try:
        with tempfile.TemporaryDirectory() as tmpdir:
            audio_path = f"{tmpdir}/audio.mp3"
            subprocess.run([
                "yt-dlp", "-x", "--audio-format", "mp3",
                "-o", audio_path, url
            ], check=True)
            model_whisper = whisper.load_model("base")
            result = model_whisper.transcribe(audio_path)
            transcript = result["text"]
    except Exception as e:
        logger.error("Transcription error: %s", e)
        return {"error": "Failed to transcribe the video."}
    sentences = nltk.sent_tokenize(transcript)
    csv_buffer = io.StringIO()
    writer = csv.writer(csv_buffer)
    writer.writerow(["text"])
    for sentence in sentences:
        writer.writerow([sentence.strip()])
    output_dir = "/app/output"
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, "youtube_transcript.csv")
    with open(output_path, "w", newline="", encoding="utf-8") as f:
        f.write(csv_buffer.getvalue())
    return {
        "message": "Transcript CSV saved successfully.",
        "file_path": output_path
    }

@app.post("/upload-audio-transcript")
async def generate_transcript_from_audio(file: UploadFile = File(...)):
    try:
        with tempfile.TemporaryDirectory() as tmpdir:
            input_path = os.path.join(tmpdir, file.filename)
            with open(input_path, "wb") as f:
                f.write(await file.read())
            model_whisper = whisper.load_model("base")
            result = model_whisper.transcribe(input_path)
            transcript = result["text"]
    except Exception as e:
        return {"error": str(e)}
    sentences = nltk.sent_tokenize(transcript)
    csv_buffer = io.StringIO()
    writer = csv.writer(csv_buffer)
    writer.writerow(["text"])
    for sentence in sentences:
        writer.writerow([sentence.strip()])
    output_dir = "/app/output"
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, "uploaded_audio_transcript.csv")
    with open(output_path, "w", newline="", encoding="utf-8") as f:
        f.write(csv_buffer.getvalue())
    return {
        "message": "Transcript CSV saved successfully.",
        "file_path": output_path
    }
