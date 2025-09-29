import logging
import torch
from fastapi import FastAPI, UploadFile, File
from pydantic import BaseModel
from transformers import AutoModelForSequenceClassification, AutoTokenizer
import pandas as pd
import whisper
import tempfile
import subprocess
import io
import os
import csv
import nltk


# ------------------------ Setup Logging ------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


# ------------------ Load model and tokenizer ------------------

def load_model(model_path="./model"):
    logger.info("Loading model and tokenizer from %s", model_path)
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForSequenceClassification.from_pretrained(model_path)
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

    # Define emotion labels (customize as per your model)
    labels = [
        "anger",
        "joy",
        "optimism",
        "sadness",
        "surprise",
        "fear",
        "disgust"
    ]

    # Get the predicted label
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

    # Save to file inside mounted volume
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

# Ensure sentence tokenizer is available
nltk.download('punkt')


@app.post("/youtube-transcript")
async def generate_transcript_csv(url: str):
    logger.info("Received YouTube URL for transcript: %s", url)

    try:
        with tempfile.TemporaryDirectory() as tmpdir:
            audio_path = f"{tmpdir}/audio.mp3"
            logger.info("Downloading audio with yt-dlp...")

            subprocess.run([
                "yt-dlp", "-x", "--audio-format", "mp3",
                "-o", audio_path, url
            ], check=True)

            logger.info("Audio downloaded to: %s", audio_path)

            logger.info("Loading Whisper model...")
            model_whisper = whisper.load_model("base")

            logger.info("Transcribing audio...")
            result = model_whisper.transcribe(audio_path)
            transcript = result["text"]
            logger.info("Transcription complete. Length: %d characters", len(transcript))

    except subprocess.CalledProcessError as e:
        logger.error("yt-dlp download error: %s", e)
        return {"error": "Failed to download audio from YouTube URL."}
    except Exception as e:
        logger.error("Transcription error: %s", e)
        return {"error": "Failed to transcribe the video."}

    try:
        logger.info("Splitting transcript into sentences...")
        sentences = nltk.sent_tokenize(transcript)
        logger.info("Total sentences: %d", len(sentences))

        csv_buffer = io.StringIO()
        writer = csv.writer(csv_buffer)
        writer.writerow(["text"])

        for sentence in sentences:
            sentence = sentence.strip()
            if sentence:
                writer.writerow([sentence])

        output_dir = "/app/output"
        os.makedirs(output_dir, exist_ok=True)
        output_path = os.path.join(output_dir, "youtube_transcript.csv")

        with open(output_path, "w", newline="", encoding="utf-8") as f:
            f.write(csv_buffer.getvalue())

        logger.info("Transcript CSV saved to: %s", output_path)

        return {
            "message": "Transcript CSV saved successfully.",
            "file_path": output_path
        }

    except Exception:
        logger.exception("Error saving transcript CSV")  # More detailed
        raise


@app.post("/upload-audio-transcript")
async def generate_transcript_from_audio(file: UploadFile = File(...)):
    logger.info("Received audio file for transcription: %s", file.filename)

    try:
        with tempfile.TemporaryDirectory() as tmpdir:
            # Save the uploaded audio file
            input_path = os.path.join(tmpdir, file.filename)
            with open(input_path, "wb") as f:
                content = await file.read()
                f.write(content)

            logger.info("Audio file saved to: %s", input_path)

            # Load Whisper model
            logger.info("Loading Whisper model...")
            model_whisper = whisper.load_model("base")

            # Transcribe
            logger.info("Transcribing audio...")
            result = model_whisper.transcribe(input_path)
            transcript = result["text"]
            logger.info("Transcription complete. Length: %d characters", len(transcript))

    except Exception as e:
        logger.error("Audio transcription error: %s", e)
        return {"error": "Failed to transcribe the uploaded audio."}

    try:
        logger.info("Splitting transcript into sentences...")
        sentences = nltk.sent_tokenize(transcript)
        logger.info("Total sentences: %d", len(sentences))

        # Write to CSV in memory
        csv_buffer = io.StringIO()
        writer = csv.writer(csv_buffer)
        writer.writerow(["text"])
        for sentence in sentences:
            sentence = sentence.strip()
            if sentence:
                writer.writerow([sentence])

        # Save to file in /app/output
        output_dir = "/app/output"
        os.makedirs(output_dir, exist_ok=True)
        output_path = os.path.join(output_dir, "uploaded_audio_transcript.csv")

        with open(output_path, "w", newline="", encoding="utf-8") as f:
            f.write(csv_buffer.getvalue())

        logger.info("Transcript CSV saved to: %s", output_path)

        return {
            "message": "Transcript CSV saved successfully.",
            "file_path": output_path
        }

    except Exception:
        logger.exception("Error saving transcript CSV")
        return {"error": "Failed to generate transcript CSV from audio."}
