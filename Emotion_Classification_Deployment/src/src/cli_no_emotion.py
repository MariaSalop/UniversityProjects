import logging
from pathlib import Path

import pandas as pd
import torch
import typer
from tqdm import tqdm
from transformers import AutoModelForSequenceClassification, AutoTokenizer

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler("app.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

app = typer.Typer()

if not hasattr(app, "main"):          # Click-runner safety-hook
    app.main = app                    # .main() will simply call Typer

# Constants
MODEL_DIR = Path(__file__).resolve().parent.parent / "emotion_7"
LABELS = [
    "anger", "joy", "optimism", "sadness",
    "surprise", "fear", "disgust"
]


# device request in the function
def choose_device() -> str:
    valid = {"cpu", "gpu"}
    while True:
        choice = typer.prompt("Which device to use? (cpu or gpu)").strip().lower()
        if choice in valid:
            break
        typer.echo("Invalid input. Please type 'cpu' or 'gpu'.")
    if choice == "gpu" and not torch.cuda.is_available():
        logging.warning("GPU selected but not available. Falling back to CPU.")
        choice = "cpu"
    return "cuda" if choice == "gpu" else "cpu"


def init_model(device: str):
    logger.info("Loading model and tokenizer from %s", MODEL_DIR)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_DIR).to(device)
    tok = AutoTokenizer.from_pretrained(MODEL_DIR)
    return model, tok


# these variables will be filled after init_model()
model = None
tokenizer = None
device = "cpu"          # default value, so that linters don't swear


def drop_column_if_exists(df: pd.DataFrame, column_name: str) -> pd.DataFrame:
    """
    Drop a specified column from the DataFrame if it exists.

    Args:
        df (pd.DataFrame): Input dataframe.
        column_name (str): Column name to drop.

    Returns:
        pd.DataFrame: DataFrame with the column dropped if it existed.
    """
    if column_name in df.columns:
        logger.info("Dropping column '%s' from DataFrame", column_name)
        df = df.drop(columns=[column_name])
    return df


def predict_emotion(text: str):
    """
    Predict the emotion of a single text input.

    Args:
        text (str): Input text.

    Returns:
        Tuple containing original text, predicted class index, label,
        and scores for all classes.
    """
    logger.debug("Predicting emotion for text: %s", text)
    inputs = tokenizer(
        text, return_tensors="pt", truncation=True, padding=True
    ).to(device)
    with torch.no_grad():
        outputs = model(**inputs)
    probs = torch.softmax(outputs.logits, dim=1)[0]
    predicted_class = torch.argmax(probs).item()
    predicted_label = LABELS[predicted_class]
    predictions = [
        {"label": label, "score": float(prob)}
        for label, prob in zip(LABELS, probs)
    ]
    logger.info("Predicted label: %s (class %d)", predicted_label, predicted_class)
    return text, predicted_class, predicted_label, predictions


@app.command()
def predict(text: str):
    """
    Command-line interface for predicting emotion from a single text string.

    Args:
        text (str): Input text.
    """
    logger.info("Predict command called with text: %s", text)
    try:
        text_out, pred_class, pred_label, scores = predict_emotion(text)
        typer.echo(f"Text: {text_out}")
        typer.echo(f"Predicted Emotion: {pred_label} (Class: {pred_class})")
        typer.echo("Emotion Scores:")
        for item in scores:
            typer.echo(f"  - {item['label']}: {item['score']:.4f}")
    except Exception as exc:
        logger.error("Error during prediction: %s", str(exc))
        typer.echo(f"Error: {str(exc)}")


@app.command()
def batch_predict(
    file_path: str = typer.Argument(...),
    text_column: str = typer.Option(
        "text", "--text-column",
        help="Name of the column containing text."
    ),
    confidence_threshold: float = typer.Option(
        0.5, "--confidence-threshold",
        help="Minimum confidence to trust prediction."
    )
):
    """
    Predict emotions for all texts in a CSV file.

    Args:
        file_path (str): Path to the CSV file.
        text_column (str): Name of the column with input texts.
        confidence_threshold (float): Threshold for low confidence flag.
    """
    logger.info("Batch predict command called with file: %s", file_path)
    try:
        path = Path(file_path)
        if not path.exists():
            typer.echo(f"File not found: {file_path}")
            raise typer.Exit(code=1)

        df = pd.read_csv(path)
        df = drop_column_if_exists(df, "emotion")

        if text_column not in df.columns:
            typer.echo(f"Column '{text_column}' not found in file.")
            raise typer.Exit(code=1)

        results = []
        confidence_flags = []

        for text in tqdm(df[text_column].astype(str), desc="Processing", unit="line"):
            _, _, predicted_label, scores = predict_emotion(text)
            results.append(predicted_label)
            max_conf = max(score["score"] for score in scores)
            confidence_flags.append(max_conf < confidence_threshold)

        df["predicted_emotion"] = results
        df["low_confidence_flag"] = confidence_flags
        output_file = path.with_name(f"{path.stem}_v2.csv")
        df.to_csv(output_file, index=False)
        typer.echo(f"Predictions saved to {output_file}")
    except Exception as exc:
        logger.error("Error during batch prediction: %s", str(exc))
        typer.echo(f"Error: {str(exc)}")


@app.command()
def model_info():
    """
    Display information about the loaded model and labels.
    """
    typer.echo("Model Info:")
    typer.echo(f"  - Model path: {MODEL_DIR}")
    typer.echo(f"  - Labels: {', '.join(LABELS)}")


# only on direct launch
if __name__ == "__main__":
    device = choose_device()
    model, tokenizer = init_model(device)
    app()          # launch Typer CLI
