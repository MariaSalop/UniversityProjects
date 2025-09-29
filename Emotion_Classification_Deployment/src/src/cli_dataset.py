import typer
import torch
import pandas as pd
from pathlib import Path
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer
)

app = typer.Typer()

if not hasattr(app, "main"):          # Click-runner safety-hook
    app.main = app                    # .main() will simply call Typer

# Constants
PACKAGE_DIR = Path(__file__).resolve().parent          # src/
MODEL_PATH = PACKAGE_DIR.parent / "emotion_7"          # Project_NLP8/emotion_7

# MODEL_PATH = "./emotion_7"
LABELS = [
    "anger", "joy", "sadness", "surprise",
    "fear", "disgust", "neutral"
]

# lazy model loader
_model = None
_tokenizer = None


def _ensure_loaded():
    global _model, _tokenizer
    if _model is None or _tokenizer is None:
        _model = AutoModelForSequenceClassification.from_pretrained(MODEL_PATH)
        _tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)


def predict_emotion(text: str):
    """Predict emotion from text using the locally loaded model."""
    _ensure_loaded()          # lazy model load
    inputs = _tokenizer(text, return_tensors="pt", truncation=True, padding=True)
    with torch.no_grad():
        outputs = _model(**inputs)
    probs = torch.softmax(outputs.logits, dim=1)[0]
    cls = torch.argmax(probs).item()
    predictions = [
        {"label": label, "score": float(prob)}
        for label, prob in zip(LABELS, probs)
    ]
    return (
        text,
        cls,
        LABELS[cls],
        predictions,
    )


@app.command()
def test(text: str):
    """Echo the input text."""
    print(text)


@app.command()
def predict(text: str):
    """Predict the emotion from a given text input."""
    try:
        text_out, pred_class, pred_label, scores = predict_emotion(text)
        typer.echo(f"Text: {text_out}")
        typer.echo(f"Predicted Emotion: {pred_label} (Class: {pred_class})")
        typer.echo("Emotion Scores:")
        for item in scores:
            typer.echo(
                f"  - {item['label']}: {item['score']:.4f}"
            )
    except Exception as exc:
        typer.echo(f"Error: {str(exc)}")


@app.command()
def batch_predict(
    file_path: str = typer.Argument(...),
    text_column: str = typer.Option(
        "text", "--text-column",
        help="Name of the column containing text."
    )
):
    """Run predictions for all rows in a CSV file on a specified column."""
    try:
        path = Path(file_path)
        if not path.exists():
            typer.echo(f"File not found: {file_path}")
            raise typer.Exit(code=1)

        df = pd.read_csv(path)
        if text_column not in df.columns:
            typer.echo(
                f"Column '{text_column}' not found in file."
            )
            raise typer.Exit(code=1)

        results = []
        for text in df[text_column].astype(str):
            _, _, predicted_label, _ = predict_emotion(text)
            results.append(predicted_label)

        df["predicted_emotion"] = results
        output_file = path.with_name(f"{path.stem}_predicted.csv")
        df.to_csv(output_file, index=False)
        typer.echo(f"Predictions saved to {output_file}")
    except Exception as exc:
        typer.echo(f"Error: {str(exc)}")


if __name__ == "__main__":
    app()
