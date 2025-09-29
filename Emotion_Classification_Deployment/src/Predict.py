import logging

import numpy as np
import tensorflow as tf
from transformers import AutoTokenizer, TFAutoModelForSequenceClassification
from utils.logging_utils import setup_logging

# Setup logging
setup_logging(
    level=logging.DEBUG,
    log_to_file=True,
    filename="logs/predict.log"
)


def load_model_and_tokenizer(model_path):
    logging.debug(
        "Loading Hugging Face model and tokenizer from %s", model_path
    )
    model = TFAutoModelForSequenceClassification.from_pretrained(model_path)
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    return model, tokenizer


def predict_emotion(model, tokenizer, texts, max_length=50):
    # Tokenize input texts
    inputs = tokenizer(
        texts,
        return_tensors="tf",
        padding=True,
        truncation=True,
        max_length=max_length
    )
    logging.debug(f"Tokenized input: {inputs}")

    # Run model prediction
    outputs = model(inputs)
    logits = outputs.logits
    predictions = tf.nn.softmax(logits, axis=-1)

    predicted_class = np.argmax(predictions, axis=-1)
    logging.debug(f"Raw logits: {logits}")
    logging.debug(f"Predictions (probabilities): {predictions}")
    logging.debug(f"Predicted emotion class indices: {predicted_class}")
    return predicted_class


if __name__ == "__main__":
    model_path = (
        r"C:\Users\User\Documents\GitHub\2024-25d-fai2-adsai-group-nlp8"
        r"\emotion_7"
    )
    model, tokenizer = load_model_and_tokenizer(model_path)

    # Dummy example input text(s) for prediction
    dummy_texts = [
        "I am so happy today!",
        "This is so frustrating...",
        "I feel neutral about this.",
    ]

    predicted_classes = predict_emotion(model, tokenizer, dummy_texts)
    for text, pred_class in zip(dummy_texts, predicted_classes):
        logging.info(
            'Text: "%s" -> Predicted emotion class: %s',
            text,
            pred_class,
        )
