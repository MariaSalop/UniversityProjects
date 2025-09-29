# train_and_evaluate.py
# Patch keras.utils.unpack_x_y_sample_weight for
# TF 2.10.x compatibility with transformers
import keras.utils

if not hasattr(keras.utils, "unpack_x_y_sample_weight"):
    def unpack_x_y_sample_weight(data):
        if len(data) == 2:
            return data[0], data[1], None
        elif len(data) == 3:
            return data
        else:
            raise ValueError(f"Unexpected input data length: {len(data)}")

    keras.utils.unpack_x_y_sample_weight = unpack_x_y_sample_weight

import logging

import tensorflow as tf
from transformers import (
    RobertaTokenizerFast, TFRobertaForSequenceClassification
)
from utils.logging_utils import setup_logging

setup_logging(level=logging.DEBUG, log_to_file=True, filename="logs/train.log")
logger = logging.getLogger(__name__)

# Path to local huggingface model directory
PROJECT_ROOT = r"C:\Users\User\Documents\GitHub\2024-25d-fai2-adsai-group-nlp8"
MODEL_DIR = rf"{PROJECT_ROOT}\emotion_7"


def load_data():
    texts = [
        "I am happy", "I feel sad", "I am excited", "I am angry", "I feel surprised"
    ]
    labels = [0, 1, 2, 3, 4]
    # Example emotion class labels (must match your dataset)
    return texts, labels


def encode_texts(tokenizer, texts, max_length=128):
    return tokenizer(
        texts,
        max_length=max_length,
        padding=True,
        truncation=True,
        return_tensors="tf"
    )


def create_tf_dataset(inputs, labels, batch_size=8, shuffle=True):
    dataset = tf.data.Dataset.from_tensor_slices((inputs, labels))
    if shuffle:
        dataset = dataset.shuffle(buffer_size=len(labels))
    dataset = dataset.batch(batch_size)
    return dataset


def train_and_evaluate():
    logger.info("Starting training with Hugging Face RoBERTa model...")

    tokenizer = RobertaTokenizerFast.from_pretrained(MODEL_DIR)
    model = TFRobertaForSequenceClassification.from_pretrained(MODEL_DIR)

    texts, labels = load_data()
    logger.debug(f"Loaded {len(texts)} texts and {len(labels)} labels")

    inputs = encode_texts(tokenizer, texts)
    logger.debug(f"Input IDs shape: {inputs['input_ids'].shape}")
    logger.debug(f"Attention mask shape: {inputs['attention_mask'].shape}")

    labels_tf = tf.constant(labels)

    dataset = tf.data.Dataset.from_tensor_slices(
        (
            {
                "input_ids": inputs["input_ids"],
                "attention_mask": inputs["attention_mask"],
            },
            labels_tf,
        )
    ).shuffle(buffer_size=len(labels))

    train_size = int(0.8 * len(labels))
    val_dataset = dataset.skip(train_size).batch(8)
    train_dataset = dataset.take(train_size).batch(8)

    optimizer = tf.keras.optimizers.Adam(learning_rate=5e-5)
    loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
    metrics = ["accuracy"]
    model.compile(optimizer=optimizer, loss=loss, metrics=metrics)

    logger.info("Starting model training...")
    model.fit(train_dataset, validation_data=val_dataset, epochs=3)

    logger.info("Training complete. Evaluating on validation set...")
    _, val_accuracy = model.evaluate(val_dataset)
    logger.info(f"Validation accuracy: {val_accuracy * 100:.2f}%")


if __name__ == "__main__":
    train_and_evaluate()
