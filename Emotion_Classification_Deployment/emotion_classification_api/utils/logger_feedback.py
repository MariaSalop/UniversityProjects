"""Logging and feedback utilities for emotion classification API."""

import logging
import hashlib
import time
from pydantic import BaseModel
from fastapi import Request

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class FeedbackInput(BaseModel):
    """Schema for feedback submission."""
    input_hash: str
    prediction: str
    actual_label: str | None = None
    rating: int | None = None


def log_prediction(input_text: str, predicted_label: str,
                   confidence: float, latency: float,
                   request: Request) -> None:
    """
    Logs inference-related metrics.
    """
    input_hash = hashlib.md5(input_text.encode()).hexdigest()
    logger.info({
        "timestamp": time.time(),
        "latency": latency,
        "input_hash": input_hash,
        "prediction": predicted_label,
        "confidence": confidence,
        "client": request.client.host,
    })


def log_feedback(feedback: FeedbackInput) -> None:
    """
    Logs feedback from end user.
    """
    logger.info({
        "feedback": feedback.dict(),
        "timestamp": time.time(),
    })
