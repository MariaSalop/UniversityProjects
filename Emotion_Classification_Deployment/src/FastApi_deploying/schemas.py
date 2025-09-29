from pydantic import BaseModel
from typing import List


class TextInput(BaseModel):
    text: str


class PredictionOutput(BaseModel):
    text: str
    predicted_label: str
    probabilities: List[float]
