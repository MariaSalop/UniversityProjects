from fastapi import APIRouter
from ..schemas import TextInput, PredictionOutput
from ..services import model_service

router = APIRouter()


@router.post("/predict", response_model=PredictionOutput)
async def predict_endpoint(payload: TextInput):
    label, probs = model_service.predict(payload.text)
    return {
        "text": payload.text,
        "predicted_label": label,
        "probabilities": probs,
    }
