from fastapi import FastAPI, Request
from utils.logger_feedback import log_prediction, log_feedback, FeedbackInput
import FastAPI as emotion_api  # import your existing API script
from fastapi.routing import APIRouter
import time

app = emotion_api.app
router = APIRouter()

@router.post("/predict")
async def predict_with_logging(input_text: emotion_api.TextInput, request: Request):
    start = time.time()
    original_response = await emotion_api.predict_emotion(input_text)
    latency, input_hash = log_prediction(
        input_text.text,
        original_response["predicted_label"],
        original_response["emotions"][original_response["predicted_class"]]["score"],
        start,
        request
    )
    original_response["latency"] = latency
    original_response["input_hash"] = input_hash
    return original_response

@router.post("/feedback")
async def receive_feedback(feedback: FeedbackInput):
    return log_feedback(feedback)

app.include_router(router)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("run:app", host="0.0.0.0", port=8000, reload=True)
