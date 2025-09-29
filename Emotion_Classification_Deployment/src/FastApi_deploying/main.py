"""Entry-point that only wires routers together (no heavy logic here)."""
from fastapi import FastAPI

from .routers import health, predict, transcript

app = FastAPI(title="Emotion Classification API")
app.include_router(health.router)
app.include_router(predict.router)
app.include_router(transcript.router)
