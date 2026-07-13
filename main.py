"""FastAPI app exposing the medical transcriber over HTTP.

Endpoints:
  POST /transcribe  - upload an audio file, get back raw + corrected text
  POST /feedback     - teach the tool a correction it should remember
  GET  /health       - liveness/readiness check
"""

from __future__ import annotations

import os
import tempfile
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, File, HTTPException, UploadFile
from pydantic import BaseModel

from transcriber import MedicalTranscriber

MODEL_SIZE = os.environ.get("WHISPER_MODEL_SIZE", "base")
DEVICE = os.environ.get("WHISPER_DEVICE", "cpu")
COMPUTE_TYPE = os.environ.get("WHISPER_COMPUTE_TYPE", "int8")

_state: dict = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    _state["transcriber"] = MedicalTranscriber(
        model_size=MODEL_SIZE,
        device=DEVICE,
        compute_type=COMPUTE_TYPE,
    )
    yield
    _state.clear()


app = FastAPI(title="Medical Speech-to-Text", lifespan=lifespan)


class FeedbackRequest(BaseModel):
    raw_text: str
    corrected_text: str


@app.get("/health")
def health():
    return {
        "status": "ok",
        "model_size": MODEL_SIZE,
        "device": DEVICE,
        "compute_type": COMPUTE_TYPE,
    }


@app.post("/transcribe")
async def transcribe(file: UploadFile = File(...)):
    transcriber: MedicalTranscriber = _state["transcriber"]

    suffix = Path(file.filename or "audio").suffix or ".wav"
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        tmp.write(await file.read())
        tmp_path = tmp.name

    try:
        result = transcriber.transcribe(tmp_path)
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    finally:
        Path(tmp_path).unlink(missing_ok=True)

    return result


@app.post("/feedback")
def feedback(request: FeedbackRequest):
    transcriber: MedicalTranscriber = _state["transcriber"]
    result = transcriber.learn_from_correction(request.raw_text, request.corrected_text)
    return result
