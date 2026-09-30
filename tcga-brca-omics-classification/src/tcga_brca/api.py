"""HTTP service for the PAM50 subtype classifier.

    MODEL_PATH=models/pam50_classifier.joblib uvicorn tcga_brca.api:app --app-dir src

Endpoints
  GET  /health   liveness/readiness: 200 when a model is loaded, else 503
  GET  /model    model card (versions, training date, held-out metrics)
  POST /predict  score samples; same code path as the batch job

Interactive docs are served at /docs.
"""
from __future__ import annotations

import logging
import os
from contextlib import asynccontextmanager
from typing import Dict, Optional

import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from . import __version__
from .artifact import DEFAULT_MODEL_PATH, load_bundle, model_card
from .inference import SchemaError, predict, predictions_as_records, report

log = logging.getLogger("tcga_brca.api")
_state: dict = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    path = os.environ.get("MODEL_PATH", str(DEFAULT_MODEL_PATH))
    try:
        _state["bundle"] = load_bundle(path)
        log.info("Loaded model from %s", path)
    except Exception as exc:  # keep running so /health can report the problem
        _state["bundle"] = None
        _state["error"] = f"{type(exc).__name__}: {exc}"
        log.error("Could not load model from %s: %s", path, _state["error"])
    yield
    _state.clear()


app = FastAPI(
    title="TCGA-BRCA PAM50 subtype classifier",
    version=__version__,
    lifespan=lifespan,
)


class PredictRequest(BaseModel):
    samples: Dict[str, Dict[str, Optional[float]]] = Field(
        ...,
        description="sample_id -> {gene symbol: log2 expression}, in the "
                    "same units as the UCSC Xena HiSeqV2 training data.",
    )


def _bundle() -> dict:
    bundle = _state.get("bundle")
    if bundle is None:
        raise HTTPException(503, detail=f"Model not loaded: {_state.get('error')}")
    return bundle


@app.get("/health")
def health():
    bundle = _bundle()
    return {"status": "ok", "model_trained_at": bundle["metadata"].get("trained_at")}


@app.get("/model")
def model_info():
    return model_card(_bundle())


@app.post("/predict")
def predict_samples(request: PredictRequest):
    bundle = _bundle()
    if not request.samples:
        raise HTTPException(422, detail="No samples given.")
    X = pd.DataFrame.from_dict(request.samples, orient="index")
    try:
        result = predict(X, bundle)
    except SchemaError as exc:
        raise HTTPException(422, detail=str(exc)) from exc
    return {
        "predictions": predictions_as_records(result["predictions"]),
        **report(result),
    }
