"""Prediction logic shared by the batch job and the HTTP API.

Both entry points call :func:`predict`, so a sample gets the same answer
whichever way it is scored.
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd

from .drift import drift_report

MAX_MISSING_MODEL_GENES = 0.05   # reject input missing >5% of the genes the model uses
LOW_CONFIDENCE = 0.6             # predictions below this are counted in the summary


class SchemaError(ValueError):
    """The input cannot be scored safely (wrong genes, empty, malformed)."""


def to_numeric_frame(X: pd.DataFrame) -> pd.DataFrame:
    """Samples x genes with string labels and float values (non-numbers -> NaN)."""
    if X is None or X.shape[0] == 0:
        raise SchemaError("No samples in the input.")
    X = X.copy()
    X.index = X.index.astype(str)
    X.columns = X.columns.astype(str)
    if X.columns.duplicated().any():
        dupes = sorted(set(X.columns[X.columns.duplicated()]))[:5]
        raise SchemaError(f"Duplicate gene names in input, e.g. {dupes}.")
    if X.index.duplicated().any():
        raise SchemaError("Duplicate sample IDs in input.")
    return X.apply(pd.to_numeric, errors="coerce").astype(float)


def prepare_features(X: pd.DataFrame, bundle: dict,
                     max_missing=MAX_MISSING_MODEL_GENES):
    """Align incoming samples to the training schema.

    * genes the model never saw are dropped; columns are put in training order
    * too many missing model genes -> SchemaError (predictions would be guesses)
    * remaining gaps are filled with training medians

    Genes outside the selected set do not affect predictions (the pipeline's
    selection step discards them), so they are only imputed to satisfy the
    input shape.
    """
    features = bundle["feature_names"]
    selected = bundle["selected_features"]
    present = set(X.columns)

    missing_model_genes = sorted(set(selected) - present)
    share_missing = len(missing_model_genes) / len(selected)
    if share_missing > max_missing:
        raise SchemaError(
            f"{len(missing_model_genes)} of the {len(selected)} genes the model uses "
            f"are missing ({share_missing:.1%}; limit {max_missing:.0%}), "
            f"e.g. {missing_model_genes[:5]}. Is the input in the training format "
            "(UCSC Xena HiSeqV2 gene symbols)?"
        )

    aligned = X.reindex(columns=features)
    n_imputed = int(aligned[selected].isna().to_numpy().sum())
    aligned = aligned.fillna(bundle["train_medians"]).astype("float32")

    validation = {
        "n_samples": int(len(X)),
        "n_input_genes": int(X.shape[1]),
        "n_unknown_genes_ignored": int(len(present - set(features))),
        "n_model_genes": len(selected),
        "n_model_genes_missing": len(missing_model_genes),
        "model_genes_missing": missing_model_genes[:20],
        "n_values_imputed_in_model_genes": n_imputed,
    }
    return aligned, validation


def predict(X: pd.DataFrame, bundle: dict, *, check_drift: bool = True) -> dict:
    """Score samples (rows) and return predictions plus monitoring reports."""
    X = to_numeric_frame(X)
    aligned, validation = prepare_features(X, bundle)

    proba = bundle["pipeline"].predict_proba(aligned)
    classes = np.asarray(bundle["classes"])
    confidence = proba.max(axis=1)

    predictions = pd.DataFrame(
        proba, index=aligned.index, columns=[f"prob_{c}" for c in classes]
    )
    predictions.insert(0, "predicted_subtype", classes[proba.argmax(axis=1)])
    predictions.insert(1, "confidence", confidence)
    predictions.index.name = "sample_id"

    counts = predictions["predicted_subtype"].value_counts()
    summary = {
        "n_samples": int(len(predictions)),
        "predicted_subtype_counts": {c: int(counts.get(c, 0)) for c in bundle["classes"]},
        "mean_confidence": round(float(confidence.mean()), 4),
        "share_low_confidence": round(float((confidence < LOW_CONFIDENCE).mean()), 4),
        "low_confidence_threshold": LOW_CONFIDENCE,
    }
    drift = drift_report(X, bundle["drift_reference"]) if check_drift else None

    return {
        "predictions": predictions,
        "validation": validation,
        "drift": drift,
        "summary": summary,
        "model": {
            "model_name": bundle["metadata"].get("model_name"),
            "trained_at": bundle["metadata"].get("trained_at"),
            "package_version": bundle["metadata"].get("package_version"),
        },
    }


def report(result: dict) -> dict:
    """Everything except the per-sample table, JSON-serialisable."""
    return {k: v for k, v in result.items() if k != "predictions"}


def predictions_as_records(predictions: pd.DataFrame) -> list:
    """Per-sample predictions as plain JSON types (for the API)."""
    records = json.loads(predictions.reset_index().to_json(orient="records", double_precision=6))
    prob_cols = [c for c in predictions.columns if c.startswith("prob_")]
    return [
        {
            "sample_id": r["sample_id"],
            "predicted_subtype": r["predicted_subtype"],
            "confidence": r["confidence"],
            "probabilities": {c[len("prob_"):]: r[c] for c in prob_cols},
        }
        for r in records
    ]
