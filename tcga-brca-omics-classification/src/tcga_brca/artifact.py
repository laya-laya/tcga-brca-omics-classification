"""Save and load the trained classifier as a self-describing model bundle.

A bundle holds everything needed to serve the model safely, not just the
fitted pipeline:

* ``pipeline``          fitted scikit-learn pipeline (select -> scale -> model)
* ``feature_names``     exact input schema (gene order) seen during training
* ``selected_features`` genes the model actually uses (kept by SelectKBest)
* ``train_medians``     per-gene training medians, used to impute gaps
* ``drift_reference``   training distribution of the selected genes
* ``metadata``          versions, training time, parameters, held-out metrics

Security note: joblib files are pickles, and loading one can execute code.
Only load bundles you trained yourself or otherwise trust.
"""
from __future__ import annotations

import datetime as _dt
import platform
import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import sklearn

from . import __version__
from .drift import reference_profile

BUNDLE_FORMAT = 1
DEFAULT_MODEL_PATH = Path("models") / "pam50_classifier.joblib"


def selected_features(pipeline, feature_names) -> list:
    """Genes kept by the pipeline's feature-selection step."""
    mask = pipeline.named_steps["select"].get_support()
    return [str(g) for g in np.asarray(feature_names)[mask]]


def build_bundle(pipeline, X_train: pd.DataFrame, *, metrics=None, params=None,
                 model_name="elastic_net") -> dict:
    """Package a fitted pipeline with its schema, reference data and metadata."""
    feature_names = [str(c) for c in X_train.columns]
    selected = selected_features(pipeline, feature_names)
    return {
        "format": BUNDLE_FORMAT,
        "pipeline": pipeline,
        "feature_names": feature_names,
        "selected_features": selected,
        "classes": [str(c) for c in pipeline.classes_],
        "train_medians": X_train.median().astype("float32"),
        "drift_reference": reference_profile(X_train[selected]),
        "metadata": {
            "model_name": model_name,
            "package_version": __version__,
            "sklearn_version": sklearn.__version__,
            "python_version": platform.python_version(),
            "trained_at": _dt.datetime.now(_dt.timezone.utc).isoformat(timespec="seconds"),
            "n_train_samples": int(len(X_train)),
            "params": dict(params or {}),
            "heldout_metrics": {k: round(float(v), 4) for k, v in (metrics or {}).items()},
        },
    }


def save_bundle(bundle: dict, path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(bundle, path, compress=3)
    return path


def load_bundle(path) -> dict:
    """Load a bundle and warn if it was trained with another scikit-learn."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"No model at {path}. Train one with `python scripts/train_model.py`."
        )
    bundle = joblib.load(path)
    if not isinstance(bundle, dict) or bundle.get("format") != BUNDLE_FORMAT:
        raise ValueError(f"{path} is not a model bundle (format {BUNDLE_FORMAT}).")
    trained_with = bundle["metadata"].get("sklearn_version")
    if trained_with != sklearn.__version__:
        warnings.warn(
            f"Model trained with scikit-learn {trained_with}, running "
            f"{sklearn.__version__}; retrain in this environment to be safe.",
            stacklevel=2,
        )
    return bundle


def model_card(bundle: dict) -> dict:
    """JSON-safe summary of a bundle, for the API and for version control."""
    return {
        **bundle["metadata"],
        "classes": bundle["classes"],
        "n_input_genes": len(bundle["feature_names"]),
        "n_model_genes": len(bundle["selected_features"]),
        "drift_monitoring": {
            "n_reference_samples": bundle["drift_reference"]["n_reference_samples"],
            "n_bins": bundle["drift_reference"]["n_bins"],
        },
    }
