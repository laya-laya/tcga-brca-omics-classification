"""Fit the final model with fixed hyperparameters and evaluate it once.

Hyperparameters are chosen beforehand by nested cross-validation
(``scripts/run_nested_cv.py``); this step only refits them on the training
split and measures performance on the untouched held-out split.
"""
from __future__ import annotations

from sklearn.base import clone
from sklearn.metrics import balanced_accuracy_score, f1_score, roc_auc_score
from sklearn.model_selection import train_test_split

from .constants import RANDOM_STATE

TEST_SIZE = 0.20


def split_train_test(X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE):
    """The same stratified 80/20 split used throughout the analysis scripts."""
    return train_test_split(
        X, y, test_size=test_size, stratify=y, random_state=random_state
    )


def evaluate(model, X, y) -> dict:
    pred = model.predict(X)
    proba = model.predict_proba(X)
    return {
        "balanced_accuracy": balanced_accuracy_score(y, pred),
        "f1_macro": f1_score(y, pred, average="macro"),
        "f1_weighted": f1_score(y, pred, average="weighted"),
        "roc_auc_macro_ovr": roc_auc_score(
            y, proba, multi_class="ovr", average="macro", labels=model.classes_
        ),
    }


def fit_final_model(pipeline, params, X, y, *, test_size=TEST_SIZE,
                    random_state=RANDOM_STATE):
    """Refit ``pipeline`` with ``params`` on the training split.

    Returns (fitted model, held-out metrics, (X_train, X_test, y_train, y_test)).
    """
    X_train, X_test, y_train, y_test = split_train_test(
        X, y, test_size=test_size, random_state=random_state
    )
    model = clone(pipeline).set_params(**params)
    model.fit(X_train, y_train)
    metrics = evaluate(model, X_test, y_test)
    return model, metrics, (X_train, X_test, y_train, y_test)
