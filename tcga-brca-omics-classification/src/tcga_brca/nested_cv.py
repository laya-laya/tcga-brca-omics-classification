import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.model_selection import StratifiedKFold, GridSearchCV
from sklearn.metrics import (
    balanced_accuracy_score,
    f1_score,
    roc_auc_score,
)
from sklearn.preprocessing import label_binarize

def multiclass_auc(y_true, proba, classes):
    classes = list(classes)
    if len(classes) == 2:
        positive = classes[1]
        y_binary = (np.asarray(y_true) == positive).astype(int)
        return roc_auc_score(y_binary, np.asarray(proba)[:, 1])

    y_bin = label_binarize(y_true, classes=classes)
    return roc_auc_score(
        y_bin, proba,
        average="macro",
        multi_class="ovr"
    )

def nested_cv_model(
    name, pipeline, param_grid, X, y, classes,
    outer_splits=4, inner_splits=3, random_state=42
):
    outer = StratifiedKFold(
        outer_splits, shuffle=True, random_state=random_state
    )
    rows = []
    best_params = []

    for fold, (tr, va) in enumerate(outer.split(X, y), start=1):
        inner = StratifiedKFold(
            inner_splits, shuffle=True, random_state=random_state + fold
        )
        search = GridSearchCV(
            clone(pipeline),
            param_grid=param_grid,
            scoring="balanced_accuracy",
            cv=inner,
            n_jobs=-1,
            refit=True,
        )
        search.fit(X.iloc[tr], y.iloc[tr])
        pred = search.predict(X.iloc[va])
        proba = search.predict_proba(X.iloc[va])

        rows.append({
            "model": name,
            "fold": fold,
            "balanced_accuracy": balanced_accuracy_score(y.iloc[va], pred),
            "f1_macro": f1_score(y.iloc[va], pred, average="macro"),
            "f1_weighted": f1_score(y.iloc[va], pred, average="weighted"),
            "roc_auc_macro_ovr": multiclass_auc(
                y.iloc[va], proba, list(search.best_estimator_.classes_)
            ),
        })
        best_params.append(search.best_params_)

    return pd.DataFrame(rows), best_params

def summarize_nested_cv(results):
    metrics = ["balanced_accuracy","f1_macro","f1_weighted","roc_auc_macro_ovr"]
    rows = []
    for model, g in results.groupby("model"):
        row = {"model": model}
        for m in metrics:
            row[f"{m}_mean"] = g[m].mean()
            row[f"{m}_std"] = g[m].std(ddof=1)
        rows.append(row)
    return pd.DataFrame(rows).sort_values(
        "balanced_accuracy_mean", ascending=False
    )
