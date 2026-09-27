import numpy as np
import pandas as pd
from sklearn.metrics import balanced_accuracy_score, f1_score, roc_auc_score
from sklearn.preprocessing import label_binarize

def _auc_score(y_true, proba, classes):
    classes = list(classes)
    if len(classes) == 2:
        positive = classes[1]
        y_binary = (np.asarray(y_true) == positive).astype(int)
        return roc_auc_score(y_binary, np.asarray(proba)[:, 1])

    y_bin = label_binarize(y_true, classes=classes)
    return roc_auc_score(
        y_bin,
        proba,
        average="macro",
        multi_class="ovr",
    )

def bootstrap_classification_metrics(
    y_true, pred, proba, classes, n_boot=1000, random_state=42
):
    rng = np.random.default_rng(random_state)
    y_true = np.asarray(y_true)
    pred = np.asarray(pred)
    proba = np.asarray(proba)
    classes = list(classes)

    rows = []
    n = len(y_true)
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        yt = y_true[idx]
        yp = pred[idx]
        pp = proba[idx]

        # Require every expected class in a resample so all metrics are comparable.
        if set(yt) != set(classes):
            continue

        rows.append({
            "balanced_accuracy": balanced_accuracy_score(yt, yp),
            "f1_macro": f1_score(yt, yp, average="macro"),
            "f1_weighted": f1_score(yt, yp, average="weighted"),
            "roc_auc_macro_ovr": _auc_score(yt, pp, classes),
        })

    df = pd.DataFrame(rows)
    if df.empty:
        raise RuntimeError(
            "No valid bootstrap resamples contained all classes. "
            "Increase sample size or reduce class count."
        )

    summary = []
    for col in df.columns:
        summary.append({
            "metric": col,
            "estimate_mean": df[col].mean(),
            "ci_2.5": df[col].quantile(0.025),
            "ci_97.5": df[col].quantile(0.975),
        })
    return df, pd.DataFrame(summary)
