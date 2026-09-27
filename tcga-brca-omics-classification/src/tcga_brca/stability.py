import pandas as pd
from sklearn.base import clone
from sklearn.model_selection import StratifiedKFold

def feature_selection_stability(estimator, X, y, n_splits=5, random_state=42):
    """Count how often each gene is selected across CV folds."""
    cv = StratifiedKFold(
        n_splits=n_splits, shuffle=True, random_state=random_state
    )
    counts = pd.Series(0, index=X.columns, dtype=int)

    for tr, _ in cv.split(X, y):
        est = clone(estimator)
        est.fit(X.iloc[tr], y.iloc[tr])
        selector = est.named_steps["select"]
        selected = X.columns[selector.get_support()]
        counts.loc[selected] += 1

    out = counts[counts > 0].sort_values(ascending=False).to_frame("times_selected")
    out["selection_frequency"] = out["times_selected"] / n_splits
    out.index.name = "gene"
    return out.reset_index()
