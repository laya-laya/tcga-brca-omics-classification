from sklearn.pipeline import Pipeline
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

def elastic_net_pipeline(k=500, random_state=42):
    return Pipeline([
        ("select", SelectKBest(score_func=f_classif, k=k)),
        ("scale", StandardScaler()),
        ("model", LogisticRegression(
            penalty="elasticnet",
            solver="saga",
            class_weight="balanced",
            max_iter=10000,
            random_state=random_state,
        )),
    ])

def random_forest_pipeline(k=500, random_state=42):
    return Pipeline([
        ("select", SelectKBest(score_func=f_classif, k=k)),
        ("model", RandomForestClassifier(
            n_estimators=500,
            class_weight="balanced",
            random_state=random_state,
            n_jobs=1,
        )),
    ])

def search_spaces():
    """Moderate grids suitable for a portfolio-scale nested CV analysis."""
    return {
        "elastic_net": (
            elastic_net_pipeline(),
            {
                "select__k": [100, 250, 500],
                "model__C": [0.1, 1.0, 10.0],
                "model__l1_ratio": [0.1, 0.5, 0.9],
            },
        ),
        "random_forest": (
            random_forest_pipeline(),
            {
                "select__k": [250, 500],
                "model__max_depth": [None, 20],
                "model__min_samples_leaf": [1, 3],
            },
        ),
    }
