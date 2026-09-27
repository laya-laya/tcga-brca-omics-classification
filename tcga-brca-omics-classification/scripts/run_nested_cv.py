from pathlib import Path
import sys
import json
import pandas as pd
from sklearn.model_selection import train_test_split

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/"src"))

from tcga_brca.preprocess import build_dual_cohorts
from tcga_brca.models import search_spaces
from tcga_brca.nested_cv import nested_cv_model, summarize_nested_cv

RESULTS = ROOT/"results"
RESULTS.mkdir(exist_ok=True)

X_ml, _, y, _ = build_dual_cohorts(
    ROOT/"data/raw/HiSeqV2.gz",
    ROOT/"data/raw/BRCA_clinicalMatrix",
)

X_train, _, y_train, _ = train_test_split(
    X_ml,
    y,
    test_size=0.20,
    stratify=y,
    random_state=42,
)

all_results = []
best_params = {}

for name, (pipeline, grid) in search_spaces().items():
    folds, params = nested_cv_model(
        name,
        pipeline,
        grid,
        X_train,
        y_train,
        classes=sorted(y_train.unique()),
        outer_splits=4,
        inner_splits=3,
        random_state=42,
    )
    all_results.append(folds)
    best_params[name] = params

fold_results = pd.concat(all_results, ignore_index=True)
summary = summarize_nested_cv(fold_results)

fold_results.to_csv(RESULTS/"nested_cv_folds.csv", index=False)
summary.to_csv(RESULTS/"nested_cv_summary.csv", index=False)
(RESULTS/"nested_cv_best_params.json").write_text(
    json.dumps(best_params, indent=2)
)

print("\nNested cross-validation summary\n")
print(summary.round(4).to_string(index=False))
