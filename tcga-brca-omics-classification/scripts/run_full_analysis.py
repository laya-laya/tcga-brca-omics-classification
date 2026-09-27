from pathlib import Path
import sys
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.metrics import (
    classification_report, ConfusionMatrixDisplay,
    balanced_accuracy_score, f1_score
)

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from tcga_brca.preprocess import build_dual_cohorts
from tcga_brca.models import search_spaces
from tcga_brca.stability import feature_selection_stability
from tcga_brca.bootstrap import bootstrap_classification_metrics
from tcga_brca.roc import one_vs_rest_roc
from tcga_brca.differential_expression import welch_de
from tcga_brca.plots import plot_volcano

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/"src"))
RESULTS = ROOT/"results"; FIGS = ROOT/"figures"
RESULTS.mkdir(exist_ok=True); FIGS.mkdir(exist_ok=True)

X_ml, X_bio, y, excluded = build_dual_cohorts(
    ROOT/"data/raw/HiSeqV2.gz",
    ROOT/"data/raw/BRCA_clinicalMatrix",
)

Xtr, Xte, ytr, yte = train_test_split(
    X_ml, y, test_size=.20, stratify=y, random_state=42
)

# Choose model family using nested-CV summary generated previously.
summary_path = RESULTS/"nested_cv_summary.csv"
if not summary_path.exists():
    raise FileNotFoundError(
        "Run `python scripts/run_nested_cv.py` first."
    )

summary = pd.read_csv(summary_path)
winner = summary.iloc[0]["model"]
pipe, grid = search_spaces()[winner]

inner = StratifiedKFold(3, shuffle=True, random_state=42)
search = GridSearchCV(
    pipe, grid, scoring="balanced_accuracy",
    cv=inner, n_jobs=-1, refit=True
)
search.fit(Xtr, ytr)
best = search.best_estimator_

pred = best.predict(Xte)
proba = best.predict_proba(Xte)
classes = list(best.classes_)

metrics = {
    "model": winner,
    "balanced_accuracy": balanced_accuracy_score(yte, pred),
    "f1_macro": f1_score(yte, pred, average="macro"),
    "f1_weighted": f1_score(yte, pred, average="weighted"),
}
pd.DataFrame([metrics]).to_csv(RESULTS/"heldout_metrics.csv", index=False)
(RESULTS/"classification_report.txt").write_text(
    classification_report(yte, pred)
)
(RESULTS/"final_best_params.json").write_text(
    json.dumps(search.best_params_, indent=2)
)

# Confusion matrix
fig, ax = plt.subplots(figsize=(6,5))
ConfusionMatrixDisplay.from_predictions(
    yte, pred, labels=classes, ax=ax
)
ax.set_title("Held-out TCGA-BRCA subtype classification")
fig.tight_layout()
fig.savefig(FIGS/"confusion_matrix.png", dpi=180)
plt.close(fig)

# One-vs-rest ROC
curves = one_vs_rest_roc(yte, proba, classes)
fig, ax = plt.subplots(figsize=(7,6))
for cls, c in curves.items():
    ax.plot(c["fpr"], c["tpr"], label=f"{cls} (AUC={c['auc']:.3f})")
ax.plot([0,1],[0,1],linestyle="--")
ax.set_xlabel("False positive rate")
ax.set_ylabel("True positive rate")
ax.set_title("One-vs-rest ROC curves")
ax.legend()
fig.tight_layout()
fig.savefig(FIGS/"roc_ovr.png", dpi=180)
plt.close(fig)

# Bootstrap uncertainty
boot, ci = bootstrap_classification_metrics(
    yte.to_numpy(), pred, proba, classes, n_boot=1000
)
boot.to_csv(RESULTS/"bootstrap_metrics.csv", index=False)
ci.to_csv(RESULTS/"bootstrap_ci.csv", index=False)

# Stability of final tuned model across training folds
stability = feature_selection_stability(best, Xtr, ytr)
stability.to_csv(RESULTS/"feature_selection_stability.csv", index=False)
stability.head(50).to_csv(RESULTS/"top_stable_features.csv", index=False)

# Biology branch: use FULL transcriptome and TRAINING PATIENTS ONLY.
bio_train = X_bio.loc[Xtr.index]
de = welch_de(bio_train, ytr, "Basal", "LumA")
de.to_csv(RESULTS/"de_basal_vs_luma_full_transcriptome.csv", index=False)

fig, ax = plot_volcano(de)
fig.savefig(FIGS/"volcano_basal_vs_luma.png", dpi=180)
plt.close(fig)

print("Final model:", winner)
print("Best params:", search.best_params_)
print(metrics)
print(ci.to_string(index=False))
