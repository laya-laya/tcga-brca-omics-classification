from pathlib import Path
import json
import pandas as pd
from .io import load_xena_expression, load_xena_clinical
from .constants import PAM50_GENES, MODEL_CLASSES

def sample_type_code(sample_ids):
    """Extract the two-digit TCGA sample type from barcodes."""
    s = pd.Index(sample_ids).astype(str)
    part = s.str.split("-").str[3]
    return part.str[:2]

def primary_tumor_mask(sample_ids):
    """Return boolean mask for TCGA primary solid tumor samples (type 01)."""
    return sample_type_code(sample_ids) == "01"

def clean_expression(expr):
    """Remove unusable genes and median-impute remaining missing values."""
    expr = expr.dropna(axis=1, how="all")
    expr = expr.loc[:, expr.nunique(dropna=True) > 1]
    medians = expr.median(numeric_only=True)
    expr = expr.fillna(medians)
    return expr

def build_cohort(expression_path, clinical_path, exclude_pam50):
    """Build aligned primary-tumor TCGA-BRCA cohort.

    Returns
    -------
    X : DataFrame
        Samples x genes.
    y : Series
        PAM50 subtype labels.
    excluded : list[str]
        PAM50 genes removed from X when exclude_pam50=True.
    """
    expr = load_xena_expression(expression_path)
    clinical = load_xena_clinical(clinical_path)

    expr = expr.loc[primary_tumor_mask(expr.index)]
    if "PAM50Call_RNAseq" not in clinical.columns:
        raise KeyError("Clinical matrix lacks PAM50Call_RNAseq.")

    y = clinical["PAM50Call_RNAseq"].reindex(expr.index)
    keep = y.isin(MODEL_CLASSES)
    expr = expr.loc[keep]
    y = y.loc[keep].astype(str)

    expr = clean_expression(expr)

    excluded = []
    if exclude_pam50:
        excluded = sorted(set(expr.columns).intersection(PAM50_GENES))
        expr = expr.drop(columns=excluded, errors="ignore")

    return expr.astype("float32"), y.rename("PAM50"), excluded

def build_dual_cohorts(expression_path, clinical_path):
    """Create one matrix for prediction and one full matrix for biology."""
    X_ml, y_ml, excluded = build_cohort(
        expression_path, clinical_path, exclude_pam50=True
    )
    X_bio, y_bio, _ = build_cohort(
        expression_path, clinical_path, exclude_pam50=False
    )

    common = X_ml.index.intersection(X_bio.index)
    X_ml = X_ml.loc[common]
    X_bio = X_bio.loc[common]
    y_ml = y_ml.loc[common]
    y_bio = y_bio.loc[common]

    if not y_ml.equals(y_bio):
        raise ValueError("ML and biology cohort labels are not aligned.")

    return X_ml, X_bio, y_ml, excluded

def save_processed(expression_path, clinical_path, out_dir):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    X_ml, X_bio, y, excluded = build_dual_cohorts(expression_path, clinical_path)

    X_ml.to_pickle(out_dir/"expression_ml_nonpam50.pkl.gz", compression="gzip")
    X_bio.to_pickle(out_dir/"expression_biology_full.pkl.gz", compression="gzip")
    y.to_frame().to_csv(out_dir/"labels.csv")

    meta = {
        "n_samples": int(len(y)),
        "n_ml_genes": int(X_ml.shape[1]),
        "n_biology_genes": int(X_bio.shape[1]),
        "class_counts": y.value_counts().to_dict(),
        "excluded_pam50_genes": excluded,
    }
    (out_dir/"cohort_summary.json").write_text(json.dumps(meta, indent=2))
    return meta
