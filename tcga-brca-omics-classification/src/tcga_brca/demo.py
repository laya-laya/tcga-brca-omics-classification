"""Synthetic stand-in for the TCGA data.

Used by the tests, the CI smoke tests and `make demo`, so the serving code
can be exercised without downloading TCGA. The data only mimic the layout
(log2-like expression, four PAM50 classes, TCGA-style barcodes); they carry
no biology.

    python -m tcga_brca.demo --model models/demo.joblib --samples data/demo/batch.tsv
    python -m tcga_brca.demo --samples data/demo/shifted.tsv --shift 1.5   # simulate drift
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from .artifact import build_bundle, save_bundle
from .constants import MODEL_CLASSES
from .models import elastic_net_pipeline
from .training import fit_final_model

DEMO_PARAMS = {"select__k": 50, "model__C": 1.0, "model__l1_ratio": 0.5}


def synthetic_cohort(n_samples=240, n_genes=300, seed=0, shift=0.0,
                     shifted_share=0.5):
    """Samples x genes expression and labels.

    Each class raises its own block of 10 genes. ``shift`` adds a constant to
    the first ``shifted_share`` of genes, simulating a batch effect.
    """
    rng = np.random.default_rng(seed)
    y = rng.choice(MODEL_CLASSES, size=n_samples, p=[0.2, 0.15, 0.4, 0.25])
    X = rng.normal(8.0, 1.5, size=(n_samples, n_genes))
    for i, cls in enumerate(MODEL_CLASSES):
        X[y == cls, i * 10:(i + 1) * 10] += 2.0
    if shift:
        X[:, : int(n_genes * shifted_share)] += shift
    index = [f"TCGA-SY-{seed:02d}{i:04d}-01A" for i in range(n_samples)]
    columns = [f"GENE{j:04d}" for j in range(n_genes)]
    X = pd.DataFrame(X, index=index, columns=columns).astype("float32")
    return X, pd.Series(y, index=index, name="PAM50")


def demo_bundle(seed=0):
    """Train the production pipeline (small k) on synthetic data."""
    X, y = synthetic_cohort(seed=seed)
    model, metrics, (X_train, *_rest) = fit_final_model(
        elastic_net_pipeline(), DEMO_PARAMS, X, y
    )
    return build_bundle(model, X_train, metrics=metrics, params=DEMO_PARAMS,
                        model_name="elastic_net (synthetic demo data)")


def write_xena(X: pd.DataFrame, path) -> Path:
    """Write samples x genes as a Xena-style genes x samples TSV."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    X.T.to_csv(path, sep="\t", index_label="sample", float_format="%.4f")
    return path


def main(argv=None):
    parser = argparse.ArgumentParser(description="Create a demo model and/or batch.")
    parser.add_argument("--model", type=Path, help="where to save a demo model bundle")
    parser.add_argument("--samples", type=Path, help="where to write a synthetic batch (Xena TSV)")
    parser.add_argument("--n", type=int, default=60, help="samples in the batch")
    parser.add_argument("--shift", type=float, default=0.0, help="simulated batch effect")
    parser.add_argument("--seed", type=int, default=1)
    args = parser.parse_args(argv)
    if not (args.model or args.samples):
        parser.error("give --model and/or --samples")
    if args.model:
        save_bundle(demo_bundle(), args.model)
        print(f"Saved demo model to {args.model}")
    if args.samples:
        X, _ = synthetic_cohort(n_samples=args.n, seed=args.seed, shift=args.shift)
        write_xena(X, args.samples)
        print(f"Wrote {args.n} synthetic samples to {args.samples}")


if __name__ == "__main__":
    main()
