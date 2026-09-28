"""Data-drift monitoring for incoming expression data.

The classifier learned one distribution of gene expression. If new samples
come from a different lab, sequencing protocol or normalisation (for example
raw counts instead of log2 values), predictions can degrade silently, because
the model still returns confident-looking probabilities. This module compares
each monitored gene in a new batch against the training distribution.

For every gene the model actually uses:

* Bins are fixed on the training data (deciles) when the model is trained.
* PSI (Population Stability Index) measures how far the new batch's bin
  proportions moved. It is the usual industry effect size: < 0.1 stable,
  0.1-0.25 moderate shift, > 0.25 large shift.
* Raw PSI is inflated by sampling noise, especially in small batches, so the
  expected no-drift value is subtracted before comparing with the threshold
  (``psi_adjusted``).
* A two-sample chi-square test on the same bins checks the shift is not just
  noise, with Benjamini-Hochberg FDR correction across genes.

A gene counts as drifted only if the shift is both significant (q < FDR) and
large (adjusted PSI >= threshold). The batch is flagged when the share of
drifted genes exceeds a threshold.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import chi2
from statsmodels.stats.multitest import multipletests

N_BINS = 10
PSI_THRESHOLD = 0.2          # per gene: minimum shift size that matters
FDR = 0.05                   # per gene: significance after BH correction
SHARE_THRESHOLD = 0.10       # per batch: flag if >10% of monitored genes drift
MIN_RELIABLE_BATCH = 30      # below this, report a warning note
_SMOOTHING = 0.5             # additive smoothing so empty bins stay finite
_MIN_PROPORTION = 1e-6


def _finite(values) -> np.ndarray:
    v = pd.to_numeric(pd.Series(values), errors="coerce").to_numpy(dtype=float)
    return v[np.isfinite(v)]


def _bin_counts(values: np.ndarray, edges: np.ndarray) -> np.ndarray:
    # A value equal to an edge goes to the upper bin (same as np.digitize).
    idx = np.searchsorted(edges, values, side="right")
    return np.bincount(idx, minlength=len(edges) + 1)


def reference_profile(X_ref: pd.DataFrame, n_bins: int = N_BINS) -> dict:
    """Store per-gene bin edges and bin proportions from the training data."""
    quantiles = np.linspace(0, 1, n_bins + 1)[1:-1]
    edges, expected = {}, {}
    for gene in X_ref.columns:
        values = _finite(X_ref[gene])
        if values.size == 0:
            continue
        gene_edges = np.unique(np.quantile(values, quantiles))
        counts = _bin_counts(values, gene_edges)
        edges[str(gene)] = gene_edges
        expected[str(gene)] = counts / counts.sum()
    return {
        "n_reference_samples": int(len(X_ref)),
        "n_bins": n_bins,
        "edges": edges,
        "expected": expected,
    }


def population_stability_index(expected, counts) -> float:
    """PSI between reference proportions and observed bin counts."""
    exp = np.clip(np.asarray(expected, dtype=float), _MIN_PROPORTION, None)
    exp = exp / exp.sum()
    counts = np.asarray(counts, dtype=float)
    act = (counts + _SMOOTHING) / (counts.sum() + _SMOOTHING * len(counts))
    return float(np.sum((act - exp) * np.log(act / exp)))


def psi_null_bias(n_bins: int, n_new: int, n_reference: int) -> float:
    """Approximate PSI expected from sampling noise alone (no real drift).

    For small differences PSI behaves like a chi-square statistic divided by
    the effective sample size, so its mean under no drift is about
    (bins - 1) * (1/n_new + 1/n_reference). Subtracting it makes the PSI
    threshold mean the same thing for a batch of 20 or of 2,000 samples.
    """
    return (n_bins - 1) * (1.0 / max(n_new, 1) + 1.0 / max(n_reference, 1))


def two_sample_chi_square_pvalue(expected, n_reference, counts) -> float:
    """Homogeneity test: are new counts and reference counts from one distribution?

    Two-sample (rather than goodness-of-fit) because the reference bins are
    themselves estimated from a finite training set.
    """
    ref = np.asarray(expected, dtype=float) * n_reference
    new = np.asarray(counts, dtype=float)
    n_new = new.sum()
    keep = (ref + new) > 0
    ref, new = ref[keep], new[keep]
    if n_new == 0 or keep.sum() < 2:
        return 1.0
    pooled = (ref + new) / (n_reference + n_new)
    exp_ref, exp_new = pooled * n_reference, pooled * n_new
    stat = float(np.sum((ref - exp_ref) ** 2 / exp_ref) + np.sum((new - exp_new) ** 2 / exp_new))
    return float(chi2.sf(stat, df=keep.sum() - 1))


def drift_report(
    X_new: pd.DataFrame,
    reference: dict,
    *,
    psi_threshold: float = PSI_THRESHOLD,
    fdr: float = FDR,
    share_threshold: float = SHARE_THRESHOLD,
    min_reliable_batch: int = MIN_RELIABLE_BATCH,
    top_n: int = 20,
) -> dict:
    """Compare a batch of new samples (samples x genes) with the reference.

    Missing values are ignored per gene; genes absent from ``X_new`` are
    skipped (input validation reports those separately). Returns a
    JSON-serialisable dict.
    """
    thresholds = {
        "psi": psi_threshold,
        "fdr": fdr,
        "share_of_genes": share_threshold,
    }
    n_samples = int(len(X_new))
    n_reference = reference["n_reference_samples"]

    rows = []
    for gene, edges in reference["edges"].items():
        if gene not in X_new.columns:
            continue
        values = _finite(X_new[gene])
        if values.size == 0:
            continue
        counts = _bin_counts(values, edges)
        expected = reference["expected"][gene]
        psi = population_stability_index(expected, counts)
        bias = psi_null_bias(len(counts), values.size, n_reference)
        rows.append((
            gene,
            psi,
            max(psi - bias, 0.0),
            two_sample_chi_square_pvalue(expected, n_reference, counts),
        ))

    if not rows:
        return {
            "n_samples": n_samples,
            "n_genes_monitored": 0,
            "n_genes_drifted": 0,
            "share_genes_drifted": 0.0,
            "median_psi": None,
            "drift_detected": False,
            "thresholds": thresholds,
            "top_drifted_genes": [],
            "note": "None of the monitored genes had usable values.",
        }

    table = pd.DataFrame(
        rows, columns=["gene", "psi", "psi_adjusted", "p_value"]
    ).set_index("gene")
    table["q_value"] = multipletests(table["p_value"], method="fdr_bh")[1]
    table["drifted"] = (table["q_value"] < fdr) & (table["psi_adjusted"] >= psi_threshold)
    share = float(table["drifted"].mean())

    top = table[table["drifted"]].sort_values("psi_adjusted", ascending=False).head(top_n)
    note = None
    if n_samples < min_reliable_batch:
        note = (
            f"Only {n_samples} samples: drift statistics have little power "
            f"below {min_reliable_batch}, so a clean result is weak evidence."
        )

    return {
        "n_samples": n_samples,
        "n_genes_monitored": int(len(table)),
        "n_genes_drifted": int(table["drifted"].sum()),
        "share_genes_drifted": round(share, 4),
        "median_psi": round(float(table["psi"].median()), 4),
        "drift_detected": bool(share > share_threshold),
        "thresholds": thresholds,
        "top_drifted_genes": [
            {
                "gene": gene,
                "psi": round(float(r.psi), 4),
                "psi_adjusted": round(float(r.psi_adjusted), 4),
                "q_value": float(f"{r.q_value:.3g}"),
            }
            for gene, r in top.iterrows()
        ],
        "note": note,
    }
