import numpy as np
import pandas as pd

from tcga_brca.drift import (
    drift_report,
    population_stability_index,
    reference_profile,
    two_sample_chi_square_pvalue,
)


def _normal(n, shift=0.0, seed=0, genes=20):
    rng = np.random.default_rng(seed)
    return pd.DataFrame(rng.normal(shift, 1.0, size=(n, genes)),
                        columns=[f"G{i}" for i in range(genes)])


def test_psi_is_zero_for_identical_distributions():
    expected = np.full(10, 0.1)
    assert population_stability_index(expected, np.full(10, 1000)) < 1e-6


def test_chi_square_detects_a_clear_shift():
    expected = np.full(10, 0.1)
    same = np.full(10, 20)
    shifted = np.array([0, 0, 0, 0, 0, 40, 40, 40, 40, 40])
    assert two_sample_chi_square_pvalue(expected, 500, same) > 0.5
    assert two_sample_chi_square_pvalue(expected, 500, shifted) < 1e-6


def test_no_drift_for_same_distribution():
    ref = reference_profile(_normal(500, seed=0))
    for n in (10, 50, 400):
        report = drift_report(_normal(n, seed=n), ref)
        assert not report["drift_detected"]
        assert report["n_genes_monitored"] == 20


def test_drift_detected_for_shifted_distribution():
    ref = reference_profile(_normal(500, seed=0))
    report = drift_report(_normal(100, shift=1.0, seed=3), ref)
    assert report["drift_detected"]
    assert report["n_genes_drifted"] > 10
    assert report["top_drifted_genes"][0]["psi_adjusted"] >= 0.2


def test_small_batch_gets_a_warning_note():
    ref = reference_profile(_normal(500, seed=0))
    assert drift_report(_normal(10, seed=1), ref)["note"]
    assert drift_report(_normal(100, seed=1), ref)["note"] is None


def test_missing_values_and_genes_are_skipped():
    ref = reference_profile(_normal(500, seed=0))
    X = _normal(80, seed=4).drop(columns=["G0", "G1"])
    X.iloc[:10, 0] = np.nan
    report = drift_report(X, ref)
    assert report["n_genes_monitored"] == 18
    assert not report["drift_detected"]
