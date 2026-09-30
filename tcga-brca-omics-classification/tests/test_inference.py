import json

import numpy as np
import pandas as pd
import pytest

from tcga_brca.artifact import load_bundle, model_card
from tcga_brca.demo import write_xena
from tcga_brca.inference import SchemaError, predict
from tcga_brca.predict import EXIT_DRIFT, EXIT_INPUT_REJECTED, EXIT_OK, main


def test_bundle_roundtrip_keeps_predictions(bundle, model_path, batch):
    loaded = load_bundle(model_path)
    a = predict(batch, bundle, check_drift=False)["predictions"]
    b = predict(batch, loaded, check_drift=False)["predictions"]
    pd.testing.assert_frame_equal(a, b)
    card = model_card(loaded)
    json.dumps(card)  # must be JSON-serialisable
    assert card["n_model_genes"] == len(bundle["selected_features"])


def test_predictions_are_valid_probabilities(bundle, batch):
    result = predict(batch, bundle)
    preds = result["predictions"]
    probs = preds.filter(like="prob_")
    assert len(preds) == len(batch)
    np.testing.assert_allclose(probs.sum(axis=1), 1.0, atol=1e-6)
    assert set(preds["predicted_subtype"]) <= set(bundle["classes"])
    assert not result["drift"]["drift_detected"]
    json.dumps({k: v for k, v in result.items() if k != "predictions"})


def test_column_order_and_extra_genes_do_not_change_predictions(bundle, batch):
    base = predict(batch, bundle, check_drift=False)["predictions"]
    messy = batch.sample(frac=1.0, axis=1, random_state=0).assign(NOT_A_GENE=1.0)
    result = predict(messy, bundle, check_drift=False)
    pd.testing.assert_frame_equal(base, result["predictions"])
    assert result["validation"]["n_unknown_genes_ignored"] == 1


def test_genes_the_model_ignores_may_be_missing(bundle, batch):
    unused = [g for g in bundle["feature_names"] if g not in bundle["selected_features"]]
    result = predict(batch.drop(columns=unused[:20]), bundle, check_drift=False)
    assert result["validation"]["n_model_genes_missing"] == 0


def test_too_many_missing_model_genes_is_rejected(bundle, batch):
    with pytest.raises(SchemaError, match="missing"):
        predict(batch.drop(columns=bundle["selected_features"][:10]), bundle)


def test_occasional_missing_values_are_imputed(bundle, batch):
    batch = batch.copy()
    batch.iloc[0, :5] = np.nan
    result = predict(batch, bundle, check_drift=False)
    assert result["predictions"].notna().all().all()


def test_empty_input_is_rejected(bundle, batch):
    with pytest.raises(SchemaError):
        predict(batch.iloc[:0], bundle)


def test_batch_job_writes_predictions_and_report(model_path, batch, tmp_path):
    src = write_xena(batch, tmp_path / "batch.tsv")
    out = tmp_path / "out" / "predictions.csv"
    code = main(["--model", str(model_path), "--input", str(src), "--output", str(out)])
    assert code == EXIT_OK
    preds = pd.read_csv(out, index_col="sample_id")
    assert list(preds.index) == list(batch.index)
    report = json.loads(out.with_suffix(".report.json").read_text())
    assert report["validation"]["n_samples"] == len(batch)
    assert report["drift"]["drift_detected"] is False


def test_batch_job_fails_on_drift_when_asked(model_path, shifted_batch, tmp_path):
    src = write_xena(shifted_batch, tmp_path / "shifted.tsv")
    args = ["--model", str(model_path), "--input", str(src),
            "--output", str(tmp_path / "p.csv")]
    assert main(args) == EXIT_OK
    assert main(args + ["--fail-on-drift"]) == EXIT_DRIFT


def test_batch_job_rejects_wrong_input(model_path, tmp_path):
    wrong = pd.DataFrame(np.ones((5, 3)), columns=["A", "B", "C"],
                         index=[f"S{i}" for i in range(5)])
    src = tmp_path / "wrong.csv"
    wrong.to_csv(src)
    code = main(["--model", str(model_path), "--input", str(src), "--samples-as-rows",
                 "--output", str(tmp_path / "p.csv")])
    assert code == EXIT_INPUT_REJECTED
