"""Batch prediction job: score a file of new samples and write a report.

    python -m tcga_brca.predict --input new_samples.tsv.gz --output predictions/predictions.csv

Input is expression in UCSC Xena layout (genes as rows, samples as columns,
tab-separated, optionally gzipped), i.e. the same format as the training
data. Pass --samples-as-rows for a samples x genes CSV/TSV instead.

Outputs:
  * predictions CSV: one row per sample with subtype, confidence and
    per-class probabilities
  * report JSON (default: next to the CSV): input validation, data drift
    and prediction summary

Exit codes: 0 success, 2 input rejected, 3 drift detected (with --fail-on-drift).
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from pathlib import Path

import pandas as pd

from .artifact import DEFAULT_MODEL_PATH, load_bundle
from .inference import SchemaError, predict, report
from .io import load_xena_expression

log = logging.getLogger("tcga_brca.predict")

EXIT_OK, EXIT_INPUT_REJECTED, EXIT_DRIFT = 0, 2, 3


def load_samples(path: Path, samples_as_rows: bool = False) -> pd.DataFrame:
    """Read new samples as a samples x genes table."""
    if not samples_as_rows:
        return load_xena_expression(path)
    name = path.name.lower()
    sep = "," if name.endswith((".csv", ".csv.gz")) else "\t"
    return pd.read_csv(path, sep=sep, index_col=0)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m tcga_brca.predict",
        description="Score new samples with the PAM50 subtype classifier.",
    )
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path, help="predictions CSV")
    parser.add_argument("--report", type=Path, help="report JSON (default: <output>.report.json)")
    parser.add_argument("--model", type=Path,
                        default=Path(os.environ.get("MODEL_PATH", DEFAULT_MODEL_PATH)))
    parser.add_argument("--samples-as-rows", action="store_true",
                        help="input is samples x genes instead of Xena genes x samples")
    parser.add_argument("--no-drift", action="store_true", help="skip the drift check")
    parser.add_argument("--fail-on-drift", action="store_true",
                        help="exit with code 3 when drift is detected")
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    bundle = load_bundle(args.model)
    log.info("Loaded model %s (trained %s)", args.model,
             bundle["metadata"].get("trained_at"))

    samples = load_samples(args.input, args.samples_as_rows)
    log.info("Read %d samples x %d genes from %s", *samples.shape, args.input)

    try:
        result = predict(samples, bundle, check_drift=not args.no_drift)
    except SchemaError as exc:
        log.error("Input rejected: %s", exc)
        return EXIT_INPUT_REJECTED

    args.output.parent.mkdir(parents=True, exist_ok=True)
    result["predictions"].to_csv(args.output, float_format="%.6f")
    report_path = args.report or args.output.with_suffix(".report.json")
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report(result), indent=2) + "\n")

    summary, drift = result["summary"], result["drift"]
    log.info("Predicted subtypes: %s", summary["predicted_subtype_counts"])
    log.info("Mean confidence %.3f; %.1f%% below %.2f", summary["mean_confidence"],
             100 * summary["share_low_confidence"], summary["low_confidence_threshold"])
    log.info("Wrote %s and %s", args.output, report_path)

    if drift is not None:
        if drift.get("note"):
            log.warning(drift["note"])
        if drift["drift_detected"]:
            log.warning("DATA DRIFT: %d of %d model genes shifted (%.1f%%); see %s",
                        drift["n_genes_drifted"], drift["n_genes_monitored"],
                        100 * drift["share_genes_drifted"], report_path)
            if args.fail_on_drift:
                return EXIT_DRIFT
        else:
            log.info("No data drift detected (%d genes monitored)",
                     drift["n_genes_monitored"])
    return EXIT_OK


if __name__ == "__main__":
    sys.exit(main())
