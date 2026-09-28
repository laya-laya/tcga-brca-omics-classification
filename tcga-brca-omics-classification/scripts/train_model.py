"""Train the deployable PAM50 classifier and save it as a model bundle.

No tuning happens here. The model family comes from the nested-CV summary
and the hyperparameters from results/final_best_params.json, and the 80/20
split is the same one run_full_analysis.py uses, so the held-out metrics are
directly comparable with the published results.

    python scripts/download_tcga_brca.py      # once
    python scripts/train_model.py

Writes models/pam50_classifier.joblib (the bundle) and
models/pam50_classifier.json (a human-readable model card).
"""
from pathlib import Path
import argparse
import json
import sys

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from tcga_brca.artifact import build_bundle, model_card, save_bundle  # noqa: E402
from tcga_brca.models import search_spaces  # noqa: E402
from tcga_brca.preprocess import build_dual_cohorts  # noqa: E402
from tcga_brca.training import fit_final_model  # noqa: E402


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--expression", type=Path, default=ROOT / "data/raw/HiSeqV2.gz")
    parser.add_argument("--clinical", type=Path, default=ROOT / "data/raw/BRCA_clinicalMatrix")
    parser.add_argument("--results", type=Path, default=ROOT / "results")
    parser.add_argument("--out", type=Path, default=ROOT / "models/pam50_classifier.joblib")
    args = parser.parse_args(argv)

    for path in (args.expression, args.clinical):
        if not path.exists():
            sys.exit(f"Missing {path}. Run `python scripts/download_tcga_brca.py` first.")

    summary = pd.read_csv(args.results / "nested_cv_summary.csv")
    winner = summary.iloc[0]["model"]
    params = json.loads((args.results / "final_best_params.json").read_text())
    pipeline, _ = search_spaces()[winner]

    print(f"Building cohort from {args.expression.name} ...")
    X, _, y, _ = build_dual_cohorts(args.expression, args.clinical)

    print(f"Fitting {winner} with {params} ...")
    model, metrics, (X_train, *_rest) = fit_final_model(pipeline, params, X, y)

    bundle = build_bundle(model, X_train, metrics=metrics, params=params, model_name=winner)
    path = save_bundle(bundle, args.out)
    card = model_card(bundle)
    args.out.with_suffix(".json").write_text(json.dumps(card, indent=2) + "\n")

    print(f"Saved {path} ({path.stat().st_size / 1e6:.1f} MB)")
    print("Held-out metrics:", json.dumps(card["heldout_metrics"], indent=2))

    published = args.results / "heldout_metrics.csv"
    if published.exists():
        ref = pd.read_csv(published).iloc[0]
        diff = abs(ref["balanced_accuracy"] - metrics["balanced_accuracy"])
        status = "matches" if diff < 1e-3 else f"differs by {diff:.4f} from"
        print(f"Balanced accuracy {status} results/heldout_metrics.csv")


if __name__ == "__main__":
    main()
