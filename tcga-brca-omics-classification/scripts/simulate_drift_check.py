"""How often does the drift check raise false alarms, and what does it catch?

Simulates batches of different sizes from the training distribution (no
drift) and with batch effects of increasing size, and reports how often the
batch is flagged. Uses the synthetic demo cohort, a reference of TCGA size
(about 650 training samples) and the model's 50 selected genes.

    python scripts/simulate_drift_check.py
"""
from pathlib import Path
import sys
import warnings

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from tcga_brca.demo import demo_bundle, synthetic_cohort  # noqa: E402
from tcga_brca.drift import drift_report, reference_profile  # noqa: E402

SIZES = [10, 30, 100, 400]
SHIFTS_SD = [0.0, 0.25, 0.5, 1.0, 2.0]   # shift in units of the gene's SD (1.5)
REPEATS = 100


def main():
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", FutureWarning)
        genes = demo_bundle()["selected_features"]
    X_ref, _ = synthetic_cohort(n_samples=650, seed=7)
    reference = reference_profile(X_ref[genes])

    rows = []
    for shift in SHIFTS_SD:
        row = {"shift (SD) on half the genes": shift}
        for n in SIZES:
            flags = [
                drift_report(
                    synthetic_cohort(n_samples=n, seed=1000 + r, shift=1.5 * shift)[0][genes],
                    reference,
                )["drift_detected"]
                for r in range(REPEATS)
            ]
            row[f"n={n}"] = f"{np.mean(flags):.0%}"
        rows.append(row)

    table = pd.DataFrame(rows)
    print("Share of batches flagged as drifted "
          f"({REPEATS} simulated batches per cell):\n")
    print(table.to_string(index=False))


if __name__ == "__main__":
    main()
