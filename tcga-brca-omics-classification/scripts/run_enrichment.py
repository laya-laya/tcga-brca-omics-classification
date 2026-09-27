from pathlib import Path
import sys
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/"src"))

from tcga_brca.enrichment import run_gprofiler
from tcga_brca.plots import plot_enrichment

de = pd.read_csv(ROOT/"results/de_basal_vs_luma_full_transcriptome.csv")

sig = de[
    (de["p_adj"] < 0.01)
    & (de["difference_a_minus_b"].abs() >= 1.0)
].copy()

# Analyze directions separately.
for label, subset in {
    "basal_high": sig[sig["difference_a_minus_b"] > 0],
    "luma_high": sig[sig["difference_a_minus_b"] < 0],
}.items():
    genes = subset.sort_values("p_adj").head(500)["gene"].tolist()
    res = run_gprofiler(
        genes,
        ROOT/f"results/gprofiler_{label}.csv"
    )
    if len(res):
        fig, ax = plot_enrichment(res, top_n=20)
        fig.savefig(ROOT/f"figures/enrichment_{label}.png", dpi=180)

print("Enrichment complete.")
