from pathlib import Path
import sys
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/"src"))
X = pd.read_pickle(ROOT/"data/processed/expression_biology_full.pkl.gz")
y = pd.read_csv(ROOT/"data/processed/labels.csv", index_col=0)["PAM50"]
de = pd.read_csv(ROOT/"results/de_basal_vs_luma_full_transcriptome.csv")

top = (
    de.dropna(subset=["p_adj"])
      .sort_values("p_adj")
      .head(40)["gene"]
      .tolist()
)
common = [g for g in top if g in X.columns]

# Order samples by subtype for a clean portfolio visualization.
order = y.sort_values().index
M = X.loc[order, common]
Z = StandardScaler().fit_transform(M)
Z = pd.DataFrame(Z, index=order, columns=common)

fig, ax = plt.subplots(figsize=(14,8))
im = ax.imshow(Z.T, aspect="auto", interpolation="nearest")
ax.set_yticks(range(len(common)), common, fontsize=7)
ax.set_xticks([])
ax.set_xlabel("TCGA-BRCA tumors ordered by PAM50 subtype")
ax.set_title("Top differential genes: standardized expression heatmap")
fig.colorbar(im, ax=ax, label="z-score")
fig.tight_layout()
fig.savefig(ROOT/"figures/top_gene_heatmap.png", dpi=180)
