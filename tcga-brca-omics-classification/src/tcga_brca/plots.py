import numpy as np
import matplotlib.pyplot as plt

def plot_volcano(de, padj_threshold=0.01, effect_threshold=1.0, top_labels=12):
    valid = (
        np.isfinite(de["difference_a_minus_b"])
        & np.isfinite(de["p_adj"])
        & (de["p_adj"] > 0)
    )
    d = de.loc[valid].copy()
    x = d["difference_a_minus_b"]
    y = -np.log10(d["p_adj"].clip(lower=1e-300))
    sig = (d["p_adj"] < padj_threshold) & (x.abs() >= effect_threshold)

    fig, ax = plt.subplots(figsize=(8,6))
    ax.scatter(x[~sig], y[~sig], s=8, alpha=.25)
    ax.scatter(x[sig], y[sig], s=12, alpha=.65)
    ax.axvline(-effect_threshold, linestyle="--")
    ax.axvline(effect_threshold, linestyle="--")
    ax.axhline(-np.log10(padj_threshold), linestyle="--")
    ax.set_xlabel("Mean-expression difference: Basal - LumA")
    ax.set_ylabel("-log10(FDR)")
    ax.set_title("Basal vs LumA differential expression")

    label_df = d.loc[sig].copy()
    label_df["score"] = y[sig]
    label_df = label_df.sort_values("score", ascending=False).head(top_labels)
    for _, row in label_df.iterrows():
        yy = -np.log10(max(row["p_adj"], 1e-300))
        ax.annotate(
            str(row["gene"]),
            (row["difference_a_minus_b"], yy),
            xytext=(3,3), textcoords="offset points", fontsize=8
        )
    fig.tight_layout()
    return fig, ax

def plot_enrichment(enrichment, top_n=20):
    d = enrichment.copy()
    d = d[np.isfinite(d["p_value"])].sort_values("p_value").head(top_n)
    scores = -np.log10(d["p_value"].clip(lower=1e-300))
    order = np.arange(len(d))[::-1]

    fig, ax = plt.subplots(figsize=(9, max(5, .32*len(d))))
    ax.barh(order, scores.iloc[::-1])
    ax.set_yticks(order, d["name"].iloc[::-1])
    ax.set_xlabel("-log10(adjusted enrichment p-value)")
    ax.set_title("Top enriched biological pathways/processes")
    fig.tight_layout()
    return fig, ax
