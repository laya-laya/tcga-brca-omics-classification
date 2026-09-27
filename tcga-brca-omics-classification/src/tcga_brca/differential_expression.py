import numpy as np
import pandas as pd
from scipy.stats import ttest_ind
from statsmodels.stats.multitest import multipletests

def welch_de(expression, labels, group_a="Basal", group_b="LumA"):
    """Exploratory DE on processed Xena expression.

    The input is already processed continuous expression; this is not a
    replacement for count-aware DESeq2 on raw counts.
    """
    a = expression.loc[labels == group_a]
    b = expression.loc[labels == group_b]

    stat, p = ttest_ind(
        a.to_numpy(dtype=float),
        b.to_numpy(dtype=float),
        axis=0,
        equal_var=False,
        nan_policy="omit",
    )

    valid = np.isfinite(p)
    padj = np.full(len(p), np.nan, dtype=float)
    padj[valid] = multipletests(p[valid], method="fdr_bh")[1]

    mean_a = a.mean(axis=0).to_numpy()
    mean_b = b.mean(axis=0).to_numpy()

    out = pd.DataFrame({
        "gene": expression.columns,
        "mean_a": mean_a,
        "mean_b": mean_b,
        "difference_a_minus_b": mean_a - mean_b,
        "t_statistic": stat,
        "p_value": p,
        "p_adj": padj,
    })
    return out.sort_values(["p_adj","p_value"], na_position="last")
