from pathlib import Path
import pandas as pd

def run_gprofiler(genes, output_path=None):
    try:
        from gprofiler import GProfiler
    except ImportError as exc:
        raise ImportError(
            "Install optional dependency: pip install gprofiler-official"
        ) from exc

    genes = [str(g) for g in genes if pd.notna(g)]
    gp = GProfiler(return_dataframe=True)
    res = gp.profile(
        organism="hsapiens",
        query=genes,
        sources=["GO:BP","REAC","KEGG","CORUM"],
        user_threshold=0.05,
        significance_threshold_method="fdr",
    )
    if output_path:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        res.to_csv(output_path, index=False)
    return res
