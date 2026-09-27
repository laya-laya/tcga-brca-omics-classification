from pathlib import Path
import pandas as pd

def load_xena_expression(path):
    """Load UCSC Xena HiSeqV2 expression and return samples x genes."""
    path = Path(path)
    compression = "gzip" if path.suffix == ".gz" else None
    x = pd.read_table(path, index_col=0, compression=compression).T
    x.index = x.index.astype(str)
    x = x.apply(pd.to_numeric, errors="coerce")
    return x

def load_xena_clinical(path):
    """Load UCSC Xena BRCA clinical matrix indexed by sampleID."""
    c = pd.read_table(path, index_col="sampleID", low_memory=False)
    c.index = c.index.astype(str)
    return c
