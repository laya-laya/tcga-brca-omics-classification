# Data

Large TCGA files are intentionally **not committed to Git**.

## Source

This project uses the TCGA Breast Cancer (BRCA) cohort distributed through the
UCSC Xena TCGA hub:

- gene-expression matrix: `TCGA.BRCA.sampleMap/HiSeqV2`
- clinical matrix: `TCGA.BRCA.sampleMap/BRCA_clinicalMatrix`
- subtype field: `PAM50Call_RNAseq`

Download them reproducibly with:

```bash
python scripts/download_tcga_brca.py
```

Then optionally cache the aligned matrices with:

```bash
python scripts/prepare_data.py
```

The processed cache contains two matrices:

- `expression_ml_nonpam50.pkl.gz` — PAM50 signature genes removed; used for prediction.
- `expression_biology_full.pkl.gz` — complete expression matrix; used for exploratory biology.

This separation prevents circular PAM50 prediction without unnecessarily
discarding biologically relevant genes from differential-expression analysis.

## Why the raw data are not in this repository

The expression matrix is large and is already publicly distributed by UCSC
Xena. Keeping it out of Git makes the repository lightweight and reproducible.
