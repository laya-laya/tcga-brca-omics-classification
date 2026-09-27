## Quick start

### Option A — existing Python environment

The core workflow supports Python 3.8+.

```bash
python -m pip install -r requirements.txt
python scripts/download_tcga_brca.py
python scripts/run_nested_cv.py
python scripts/run_full_analysis.py
```

For pathway enrichment:

```bash
python -m pip install gprofiler-official
python scripts/run_enrichment.py
```

### Option B — isolated Conda environment

```bash
conda env create -f environment.yml
conda activate tcga-brca-ml
```

Then:

```bash
make download
make cv
make analysis
make heatmap
```

The scripts add `src/` to `sys.path`, so an editable package installation is
not required to run the analysis.


## Data

The repository does **not** commit TCGA data. The downloader retrieves:

- UCSC Xena `TCGA.BRCA.sampleMap/HiSeqV2`
- UCSC Xena `TCGA.BRCA.sampleMap/BRCA_clinicalMatrix`

See [`data/README.md`](data/README.md) for details.





## Results

### Model selection

Nested cross-validation compared Elastic-Net logistic regression with a
Random Forest classifier on the training partition.

| Model | Balanced accuracy | Macro F1 | Macro ROC-AUC |
|---|---:|---:|---:|
| Elastic Net | 0.901 ± 0.041 | 0.888 ± 0.042 | 0.987 ± 0.005 |
| Random Forest | 0.870 ± 0.049 | 0.883 ± 0.050 | 0.981 ± 0.012 |

Elastic Net was therefore selected for final training and held-out evaluation.

### Held-out TCGA-BRCA performance

The final Elastic-Net model was evaluated on the untouched 20% test partition.

| Metric | Estimate |
|---|---:|
| Balanced accuracy | 0.885 |
| Macro F1 | 0.876 |
| Weighted F1 | 0.876 |
| Macro one-vs-rest ROC-AUC | 0.979 |

Bootstrap 95% intervals:

| Metric | 95% interval |
|---|---:|
| Balanced accuracy | 0.818–0.940 |
| Macro F1 | 0.810–0.925 |
| Weighted F1 | 0.825–0.923 |
| Macro ROC-AUC | 0.966–0.990 |

Importantly, PAM50-defining genes were excluded from the predictive feature
matrix. The results therefore show that broader transcriptomic patterns retain
substantial information about PAM50 subtype structure.

### Classification errors

The main classification ambiguity occurred between Luminal A and Luminal B,
while Basal tumors were completely separated in this particular held-out split.

![Held-out confusion matrix](figures/confusion_matrix.png)

### ROC analysis

One-vs-rest ROC-AUC values on the held-out test set were:

- Basal: 1.000
- HER2-enriched: 0.989
- Luminal A: 0.975
- Luminal B: 0.951

![One-vs-rest ROC curves](figures/roc_ovr.png)

### Basal vs Luminal A transcriptomic differences

Differential-expression analysis was performed separately from the prediction
pipeline using the full transcriptome. Genes were tested using Welch's
t-test and corrected for multiple testing using the Benjamini-Hochberg
procedure.

Positive expression differences indicate higher expression in Basal tumors;
negative differences indicate higher expression in Luminal A tumors.

![Basal vs LumA volcano plot](figures/volcano_basal_vs_luma.png)

### Feature-selection stability

Feature selection was repeated across cross-validation folds to determine
which transcripts were consistently retained by the predictive pipeline.

Several genes were selected in all five stability folds, including AR, NCAPG,
NCAPH, HMGA1, NEIL3, CDCA8, CDCA5, FOXM1, CCNA2, XBP1, SPDEF and SERPINA11.

These genes should be interpreted as reproducible predictive features rather
than causal subtype drivers.

![Top-gene expression heatmap](figures/top_gene_heatmap.png)