from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/"src"))

from tcga_brca.preprocess import save_processed

meta = save_processed(
    ROOT/"data/raw/HiSeqV2.gz",
    ROOT/"data/raw/BRCA_clinicalMatrix",
    ROOT/"data/processed",
)
print(meta)
