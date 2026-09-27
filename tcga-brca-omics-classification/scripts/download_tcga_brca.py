from pathlib import Path
import sys
from urllib.request import urlretrieve

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/"src"))
RAW = ROOT/"data/raw"
RAW.mkdir(parents=True, exist_ok=True)

FILES = {
    "HiSeqV2.gz":
        "https://tcga.xenahubs.net/download/TCGA.BRCA.sampleMap/HiSeqV2.gz",
    "BRCA_clinicalMatrix":
        "https://tcga.xenahubs.net/download/TCGA.BRCA.sampleMap/BRCA_clinicalMatrix",
}

for filename, url in FILES.items():
    target = RAW/filename
    if target.exists() and target.stat().st_size > 0:
        print("Exists:", target)
        continue
    print("Downloading:", url)
    urlretrieve(url, target)
    print("Saved:", target)
