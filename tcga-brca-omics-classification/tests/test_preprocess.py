import pandas as pd

from tcga_brca.preprocess import sample_type_code, primary_tumor_mask

def test_sample_type_parser():
    ids = pd.Index([
        "TCGA-AA-BBBB-01A-01R-1111-01",
        "TCGA-AA-CCCC-11A-01R-1111-01",
    ])
    assert list(sample_type_code(ids)) == ["01", "11"]
    assert list(primary_tumor_mask(ids)) == [True, False]
