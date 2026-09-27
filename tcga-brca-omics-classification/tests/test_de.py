import numpy as np
import pandas as pd
from tcga_brca.differential_expression import welch_de

def test_de_handles_nan_pvalues():
    rng=np.random.default_rng(0)
    X=pd.DataFrame(rng.normal(size=(20,4)),columns=list("ABCD"))
    X["D"]=1.0
    y=pd.Series(["Basal"]*10+["LumA"]*10,index=X.index)
    X.loc[:9,"A"] += 2.0
    out=welch_de(X,y)
    assert "p_adj" in out.columns
    assert out.loc[out["gene"]=="A","p_adj"].notna().all()
