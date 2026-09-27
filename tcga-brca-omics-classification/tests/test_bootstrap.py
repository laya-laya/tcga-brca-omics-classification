import numpy as np
from tcga_brca.bootstrap import bootstrap_classification_metrics

def test_bootstrap_output():
    y=np.array(["A","B","A","B","A","B","A","B"]*8)
    pred=y.copy()
    proba=np.column_stack([
        np.where(y=="A",.9,.1),
        np.where(y=="B",.9,.1),
    ])
    raw,summary=bootstrap_classification_metrics(
        y,pred,proba,["A","B"],n_boot=50,random_state=1
    )
    assert len(summary)==4
    assert (summary["ci_2.5"]<=summary["ci_97.5"]).all()
