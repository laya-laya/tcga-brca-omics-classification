import numpy as np
import pandas as pd
from sklearn.metrics import roc_curve, auc
from sklearn.preprocessing import label_binarize

def one_vs_rest_roc(y_true, proba, classes):
    y_bin = label_binarize(y_true, classes=classes)
    curves = {}
    for i, cls in enumerate(classes):
        fpr, tpr, _ = roc_curve(y_bin[:, i], proba[:, i])
        curves[cls] = {
            "fpr": fpr,
            "tpr": tpr,
            "auc": auc(fpr, tpr),
        }
    return curves
