from sklearn.metrics import (
    classification_report,
    roc_auc_score,
    average_precision_score,
    f1_score,
)

import numpy as np
from typing import Dict, List


def safe_auc(fn, Y, S, average: str):
    try:
        return fn(Y, S, average=average)
    except Exception:
        return float("nan")


def compute_global_metrics(Y: np.ndarray, S: np.ndarray, threshold: float, label_names: List[str]) -> Dict[str, object]:
    P = (S >= threshold).astype(np.int32)

    roc_micro = safe_auc(roc_auc_score, Y, S, average="micro")
    roc_macro = safe_auc(roc_auc_score, Y, S, average="macro")
    pr_micro  = safe_auc(average_precision_score, Y, S, average="micro")
    pr_macro  = safe_auc(average_precision_score, Y, S, average="macro")

    f1_micro = f1_score(Y, P, average="micro", zero_division=0)
    f1_macro = f1_score(Y, P, average="macro", zero_division=0)

    report = classification_report(Y, P, target_names=label_names, zero_division=0, digits=2)

    return dict(
        report=report,
        roc_micro=roc_micro, roc_macro=roc_macro,
        pr_micro=pr_micro, pr_macro=pr_macro,
        f1_micro=f1_micro, f1_macro=f1_macro,
    )