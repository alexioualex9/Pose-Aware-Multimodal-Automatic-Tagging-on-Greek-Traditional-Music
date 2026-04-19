import numpy as np
from typing import Dict, Tuple


def preds_threshold(probs: np.ndarray, t: float) -> np.ndarray:
    if probs.ndim == 1:
        probs = probs[None, :]
    return probs >= t


def preds_topk(probs: np.ndarray, k: int) -> np.ndarray:
    if probs.ndim == 1:
        probs = probs[None, :]
    N, C = probs.shape
    k = min(k, C)
    idx = np.argpartition(-probs, kth=k - 1, axis=1)[:, :k]
    P = np.zeros((N, C), dtype=bool)
    rows = np.arange(N)[:, None]
    P[rows, idx] = True
    return P



def make_predictions(probs_1d: np.ndarray, policy: str, threshold: float, k: int) -> np.ndarray:
    if policy == "threshold":
        return preds_threshold(probs_1d, threshold)
    if policy == "topk":
        return preds_topk(probs_1d, k)
    raise ValueError("policy must be 'threshold' or 'topk'")



def precision_from_pred(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    # per-video "score" = precision on predicted positives (policy-dependent)
    if y_true.ndim == 1:
        y_true = y_true[None, :]
    if y_pred.ndim == 1:
        y_pred = y_pred[None, :]
    y_true = y_true.astype(bool)
    y_pred = y_pred.astype(bool)

    tp = np.sum(y_true & y_pred, axis=1).astype(np.float32)
    pp = np.sum(y_pred, axis=1).astype(np.float32)
    out = np.zeros_like(tp, dtype=np.float32)
    m = pp > 0
    out[m] = tp[m] / pp[m]
    return float(out[0])



def confusion_counts(y_true: np.ndarray, y_pred: np.ndarray) -> Tuple[int, int, int, int]:
    """
    IMPORTANT: returns (TP, FP, TN, FN) to match exported columns.
    """
    if y_true.ndim == 1:
        y_true = y_true[None, :]
    if y_pred.ndim == 1:
        y_pred = y_pred[None, :]
    y_true = y_true.astype(bool)
    y_pred = y_pred.astype(bool)

    TP = int(np.sum(y_true & y_pred, axis=1)[0])
    FP = int(np.sum((~y_true) & y_pred, axis=1)[0])
    FN = int(np.sum(y_true & (~y_pred), axis=1)[0])
    TN = int(np.sum((~y_true) & (~y_pred), axis=1)[0])
    return TP, FP, TN, FN



def safe_div(num: float, den: float) -> float:
    return float(num / den) if den > 0 else 0.0


def stats_from_counts(tp: int, fp: int, fn: int, gt: int) -> Dict[str, float]:
    pp = tp + fp
    precision = safe_div(tp, pp)
    recall = safe_div(tp, gt)
    f1 = safe_div(2 * tp, (2 * tp + fp + fn))
    return {"pp": float(pp), "precision": float(precision), "recall": float(recall), "f1": float(f1)}