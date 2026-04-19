import numpy as np
from typing import Dict


def per_label_confusions(Y: np.ndarray, P: np.ndarray) -> Dict[str, np.ndarray]:
    """
    Y, P: [N, C] in {0,1}
    Returns dict of [C] arrays: TP, FP, TN, FN, support
    """
    Yb = Y.astype(bool)
    Pb = P.astype(bool)

    TP = np.sum(Yb & Pb, axis=0).astype(np.int64)
    FP = np.sum((~Yb) & Pb, axis=0).astype(np.int64)
    FN = np.sum(Yb & (~Pb), axis=0).astype(np.int64)
    TN = np.sum((~Yb) & (~Pb), axis=0).astype(np.int64)
    support = np.sum(Yb, axis=0).astype(np.int64)
    return dict(TP=TP, FP=FP, TN=TN, FN=FN, support=support)


def per_label_metrics_from_conf(conf: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
    TP, FP, TN, FN, support = conf["TP"], conf["FP"], conf["TN"], conf["FN"], conf["support"]

    prec = np.zeros_like(TP, dtype=np.float32)
    rec  = np.zeros_like(TP, dtype=np.float32)
    f1   = np.zeros_like(TP, dtype=np.float32)

    pp = TP + FP
    gt = support

    m = pp > 0
    prec[m] = TP[m] / pp[m]

    m = gt > 0
    rec[m] = TP[m] / gt[m]

    denom = (2 * TP + FP + FN)
    m = denom > 0
    f1[m] = (2 * TP[m]) / denom[m]

    return dict(precision=prec, recall=rec, f1=f1)