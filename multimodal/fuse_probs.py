from typing import List
import numpy as np

# -------------------------
# Fusion (mean/sum/weighted)
# -------------------------
def fuse_probs(
    probs_list: List[np.ndarray],
    weights: List[float],
    fusion: str,
) -> np.ndarray:
    if len(probs_list) == 0:
        raise RuntimeError("No probs to fuse (check --modalities and missing files).")

    stacked = np.stack(probs_list, axis=0).astype(np.float32)  # [M, C]

    if fusion == "mean":
        out = np.mean(stacked, axis=0)

    elif fusion == "sum":
        out = np.sum(stacked, axis=0)

    elif fusion == "weighted":
        w = np.asarray(weights, dtype=np.float32)
        denom = float(np.sum(w))
        denom = max(denom, 1e-6)
        out = np.sum(stacked * w[:, None], axis=0) / denom

    else:
        raise ValueError(f"Unknown fusion mode: {fusion}")

    # probs should remain valid
    return np.clip(out, 0.0, 1.0)

