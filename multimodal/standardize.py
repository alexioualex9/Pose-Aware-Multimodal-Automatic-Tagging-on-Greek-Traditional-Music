import json
import numpy as np
import torch
from pathlib import Path

from dataset.datasets import _row_to_video_blob_path

# ------------------------------------------------------------
# Standardize video embeddings (based on TRAIN video clips)
# ------------------------------------------------------------
def standardize_video_embeddings(train_index, standardize, exclude_ids):

    mean_t = std_t = None
    if standardize:
        if not train_index:
            raise SystemExit("--standardize requires --train_index")

        print("[info] Computing mean/std from ALL TRAIN video clips...")
        rows_tr = json.load(open(train_index, "r"))
        index_path = Path(train_index)
        blobs_dir = index_path.parent / "blobs"

        feats = []
        for r in rows_tr:
            vid = str(r["video_id"])
            if exclude_ids is not None and vid in exclude_ids:
                continue
            blob_path = _row_to_video_blob_path(r, blobs_dir)
            z = np.load(blob_path)
            feats.append(np.asarray(z["feat"], dtype=np.float32).reshape(-1))

        Xtr = np.stack(feats, 0)
        mean_v = Xtr.mean(axis=0).astype(np.float32)
        std_v = Xtr.std(axis=0).astype(np.float32)
        std_v[std_v < 1e-6] = 1e-6
        mean_t = torch.from_numpy(mean_v)
        std_t = torch.from_numpy(std_v)
        print("[info] Standardization enabled.")

    return mean_t, std_t