import numpy as np
import torch
import json

def collect_all_feats(index_json: str, exclude_ids=None) -> np.ndarray:
    rows = json.load(open(index_json, "r"))
    feats = []
    for r in rows:
        vid = str(r["video_id"])
        if exclude_ids is not None and vid in exclude_ids:
            continue
        feats.append(np.load(r["blob"])["feat"].astype(np.float32))
    return np.stack(feats, 0)


def standardization(train_index, standardize, exclude_ids, device):

    mean_t = std_t = None
    if standardize:
        print("\n\n----STANDARDIZE----\n\n")
        Xtr = collect_all_feats(train_index, exclude_ids=exclude_ids)
        mean = Xtr.mean(axis=0).astype(np.float32)
        std  = Xtr.std(axis=0).astype(np.float32)
        std[std < 1e-6] = 1e-6
        mean_t = torch.from_numpy(mean)
        std_t  = torch.from_numpy(std)
    else:
        mean = std = None

    return mean_t, std_t
