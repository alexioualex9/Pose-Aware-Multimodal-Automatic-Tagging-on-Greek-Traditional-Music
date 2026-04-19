import json
import collections
from collections import defaultdict
import numpy as np
import torch
from torch.utils.data import Dataset

from utils import load_embed_stats

# ------------------------ Dataset --------------------------

class SkeletonClipsDatasetCTv(Dataset):
    """
    Loads clip-level .npz with:
      data: [C,T,V]
    """
    def __init__(self, index_json: str, min_valid: int = 5):
        rows = json.load(open(index_json, "r"))
        self.rows = [r for r in rows if int(r.get("valid_len", 0)) >= min_valid]
        if not self.rows:
            raise RuntimeError(f"No valid samples in {index_json} after min_valid={min_valid}")
        self.index_json = index_json

        # sanity check first item
        z = np.load(self.rows[0]["blob"])
        data = z["data"]
        if data.ndim != 3:
            raise RuntimeError(f"Expected data [C,T,V], got {data.shape}")
        if data.shape[2] != 17:
            raise RuntimeError(f"Expected V=17, got V={data.shape[2]}")

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, idx):
        r = self.rows[idx]
        z = np.load(r["blob"])
        x = torch.from_numpy(z["data"].astype(np.float32))  # [C,T,V]
        y = torch.from_numpy(np.asarray(r["labels"], dtype=np.float32))
        vid = r["video_id"]
        return x, y, vid


class SkelAllClipsPerVideoCTv(Dataset):
    """
    Groups clips by video_id; returns stacked clips:
      X: [Nv,C,T,V], y: [C_classes]
    """
    def __init__(self, index_json: str, min_valid: int = 5):
        rows = json.load(open(index_json, "r"))
        by_vid = collections.defaultdict(list)
        for r in rows:
            if int(r.get("valid_len", 0)) >= min_valid:
                by_vid[r["video_id"]].append(r)

        self.groups = [
            (vid, sorted(items, key=lambda x: x.get("start_sec", 0.0)))
            for vid, items in by_vid.items() if items
        ]
        if not self.groups:
            raise RuntimeError(f"No valid video groups in {index_json}")

        p0 = np.load(self.groups[0][1][0]["blob"])
        data0 = p0["data"]
        if data0.ndim != 3:
            raise RuntimeError(f"Expected data [C,T,V], got {data0.shape}")
        self.C_in = data0.shape[0]
        self.T = data0.shape[1]
        self.V = data0.shape[2]
        self.C_out = len(self.groups[0][1][0]["labels"])
        if self.V != 17:
            raise RuntimeError(f"Expected V=17, got V={self.V}")

    def __len__(self):
        return len(self.groups)

    def __getitem__(self, i):
        vid, items = self.groups[i]
        Xs = []
        for r in items:
            z = np.load(r["blob"])
            data = z["data"].astype(np.float32)
            Xs.append(torch.from_numpy(data))
        X = torch.stack(Xs, dim=0)  # [Nv,C,T,V]
        y = torch.from_numpy(np.asarray(items[0]["labels"], dtype=np.float32))
        return X, y, vid