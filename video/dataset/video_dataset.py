import json, collections
import numpy as np
import torch
from torch.utils.data import Dataset
import random

# -------------------- Datasets --------------------


class TrainOneRandomClipPerEpoch(Dataset):
    def __init__(self, index_json: str, mean: torch.Tensor = None, std: torch.Tensor = None,
                 seed: int = 42, exclude_ids=None):
        rows = json.load(open(index_json, "r"))
        if not rows:
            raise RuntimeError(f"No rows in {index_json}")
        self.by_vid = collections.defaultdict(list)
        for r in rows:
            vid = str(r["video_id"])
            # αν έχουμε λίστα exclude_ids, αγνόησε αυτά τα videos
            if exclude_ids is not None and vid in exclude_ids:
                continue
            self.by_vid[vid].append(r)
        self.video_ids = sorted(self.by_vid.keys())
        if not self.video_ids:
            raise RuntimeError(f"No videos left after exclusion in {index_json}")

        self.mean = mean
        self.std = std
        self.seed_base = int(seed)
        self.epoch = 0

        # probe dims
        probe = np.load(self.by_vid[self.video_ids[0]][0]["blob"])
        self.emb_dim = int(probe["feat"].shape[0])
        self.C = len(self.by_vid[self.video_ids[0]][0]["labels"])

        self._sampled_rows = None
        self.set_epoch(0)


    def set_epoch(self, ep: int):
        self.epoch = int(ep)
        rng = random.Random(self.seed_base + self.epoch)
        sampled = []
        for vid in self.video_ids:
            lst = self.by_vid[vid]
            r = lst[rng.randrange(len(lst))]
            sampled.append((vid, r))
        self._sampled_rows = sampled

    def __len__(self): 
        return len(self.video_ids)

    def _standardize(self, x):
        if self.mean is None or self.std is None:
            return x
        mean = self.mean.to(device=x.device, dtype=x.dtype)
        std  = self.std.to(device=x.device, dtype=x.dtype)
        return (x - mean) / (std + 1e-6)


    def __getitem__(self, i):
        vid, r = self._sampled_rows[i]
        z = np.load(r["blob"])
        x = torch.from_numpy(z["feat"].astype(np.float32))
        x = self._standardize(x)
        y = torch.from_numpy(np.asarray(r["labels"], np.float32))
        return x, y, vid


class Video_Val_Test_Dataset(Dataset):
    def __init__(self, index_json: str, mean: torch.Tensor = None, std: torch.Tensor = None,
                 exclude_ids=None):
        rows = json.load(open(index_json, "r"))
        if not rows:
            raise RuntimeError(f"No rows in {index_json}")
        self.by_vid = collections.defaultdict(list)
        for r in rows:
            vid = str(r["video_id"])
            if exclude_ids is not None and vid in exclude_ids:
                continue
            self.by_vid[vid].append(r)

        for vid in self.by_vid:
            self.by_vid[vid] = sorted(
                self.by_vid[vid],
                key=lambda x: x.get("start_sec", 0.0)
            )

        self.videos = sorted(self.by_vid.keys())
        if not self.videos:
            raise RuntimeError(f"No videos left after exclusion in {index_json}")

        # probe
        p0 = np.load(self.by_vid[self.videos[0]][0]["blob"])
        self.emb_dim = int(p0["feat"].shape[0])
        self.C = len(self.by_vid[self.videos[0]][0]["labels"])

        self.mean = mean
        self.std = std


    def __len__(self): return len(self.videos)

    def _standardize(self, x):
        if self.mean is None or self.std is None:
            return x
        mean = self.mean.to(device=x.device, dtype=x.dtype)
        std  = self.std.to(device=x.device, dtype=x.dtype)
        return (x - mean) / (std + 1e-6)

    def __getitem__(self, i):
        vid = self.videos[i]
        items = self.by_vid[vid]
        Xs = []
        for r in items:
            z = np.load(r["blob"])
            x = torch.from_numpy(z["feat"].astype(np.float32))
            x = self._standardize(x)
            Xs.append(x)
        X = torch.stack(Xs, 0)  # [Nv, D]
        y = torch.from_numpy(np.asarray(items[0]["labels"], np.float32))
        return X, y, vid

