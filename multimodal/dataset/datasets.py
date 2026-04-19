import json
import numpy as np
import torch
import collections
from pathlib import Path
from typing import Dict
from torch.utils.data import Dataset

from dataset.skeleton_emb import SkeletonVideoStore


def _row_to_video_blob_path(row: dict, blobs_dir: Path) -> Path:
    if "blob" in row and row["blob"]:
        return Path(row["blob"])
    vid = str(row["video_id"])
    s_ms = int(float(row["start_sec"]) * 1000.0)
    e_ms = int(float(row["end_sec"]) * 1000.0)
    return blobs_dir / f"{vid}_s{s_ms}_e{e_ms}.npz"


# ------------------------------------------------------------
# Dataset builder
# ------------------------------------------------------------
def build_ds(idx, a_dir, s_dir, model_name, exclude_ids, split, mean_t, std_t):


        if idx is None:
            raise SystemExit(f"Missing {split} --*_index.")

        needs_s = model_name in [
            "seq_transformer_avs_masked",
            "seq_transformer_as_masked",
            "seq_transformer_vs_masked",
        ]

        if needs_s and (s_dir is None):
            raise SystemExit(f"Model {model_name} requires --{split}_skel_emb_dir (per-video <vid>.npz).")

        # AVS (needs audio+video+skeleton)
        if model_name in ["seq_transformer_avs_masked", "late_avs"]:
            if a_dir is None:
                raise SystemExit(f"Model {model_name} requires --{split}_audio_emb_dir")
            return AVSEmbeddingsDataset(
                index_json=idx,
                audio_emb_dir=a_dir,
                skel_emb_dir=s_dir,
                exclude_ids=exclude_ids,
                mean_v=mean_t,
                std_v=std_t,
            )

        # AS (audio+skeleton)
        if model_name in ["seq_transformer_as_masked", "late_as"]:
            if a_dir is None:
                raise SystemExit(f"Model {model_name} requires --{split}_audio_emb_dir")

            return ASEmbeddingsDataset(
                index_json=idx,
                audio_emb_dir=a_dir,
                skel_emb_dir=s_dir,
                exclude_ids=exclude_ids,
            )

        # VS (video+skeleton)
        if model_name in ["seq_transformer_vs_masked", "late_vs"]:
            return VSEmbeddingsDataset(
                index_json=idx,
                skel_emb_dir=s_dir,
                exclude_ids=exclude_ids,
                mean_v=mean_t,
                std_v=std_t,
            )

        # AV only
        if a_dir is None:
            raise SystemExit(f"Model {model_name} requires --{split}_audio_emb_dir")
        return AVEmbeddingsDataset(
            index_json=idx,
            audio_emb_dir=a_dir,
            exclude_ids=exclude_ids,
            mean_v=mean_t,
            std_v=std_t,
        )


# ============================================================
# Dataset: Audio + Video embeddings per video (base)
# ============================================================
class AVEmbeddingsDataset(Dataset):
    """
    Returns:
      emb_a: [T, D_a]
      emb_v: [T, D_v]
      y:     [C]
      vid:   str
    """

    def __init__(
        self,
        index_json: str,
        audio_emb_dir: str,
        exclude_ids=None,
        mean_v: torch.Tensor = None,
        std_v: torch.Tensor = None,
    ):
        self.index_json = index_json
        self.audio_emb_dir = Path(audio_emb_dir)
        self.mean_v = mean_v
        self.std_v = std_v

        rows = json.load(open(index_json, "r"))
        if not rows:
            raise RuntimeError(f"No rows in {index_json}")

        index_path = Path(index_json)
        blobs_dir = index_path.parent / "blobs"

        by_vid = collections.defaultdict(list)
        for r in rows:
            vid = str(r["video_id"])
            if exclude_ids is not None and vid in exclude_ids:
                continue
            r = dict(r)
            r["blob"] = str(_row_to_video_blob_path(r, blobs_dir))
            by_vid[vid].append(r)

        self.by_vid = {vid: sorted(lst, key=lambda x: float(x["start_sec"]))
                       for vid, lst in by_vid.items() if lst}
        self.videos = sorted(self.by_vid.keys())
        if not self.videos:
            raise RuntimeError(f"No videos left in {index_json} after exclusion filtering.")

        # infer dims
        probe_vid = self.videos[0]
        a_path = self.audio_emb_dir / f"{probe_vid}.npz"
        if not a_path.exists():
            raise RuntimeError(f"Missing audio embedding file: {a_path}")
        za = np.load(a_path)
        if "video_level_embs" not in za:
            raise RuntimeError(f"{a_path} missing key 'audio_clip_emb'")
        self.emb_dim_a = int(za["video_level_embs"].shape[-1])

        p0 = Path(self.by_vid[probe_vid][0]["blob"])
        if not p0.exists():
            raise RuntimeError(f"Missing video blob: {p0}")
        zv = np.load(p0)
        if "feat" not in zv:
            raise RuntimeError(f"{p0} missing key 'feat'")
        self.emb_dim_v = int(np.asarray(zv["feat"]).reshape(-1).shape[0])

        # labels dim
        lab = self.by_vid[probe_vid][0].get("labels", None)
        if lab is None:
             raise RuntimeError("Cannot infer label dim.")
        self.num_labels = int(len(lab))

        print(f"[info] AVEmbeddingsDataset {index_json}: D_a={self.emb_dim_a} D_v={self.emb_dim_v} C={self.num_labels}")

    def __len__(self):
        return len(self.videos)

    def __getitem__(self, i: int):
        vid = self.videos[i]
        items = self.by_vid[vid]
        T = len(items)

        # load audio
        a_path = self.audio_emb_dir / f"{vid}.npz"
        za = np.load(a_path)
        emb_a = np.asarray(za["video_level_embs"], dtype=np.float32)
        if emb_a.ndim != 2:
            raise RuntimeError(f"audio_clip_emb must be [T,Da], got {emb_a.shape} for {a_path}")

        # labels
        labels_np = np.asarray(items[0]["labels"], dtype=np.float32).reshape(-1)

        # Keep the same amount of embeddings for each
        emb_v = torch.zeros((T, self.emb_dim_v), dtype=torch.float32)
        Ta = emb_a.shape[0]   # audio timesteps
        Tv = emb_v.shape[0]     # video timesteps (== T)
        Tmin = min(Ta, Tv)

        if Ta != Tv:
           emb_a = emb_a[:Tmin]
           emb_v   = emb_v[:Tmin]

        # load video clips
        Xv_list = []
        for r in items[:Tmin]:
            p = Path(r["blob"])
            z = np.load(p)
            feat = np.asarray(z["feat"], dtype=np.float32).reshape(-1)
            Xv_list.append(torch.from_numpy(feat))
        emb_v = torch.stack(Xv_list, dim=0)  # [T,Dv]

        # optional standardize video
        if self.mean_v is not None and self.std_v is not None:
            mean_v = self.mean_v
            std_v = self.std_v
            eps = 1e-6
            if mean_v.device != emb_v.device:
                mean_v = mean_v.to(emb_v.device)
                std_v = std_v.to(emb_v.device)
            emb_v = (emb_v - mean_v) / (std_v + eps)

        y = torch.from_numpy(labels_np)
        return torch.from_numpy(emb_a), emb_v, y, vid, Tmin


class ASEmbeddingsDataset(Dataset):
    """
    Returns:
      emb_a: [T, D_a]
      emb_s: [T, D_s]
      mask_s:[T]
      y:     [C]
      vid:   str

    Video-level exclusion:
      keep only videos that have skel_emb_dir/<vid>.npz with N_s >= 1.
    """

    def __init__(self, index_json: str, audio_emb_dir: str, skel_emb_dir: str, exclude_ids=None):
        self.index_json = index_json
        self.audio_emb_dir = Path(audio_emb_dir)
        self.skel_store = SkeletonVideoStore(skel_emb_dir)
        """
        print("[debug] index_json:", index_json)
        print("[debug] audio_emb_dir:", audio_emb_dir)
        print("[debug] skel_emb_dir:", skel_emb_dir)
        """

        rows = json.load(open(index_json, "r"))
        if not rows:
            raise RuntimeError(f"No rows in {index_json}")

        by_vid = collections.defaultdict(list)
        for r in rows:
            vid = str(r["video_id"])
            if exclude_ids is not None and vid in exclude_ids:
                continue
            by_vid[vid].append(r)

        by_vid = {vid: sorted(lst, key=lambda x: float(x["start_sec"])) for vid, lst in by_vid.items() if lst}

        # filter videos with no skeletons at all
        kept = {}
        for vid, clips in by_vid.items():
            info = self.skel_store.load_video(vid)
            if info is None or int(info["N_s"]) <= 0:
                continue
            kept[vid] = clips

        self.by_vid = kept
        self.videos = sorted(self.by_vid.keys())
        print(f"[info] Skeleton video-level filtering (AS): {len(by_vid)} -> {len(self.by_vid)} videos kept")
        if not self.videos:
            raise RuntimeError("No videos left after skeleton video-level filtering (AS).")

        # infer dims
        probe_vid = self.videos[0]
        a_path = self.audio_emb_dir / f"{probe_vid}.npz"
        if not a_path.exists():
            raise RuntimeError(f"Missing audio embedding file: {a_path}")
        za = np.load(a_path)
        if "video_level_embs" not in za:
            raise RuntimeError(f"{a_path} missing key 'video_level_embs'")
        self.emb_dim_a = int(za["video_level_embs"].shape[-1])

        #if "labels" in za:
        #    self.num_labels = int(np.asarray(za["labels"]).shape[-1])
        #else:
        lab = self.by_vid[probe_vid][0].get("labels", None)
        if lab is None:
             raise RuntimeError("Cannot infer label dim for AS dataset.")
        self.num_labels = int(len(lab))

        info0 = self.skel_store.load_video(probe_vid)
        self.emb_dim_s = int(info0["D_s"])

        print(f"[info] ASEmbeddingsDataset {index_json}: D_a={self.emb_dim_a} D_s={self.emb_dim_s} C={self.num_labels}")

    def __len__(self):
        return len(self.videos)

    def __getitem__(self, i: int):
        vid = self.videos[i]
        items = self.by_vid[vid]
        T = len(items)

        a_path = self.audio_emb_dir / f"{vid}.npz"
        za = np.load(a_path)
        emb_a = np.asarray(za["video_level_embs"], dtype=np.float32)
        if emb_a.ndim != 2:
            raise RuntimeError(f"video_level_embs must be [T,Da], got {emb_a.shape} for {a_path}")

        # labels
        labels_np = np.asarray(items[0]["labels"], dtype=np.float32).reshape(-1)

        emb_s = torch.zeros((T, self.emb_dim_s), dtype=torch.float32)
        mask_s = torch.zeros((T,), dtype=torch.float32)

        # Keep the same amount of embeddings for each
        Ta = emb_a.shape[0]   # audio timesteps
        Ts = emb_s.shape[0]   # skeleton timesteps (== T)
        Tmin = min(Ta, Ts)

        if Ta != Ts:
           emb_a = emb_a[:Tmin]
           emb_s   = emb_s[:Tmin]
           mask_s = mask_s[:Tmin]

        for t, r in enumerate(items[:Tmin]):
            vec = self.skel_store.get_clip_emb(vid, float(r["start_sec"]), float(r["end_sec"]))
            if vec is None:
                continue
            vec = np.asarray(vec, dtype=np.float32).reshape(-1)
            emb_s[t] = torch.from_numpy(vec)
            mask_s[t] = 1.0

        return torch.from_numpy(emb_a), emb_s, mask_s, torch.from_numpy(labels_np), vid, Tmin



class VSEmbeddingsDataset(Dataset):
    """
    Returns:
      emb_v: [T, D_v]
      emb_s: [T, D_s]
      mask_s:[T]
      y:     [C]
      vid:   str

    Video-level exclusion:
      keep only videos that have skel_emb_dir/<vid>.npz with N_s >= 1.
    """

    def __init__(
        self,
        index_json: str,
        skel_emb_dir: str,
        exclude_ids=None,
        mean_v: torch.Tensor = None,
        std_v: torch.Tensor = None,
    ):
        self.index_json = index_json
        self.mean_v = mean_v
        self.std_v = std_v
        self.skel_store = SkeletonVideoStore(skel_emb_dir)

        rows = json.load(open(index_json, "r"))
        if not rows:
            raise RuntimeError(f"No rows in {index_json}")

        index_path = Path(index_json)
        blobs_dir = index_path.parent / "blobs"

        by_vid = collections.defaultdict(list)
        for r in rows:
            vid = str(r["video_id"])
            if exclude_ids is not None and vid in exclude_ids:
                continue
            r = dict(r)
            r["blob"] = str(_row_to_video_blob_path(r, blobs_dir))
            by_vid[vid].append(r)

        by_vid = {vid: sorted(lst, key=lambda x: float(x["start_sec"])) for vid, lst in by_vid.items() if lst}

        # filter videos with no skeletons at all
        kept = {}
        for vid, clips in by_vid.items():
            info = self.skel_store.load_video(vid)
            if info is None or int(info["N_s"]) <= 0:
                continue
            kept[vid] = clips

        self.by_vid = kept
        self.videos = sorted(self.by_vid.keys())
        print(f"[info] Skeleton video-level filtering (VS): {len(by_vid)} -> {len(self.by_vid)} videos kept")
        if not self.videos:
            raise RuntimeError("No videos left after skeleton video-level filtering (VS).")

        # infer dims
        probe_vid = self.videos[0]
        p0 = Path(self.by_vid[probe_vid][0]["blob"])
        z0 = np.load(p0)
        if "feat" not in z0:
            raise RuntimeError(f"{p0} missing key 'feat'")
        self.emb_dim_v = int(np.asarray(z0["feat"]).reshape(-1).shape[0])

        lab = self.by_vid[probe_vid][0].get("labels", None)
        if lab is None:
            raise RuntimeError("Cannot infer label dim for VS dataset: missing 'labels' in index rows.")
        self.num_labels = int(len(lab))

        info0 = self.skel_store.load_video(probe_vid)
        self.emb_dim_s = int(info0["D_s"])

        print(f"[info] VSEmbeddingsDataset {index_json}: D_v={self.emb_dim_v} D_s={self.emb_dim_s} C={self.num_labels}")

    def __len__(self):
        return len(self.videos)

    def __getitem__(self, i: int):
        vid = self.videos[i]
        items = self.by_vid[vid]
        T = len(items)

        # labels
        labels_np = np.asarray(items[0]["labels"], dtype=np.float32).reshape(-1)

        Xv_list = []
        for r in items:
            p = Path(r["blob"])
            z = np.load(p)
            feat = np.asarray(z["feat"], dtype=np.float32).reshape(-1)
            Xv_list.append(torch.from_numpy(feat))
        emb_v = torch.stack(Xv_list, dim=0)

        if self.mean_v is not None and self.std_v is not None:
            mean_v = self.mean_v
            std_v = self.std_v
            if mean_v.device != emb_v.device:
                mean_v = mean_v.to(emb_v.device)
                std_v = std_v.to(emb_v.device)
            emb_v = (emb_v - mean_v) / std_v

        emb_s = torch.zeros((T, self.emb_dim_s), dtype=torch.float32)
        mask_s = torch.zeros((T,), dtype=torch.float32)


        # strict alignment with index timeline
        Tv = emb_v.shape[0]   # video timesteps
        Ts = emb_s.shape[0]   # skeleton timesteps (== T)
        Tmin = min(Tv, Ts)

        if Tv != Ts:
           emb_v = emb_v[:Tmin]
           emb_s   = emb_s[:Tmin]
           mask_s = mask_s[:Tmin]

        for t, r in enumerate(items[:Tmin]):
            vec = self.skel_store.get_clip_emb(vid, float(r["start_sec"]), float(r["end_sec"]))
            if vec is None:
                continue
            vec = np.asarray(vec, dtype=np.float32).reshape(-1)
            emb_s[t] = torch.from_numpy(vec)
            mask_s[t] = 1.0

        return emb_v, emb_s, mask_s, torch.from_numpy(labels_np), vid, Tmin



class AVSEmbeddingsDataset(AVEmbeddingsDataset):
    """
    Extends AVEmbeddingsDataset to also return:
      emb_s:  [T, D_s]
      mask_s: [T]  (1 if skeleton exists else 0)

    Video-level exclusion rule (for skeleton-needed models):
      keep only videos that have skel_emb_dir/<vid>.npz with N_s >= 1.
    """

    def __init__(
        self,
        index_json: str,
        audio_emb_dir: str,
        skel_emb_dir: str,
        exclude_ids=None,
        mean_v: torch.Tensor = None,
        std_v: torch.Tensor = None,
    ):
        super().__init__(
            index_json=index_json,
            audio_emb_dir=audio_emb_dir,
            exclude_ids=exclude_ids,
            mean_v=mean_v,
            std_v=std_v,
        )

        self.skel_store = SkeletonVideoStore(skel_emb_dir)

        # keep only videos with at least 1 skeleton clip in per-video file
        kept = {}
        for vid, clips in self.by_vid.items():
            info = self.skel_store.load_video(vid)
            if info is None:
                continue
            if int(info["N_s"]) <= 0:
                continue
            kept[vid] = clips

        before = len(self.by_vid)
        self.by_vid = kept
        self.videos = sorted(self.by_vid.keys())
        after = len(self.by_vid)
        print(f"[info] Skeleton video-level filtering (AVS): {before} -> {after} videos kept")
        if not self.videos:
            raise RuntimeError("No videos left after skeleton video-level filtering (AVS).")

        # infer D_s from first kept video
        info0 = self.skel_store.load_video(self.videos[0])
        self.emb_dim_s = int(info0["D_s"])
        print(f"[info] AVSEmbeddingsDataset {index_json}: D_s={self.emb_dim_s}")

    def __getitem__(self, i: int):
        emb_a, emb_v, y, vid, _ = super().__getitem__(i)
        items = self.by_vid[vid]
        T = len(items)

        emb_s = torch.zeros((T, self.emb_dim_s), dtype=torch.float32)
        mask_s = torch.zeros((T,), dtype=torch.float32)


        # strict alignment with index timeline
        Ta = emb_a.shape[0]   # video timesteps
        Tv = emb_v.shape[0]   # video timesteps
        Ts = emb_s.shape[0]   # skeleton timesteps (== T)
        Tmin = min(Ta, Tv)
        Tmin = min(Tmin, Ts)

        if Ta != Tv or Tv != Ts or Ta != Ts:
           emb_a = emb_a[:Tmin]
           emb_v = emb_v[:Tmin]
           emb_s   = emb_s[:Tmin]
           mask_s = mask_s[:Tmin]

        for t, r in enumerate(items[:Tmin]):
            s_sec = float(r["start_sec"])
            e_sec = float(r["end_sec"])
            vec = self.skel_store.get_clip_emb(vid, s_sec, e_sec)
            if vec is None:
                continue
            vec = np.asarray(vec, dtype=np.float32).reshape(-1)
            if vec.shape[0] != self.emb_dim_s:
                raise RuntimeError(f"Skeleton dim mismatch for vid={vid}: got {vec.shape[0]} expected {self.emb_dim_s}")
            emb_s[t] = torch.from_numpy(vec)
            mask_s[t] = 1.0

        return emb_a, emb_v, emb_s, mask_s, y, vid, Tmin
