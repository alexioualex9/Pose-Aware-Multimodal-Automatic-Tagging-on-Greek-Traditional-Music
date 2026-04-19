import numpy as np
import torch
import torch.nn as nn
import json
from pathlib import Path
from typing import List, Optional, Tuple


def load_video_ids_from_index(index_json: str) -> List[str]:
    rows = json.load(open(index_json, "r", encoding="utf-8"))
    vids = sorted({str(r["video_id"]) for r in rows})
    return vids


def load_label_names(labels_json_path: str) -> List[str]:
    obj = json.load(open(labels_json_path, "r", encoding="utf-8"))
    if isinstance(obj, list):
        labels = obj
    elif isinstance(obj, dict) and "labels" in obj and isinstance(obj["labels"], list):
        labels = obj["labels"]
    else:
        raise RuntimeError("--labels_json must be a JSON list OR {'labels': [...]}")

    if not labels or not all(isinstance(x, str) for x in labels):
        raise RuntimeError("--labels_json must contain a non-empty list of strings.")
    return labels


def load_ids_txt(path: Optional[str]):
    if path is None:
        return None
    ids = set()
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                ids.add(line)
    print(f"[info] Loaded {len(ids)} ids from {path}")
    return ids


def load_labels(npz_path: Path) -> np.ndarray:
    z = np.load(npz_path, allow_pickle=True)
    if "labels" not in z:
        raise KeyError(f"Missing key 'labels' in {npz_path}. Keys: {list(z.keys())}")
    return np.asarray(z["labels"], dtype=np.int32).reshape(-1)



def load_ckpt_into_model(model: nn.Module, ckpt_path: str, device: str, strict: bool = True):
    ckpt = torch.load(ckpt_path, map_location=device)
    sd = ckpt["fusion_state_dict"] if (isinstance(ckpt, dict) and "fusion_state_dict" in ckpt) else ckpt
    model.load_state_dict(sd, strict=strict)


# Skeleton per-video load + map
def load_skel_video(vid: str, skel_emb_dir):
        p = skel_emb_dir / f"{vid}.npz"
        if not p.exists():
            return None
        z = np.load(p, allow_pickle=False)
        if "clip_emb" not in z or "clip_start" not in z or "clip_end" not in z:
            return None
        clip_emb = np.asarray(z["clip_emb"], dtype=np.float32)
        clip_start = np.asarray(z["clip_start"], dtype=np.float32).reshape(-1)
        clip_end = np.asarray(z["clip_end"], dtype=np.float32).reshape(-1)
        mapping = {}
        for i in range(clip_emb.shape[0]):
            mapping[(int(round(clip_start[i]*1000.0)), int(round(clip_end[i]*1000.0)))] = int(i)
        return dict(clip_emb=clip_emb, map=mapping, Ds=int(clip_emb.shape[1]), Ns=int(clip_emb.shape[0]))




def load_npz_1d(path: Path, key: str) -> np.ndarray:
    z = np.load(path, allow_pickle=True)
    if key not in z:
        raise KeyError(f"Missing key '{key}' in {path}. Keys: {list(z.keys())}")
    arr = z[key]
    arr = np.asarray(arr, dtype=np.float32)
    if arr.ndim != 1:
        arr = arr.reshape(-1)
    return arr



def load_modal_probs_for_vid(
    vid: str,
    a_dir: Optional[Path],
    v_dir: Optional[Path],
    s_dir: Optional[Path],
    modalities: List[str],
) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], Optional[np.ndarray], Optional[np.ndarray], bool]:

    need_a = "a" in modalities
    need_v = "v" in modalities
    need_s = "s" in modalities

    require_a = need_a
    require_v = need_v

    a_path = (a_dir / f"{vid}.npz") if a_dir is not None else None
    v_path = (v_dir / f"{vid}.npz") if v_dir is not None else None
    s_path = (s_dir / f"{vid}.npz") if s_dir is not None else None

    has_a = (a_path is not None and a_path.exists())
    has_v = (v_path is not None and v_path.exists())
    has_s = (s_path is not None and s_path.exists())

    # --- required checks  ---
    if need_a and not has_a:
       raise FileNotFoundError(f"Missing audio probs file: {a_path}")

    if need_v and not has_v:
       raise FileNotFoundError(f"Missing video probs file: {v_path}")

    if need_s and not has_s:
       has_s = False
       ps = None

    # --- pick labels source (first available among requested modalities) ---
    y = None
    #if need_a and has_a:
    #    y = load_labels(a_path)
    if need_v and has_v:
        y = load_labels(v_path)
    elif need_s and has_s:
        y = load_labels(s_path)
    else:
        return None, None, None, None, False

    C = int(y.shape[0])

    pa = pv = ps = None

    if need_a:
        pa = load_npz_1d(a_path, "audio_probs")       # probs
        if pa.shape != (C,):
            raise RuntimeError(f"audio_probs shape mismatch for vid={vid}: got {pa.shape}, expected {(C,)}")

    if need_v:
        pv = load_npz_1d(v_path, "video_probs")   # probs
        if pv.shape != (C,):
            raise RuntimeError(f"video_probs shape mismatch for vid={vid}: got {pv.shape}, expected {(C,)}")

    if need_s and has_s:
        ys = load_labels(s_path)
        if ys.shape != y.shape or not np.allclose(ys, y):
            raise RuntimeError(f"Label mismatch between modalities for vid={vid}")

        ps = load_npz_1d(s_path, "skel_probs")
        if ps.shape != (C,):
            raise RuntimeError(f"skeleton_probs shape mismatch for vid={vid}: got {ps.shape}, expected {(C,)}")

    return y, pa, pv, ps, has_s

