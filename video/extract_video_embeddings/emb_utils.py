from pathlib import Path
from typing import Dict, List, Tuple
import torch
import random
import numpy as np
import pandas as pd

from pytorchvideo.data.encoded_video import EncodedVideo


# -----------------------------
# Label utilities
# -----------------------------
def build_topN_labels_from_training_tsv(data_root: str, top_N: int = 30) -> List[str]:
    ann = Path(data_root) / "split" / "training.tsv"
    if not ann.exists():
        raise FileNotFoundError(f"training.tsv not found at {ann}")
    df = pd.read_csv(ann, sep="\t", keep_default_na=False)
    if "id" not in df.columns:
        raise RuntimeError("Expected column 'id' in training.tsv")

    all_tags = []
    for _, r in df.iterrows():
        for col in ["genres", "instruments", "place"]:
            v = r.get(col, "")
            if isinstance(v, str) and v:
                all_tags.extend([f"{col}--{t}" for t in v.split("|") if t])
    counts = {}
    for t in all_tags:
        counts[t] = counts.get(t, 0) + 1
    top = sorted(counts.items(), key=lambda x: x[1], reverse=True)[:top_N]
    labels = [t for t, _ in top]
    if len(labels) < top_N:
        print(f"[warn] only {len(labels)} labels found (requested {top_N})")
    return labels

def _build_vid2lab_from_tsv(tsv_path: Path, labels: List[str]) -> Dict[str, np.ndarray]:
    df = pd.read_csv(tsv_path, sep="\t", keep_default_na=False)
    if "id" not in df.columns:
        raise RuntimeError(f"{tsv_path} must contain an 'id' column")
    tag_to_idx = {t: i for i, t in enumerate(labels)}
    C = len(labels)
    vid2lab: Dict[str, np.ndarray] = {}
    for _, r in df.iterrows():
        vid = str(r["id"])
        y = np.zeros(C, dtype=np.float32)
        for col in ["genres", "instruments", "place"]:
            v = r.get(col, "")
            if isinstance(v, str) and v:
                for t in v.split("|"):
                    key = f"{col}--{t}"
                    if key in tag_to_idx:
                        y[tag_to_idx[key]] = 1.0
        vid2lab[vid] = y
    return vid2lab




# -----------------------------
# Splitting utilities (video-level)
# -----------------------------
def load_training_ids(data_root: Path) -> List[str]:
    df = pd.read_csv(data_root / "split" / "training.tsv", sep="\t", keep_default_na=False)
    if "id" not in df.columns:
        raise RuntimeError("training.tsv must contain an 'id' column")
    return [str(x) for x in df["id"].tolist()]

def load_test_ids(data_root: Path) -> List[str]:
    df = pd.read_csv(data_root / "split" / "test.tsv", sep="\t", keep_default_na=False)
    if "id" not in df.columns:
        raise RuntimeError("test.tsv must contain an 'id' column")
    return [str(x) for x in df["id"].tolist()]

def split_train_val_ids(all_train_ids: List[str], val_ratio: float, seed: int) -> Tuple[List[str], List[str]]:
    rng = random.Random(seed)
    ids = sorted(set(all_train_ids))
    rng.shuffle(ids)
    n_val = int(round(len(ids) * val_ratio))
    val_ids = set(ids[:n_val])
    train_ids = [x for x in ids if x not in val_ids]
    return train_ids, sorted(val_ids)


# -----------------------------
# Windowing helpers (drop tail)
# -----------------------------
def iter_windows_like_split_spectrogram(
    split_length: int,          # ίδιο με test_dataset.input_length
    n_mel_frames,
    sr: int = 16000,
    hop_length: int = 256,
    offset_frames: int = 0,     # αν θες random offset σε frames
    keep_residual: bool = False,
):
    """
    Παράγει windows με ΑΚΡΙΒΩΣ την ίδια λογική με split_spectrogram,
    αλλά επιστρέφει start/end και σε sec και σε samples για perfect join.
    """
    if n_mel_frames <= 0 or split_length <= 0:
        return

    spectr_length = n_mel_frames
    usable = spectr_length - offset_frames
    if usable < split_length:
        return

    num_spectrs = usable // split_length
    residual = usable % split_length

    frame_dur = hop_length / sr  # 0.016s

    # κύρια chunks
    for i in range(int(num_spectrs)):
        s_frame = offset_frames + i * split_length
        e_frame = s_frame + split_length

        yield {
            "start_mel": int(s_frame),
            "end_mel": int(e_frame),
            "start_sample": int(s_frame * hop_length),
            "end_sample": int(e_frame * hop_length),
            "start_sec": float(s_frame * frame_dur),
            "end_sec": float(e_frame * frame_dur),
        }

    # residual (αν ποτέ θες keep_residual=True όπως στο audio helper)
    if keep_residual and residual:
        s_frame = offset_frames + num_spectrs * split_length
        e_frame = offset_frames + num_spectrs * split_length + residual
        yield {
            "start_mel": int(s_frame),
            "end_mel": int(e_frame),
            "start_sample": int(s_frame * hop_length),
            "end_sample": int(e_frame * hop_length),
            "start_sec": float(s_frame * frame_dur),
            "end_sec": float(e_frame * frame_dur),
            "is_residual": True,
        }




def get_duration(path: str) -> float:
    try:
        return float(EncodedVideo.from_path(path).duration)
    except Exception:
        return 0.0



#####################
# Sampling methods 
#####################
def sample_frame_indices(T: int, num_frames: int) -> np.ndarray:
    """
    Uniform sampling num_frames indices σε [0, T-1].
    Αν T < num_frames -> παίρνει όλα τα frames (χωρίς duplication).
    """
    if T <= 0:
        return np.zeros((0,), dtype=np.int64)
    if num_frames >= T:
        return np.arange(T, dtype=np.int64)
    return np.linspace(0, T - 1, num_frames).astype(np.int64)



def sample_frame_indices_2d(T: int, num_frames: int, mode: str):
    """
    TSN-style: TRAIN -> 1 random index per temporal segment; EVAL -> uniform indices.
    If num_frames >= T, returns T uniform indices (no duplicate sampling here).
    """
    if T <= 0:
        return torch.zeros(0, dtype=torch.long)
    if num_frames >= T:
        return torch.linspace(0, T-1, steps=T).long()
    if mode == "train":
        seg_edges = np.linspace(0, T, num_frames + 1).astype(int)
        idxs = []
        for s, e in zip(seg_edges[:-1], seg_edges[1:]):
            if e <= s:
                e = min(s + 1, T)
            choice = np.random.randint(s, e) if e > s else s
            idxs.append(choice)
        return torch.as_tensor(idxs, dtype=torch.long)
    else:
        return torch.linspace(0, T-1, steps=num_frames).long()



def _uniform_subsample_cthw(v: torch.Tensor, num_frames: int) -> torch.Tensor:
    # v: (C,T,H,W)
    C, T, H, W = v.shape
    if T == num_frames:
        return v
    if T > num_frames:
        idx = torch.linspace(0, T - 1, steps=num_frames).long()
        return v.index_select(1, idx)
    pad = v[:, -1:].repeat(1, num_frames - T, 1, 1)
    return torch.cat([v, pad], dim=1)


###########################################
# Compute number of frames for each model
###########################################
def compute_num_frames(model_name):
    if model_name == "slowfast50":
       return 32
    elif model_name == "timesformer":
       return 8
    else:
       return 16



