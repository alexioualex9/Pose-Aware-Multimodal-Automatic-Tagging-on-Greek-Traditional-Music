
from __future__ import annotations

from pathlib import Path
from typing import List, Dict, Callable, Any, Optional
import json
import random
import numpy as np
import pandas as pd
import sys
import os


from video.extract_video_embeddings.emb_utils import get_duration, iter_windows_like_split_spectrogram

sys.path.append(os.path.join(
    os.path.dirname(__file__),
    '/home/alexalexiou/Unimodals/'
))

from ccml.config import SPECTROGRAMS_ATTRIBUTES, MODELS_CONFIG



def _np_dtype_for_storage(dtype: str):
    # numpy δεν υποστηρίζει παντού bf16, άρα αποθηκεύουμε fp16 για fp16/bf16
    return np.float32 if dtype == "fp32" else np.float16


def extract_emb(
    *,
    split_name: str,
    ids: List[str],
    df_source: pd.DataFrame,
    vid2lab: Dict[str, np.ndarray],
    labels: List[str],
    out_dir: Path,
    extract_fn: Callable[..., "Any"],      # επιστρέφει torch.Tensor [D]
    extract_kwargs: Optional[Dict[str, Any]] = None,
    dtype: str,
    num_frames: int,
    config: Dict,
):
    """
    Generic split processor.
    - extract_fn(video_path, start_sec, end_sec, device, dtype, **extract_kwargs) -> torch.Tensor [D]
    """

    extract_kwargs = extract_kwargs or {}

    out_dir.mkdir(parents=True, exist_ok=True)
    blobs_dir = out_dir / "blobs"
    blobs_dir.mkdir(parents=True, exist_ok=True)

    if "id" not in df_source.columns:
        raise RuntimeError(f"{split_name} source TSV must contain an 'id' column")

    df = df_source[df_source["id"].astype(str).isin(set(ids))]
    C = len(labels)

    manifest = []
    skipped = 0
    processed = 0

    sr = SPECTROGRAMS_ATTRIBUTES["audio_sr"]          # 16000
    hop_length = SPECTROGRAMS_ATTRIBUTES["hop_length"]       # 256
    if config['audio_model_name'] == "vgg_ish":
       input_length_secs = MODELS_CONFIG["lyra"]["vgg_ish"]["input_length_in_secs"]
       split_length_frames = int(round(input_length_secs * sr / hop_length)) - 1
    elif config['audio_model_name'] == "ast":
       input_length_secs = MODELS_CONFIG["lyra"]["ast"]["input_length_in_secs"]
       split_length_frames = int(round(input_length_secs * sr / hop_length))
    else:
       raise ValueError ("No implementation for synchronization with {config['audio_model_name']} audio model")

    mel_root = "/data/datasets/mir_datasets/lyra/mel-spectrograms/"
    for _, row in df.iterrows():
        vid_id = str(row["id"])
        vrel = config['video_template'].format(id=vid_id)
        vpath = config['video_dir'] / vrel

        if not vpath.exists():
            print(f"[warn] [{split_name}] missing video for id={vid_id}: {vpath}")
            skipped += 1
            continue

        mel_file = os.path.join(mel_root, f"{vid_id}.npy")
        mel = np.load(mel_file)  # ή όπως το φορτώνεις
        n_mel_frames = mel.shape[1]

        y = vid2lab.get(vid_id, np.zeros(C, dtype=np.float32))

        for clip_idx, w in enumerate(
            iter_windows_like_split_spectrogram(split_length=split_length_frames, sr=sr, n_mel_frames=n_mel_frames, hop_length=hop_length, offset_frames=0, keep_residual=False)):
            s = w["start_sec"]
            e = w["end_sec"]
            try:
                feats = extract_fn(
                    video_path=str(vpath),
                    start_sec=float(s),
                    end_sec=float(e),
                    device=config['device'],
                    dtype=dtype,
                    **extract_kwargs,
                )
            except Exception as ex:
                print(f"[warn] [{split_name}] failed on {vpath} window ({s:.2f},{e:.2f}): {ex}")
                skipped += 1
                continue

            # expect torch.Tensor [D]
            try:
                emb_dim = int(feats.numel())
            except Exception:
                raise RuntimeError("extract_fn must return a torch.Tensor-like object with .numel()")

            stem = f"{vid_id}_s{w['start_sample']}_e{w['end_sample']}"
            out_path = blobs_dir / f"{stem}.npz"

            meta = {
                "split": split_name,
                "video_id": vid_id,
                "video_path": str(vpath),
                "start_sec": float(s),
                "end_sec": float(e),
                "start_sample": int(w["start_sample"]),
                "end_sample": int(w["end_sample"]),
                "emb_dim": emb_dim,
                "dtype": dtype,
                "num_frames": int(num_frames),
                "labels": y.astype(np.float32).tolist(),
                "C": int(C),
            }

            np.savez_compressed(
                out_path,
                feat=np.asarray(feats, dtype=_np_dtype_for_storage(dtype)),
                **meta
            )
            manifest.append({"blob": str(out_path), **meta})

        processed += 1

    with open(out_dir / "index.json", "w") as f:
        json.dump(manifest, f, indent=2)

    print(f"[done] {split_name}: videos={processed} | clips={len(manifest)} | skipped={skipped}")


