#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import collections
import json
import os
from pathlib import Path
import warnings
import numpy as np
from tqdm import tqdm
import torch
import torch.nn as nn

from STGCN_model import STGCNModel
from config import LABELS_SUBSET_DIR, EMBEDDINGS_DIR, SKELETONS_CONFIG, MODELS_CONFIG, MODELS_DIR, OUT_DIR
if not torch.cuda.is_available():
    warnings.filterwarnings("ignore", message=".*TF32.*", category=UserWarning)

# --------------------------
# Extraction
# --------------------------
@torch.no_grad()
def extract_embeddings(
    index_json: str,
    labels_json: str,
    model_ckpt: str,
    out_dir: str,
    device: str,
    hidden_channels: int,
    num_layers: int,
    min_valid: int,
    use_edge_importance: bool,
    multi_scale_tcn: bool,
    root: int,
    max_clips: int,
):

    num_classes = len(labels_json)

    rows = json.load(open(index_json, "r", encoding="utf-8"))
    if not rows:
        raise RuntimeError(f"No rows in {index_json}")

    by_vid = collections.defaultdict(list)
    for r in rows:
        if int(r.get("valid_len", 0)) >= int(min_valid):
            by_vid[str(r["video_id"])].append(r)

    groups = {vid: sorted(items, key=lambda x: float(x.get("start_sec", 0.0)))
              for vid, items in by_vid.items() if items}
    video_ids = sorted(groups.keys())
    if not video_ids:
        raise RuntimeError(f"No videos after min_valid={min_valid}")

    # infer input shape from first clip
    z0 = np.load(groups[video_ids[0]][0]["blob"])
    if "data" not in z0.files:
        raise RuntimeError(f"Expected 'data' in npz; found: {z0.files}")
    d0 = z0["data"].astype(np.float32)   # [C_in,T,V]
    C_in, T, V = d0.shape
    print(f"[info] inferred input: C_in={C_in}, T={T}, V={V}")

    model = STGCNModel(
        num_class=num_classes,
        in_channels=C_in,
        hidden_channels=hidden_channels,
        num_layers=num_layers,
        use_edge_importance=use_edge_importance,
        multi_scale_tcn=multi_scale_tcn,
        root=root,
    ).to(device)

    print(f"[info] loading ckpt: {model_ckpt}")
    sd = torch.load(model_ckpt, map_location=device)
    model.load_state_dict(sd)
    model.eval()

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    for vid in tqdm(video_ids, desc="Extract skeleton embeddings"):
        items = groups[vid]

        # optional truncation for fixed-ish token budget
        if max_clips > 0 and len(items) > max_clips:
            # keep evenly spaced clips
            idx = np.linspace(0, len(items) - 1, max_clips).round().astype(int)
            items = [items[i] for i in idx]

        N = len(items)
        clip_start = np.asarray([float(r.get("start_sec", 0.0)) for r in items], dtype=np.float32)
        clip_end   = np.asarray([float(r.get("end_sec", 0.0)) for r in items], dtype=np.float32)
        valid_len  = np.asarray([int(r.get("valid_len", -1)) for r in items], dtype=np.int32)
        pad        = np.asarray([int(r.get("pad", -1)) for r in items], dtype=np.int32)

        y = np.asarray(items[0]["labels"], dtype=np.float32)
        if y.shape[0] != num_classes:
            raise RuntimeError(f"Label length mismatch for vid={vid}: got {y.shape[0]} expected {num_classes}")

        X = []
        for r in items:
            z = np.load(r["blob"])
            data = z["data"].astype(np.float32)
            if data.shape != (C_in, T, V):
                raise RuntimeError(f"Bad data shape in {r['blob']}: {data.shape} expected {(C_in,T,V)}")
            X.append(data)
        X = np.stack(X, axis=0)  # [N,C_in,T,V]
        X = torch.from_numpy(X).to(device)

        emb, logits = model.forward_features(X)   # emb:[N,D], logits:[N,C]
        emb = emb.detach().cpu().numpy().astype(np.float32)
        logits = logits.detach().cpu().numpy().astype(np.float32)

        video_emb_mean = emb.mean(axis=0).astype(np.float32)

        # if you want a mask for transformer (no padding here unless you choose to pad later)
        mask = np.ones((N,), dtype=np.int32)

        out_path = out_dir / f"{vid}.npz"
        np.savez_compressed(
            out_path,
            clip_emb=emb,                 # [N,D]  <-- transformer tokens
            clip_logits=logits,           # [N,C]  (optional but handy)
            video_emb_mean=video_emb_mean,# [D]
            labels=y,
            clip_start=clip_start,
            clip_end=clip_end,
            valid_len=valid_len,
            pad=pad,
            mask=mask,
            n_clips=np.asarray(N, dtype=np.int32),
            D=np.asarray(emb.shape[1], dtype=np.int32),
            C_in=np.asarray(C_in, dtype=np.int32),
            T=np.asarray(T, dtype=np.int32),
            V=np.asarray(V, dtype=np.int32),
            num_classes=np.asarray(num_classes, dtype=np.int32),
            min_valid=np.asarray(int(min_valid), dtype=np.int32),
            hidden_channels=np.asarray(int(hidden_channels), dtype=np.int32),
            num_layers=np.asarray(int(num_layers), dtype=np.int32),
            use_edge_importance=np.asarray(int(bool(use_edge_importance)), dtype=np.int32),
            multi_scale_tcn=np.asarray(int(bool(multi_scale_tcn)), dtype=np.int32),
            root=np.asarray(int(root), dtype=np.int32),
            max_clips=np.asarray(int(max_clips), dtype=np.int32),
        )

    print(f"[done] saved embeddings to: {out_dir}")


def main():
    ap = argparse.ArgumentParser("Extract per-video skeleton embeddings (clip tokens).")
    ap.add_argument("--model_name", type=str, default="STGCN", choices=['STGCN', 'CTRGCN'])
    ap.add_argument("--time_window", type=str, default="3.69", choices=['3.69', '8.00'])
    ap.add_argument("--dataset", type=str, default="lyra")
    ap.add_argument("--device", default="cpu", choices=["cpu", "cuda", "mps"])
    ap.add_argument("--seed", default="42", choices=["42", "123", "1337", "2024", "9999"])
    ap.add_argument("--max_clips", type=int, default=0,
                    help="If >0, subsample to at most this many clips per video (for token budget). 0=keep all.")

    args = ap.parse_args()


    config_skel = SKELETONS_CONFIG[args.dataset].copy()
    config_skel['dataset'] = args.dataset

    config_model = MODELS_CONFIG[args.dataset].copy()
    config_model['model_name'] = args.model_name
    config_model['device'] = torch.device(args.device)

    # Load labels
    labels = json.load(open(LABELS_SUBSET_DIR))["labels"]

    # Define train, val and test indices
    skel_train_index = os.path.join(OUT_DIR, config_skel['dataset'], args.time_window, config_model['model_name'], "train", "skeleton_index.json")
    skel_val_index = os.path.join(OUT_DIR, config_skel['dataset'], args.time_window, config_model['model_name'], "val", "skeleton_index.json")
    skel_test_index = os.path.join(OUT_DIR, config_skel['dataset'], args.time_window, config_model['model_name'], "test", "skeleton_index.json")

    # Define model path
    model_path = os.path.join(MODELS_DIR, args.time_window, config_model['model_name'], args.seed, "best.pt")

    # Define output dirs for train, val and test embeddings
    out_train = os.path.join(EMBEDDINGS_DIR, config_skel['dataset'], args.time_window, config_model['model_name'], args.seed, "train")
    out_val = os.path.join(EMBEDDINGS_DIR, config_skel['dataset'], args.time_window, config_model['model_name'], args.seed, "val")
    out_test = os.path.join(EMBEDDINGS_DIR, config_skel['dataset'], args.time_window, config_model['model_name'], args.seed, "test")

    # Extract embeddings
    extract_embeddings(
        index_json=skel_train_index,
        labels_json=labels,
        model_ckpt=model_path,
        out_dir=out_train,
        device=config_model['device'],
        hidden_channels=config_model['hidden_channels'],
        num_layers=config_model['num_layers'],
        min_valid=config_model['min_valid'],
        use_edge_importance=config_model['no_edge_importance'],
        multi_scale_tcn=config_model['multi_scale_tcn'],
        root=config_model['root'],
        max_clips=args.max_clips,
    )


    extract_embeddings(
        index_json=skel_val_index,
        labels_json=labels,
        model_ckpt=model_path,
        out_dir=out_val,
        device=config_model['device'],
        hidden_channels=config_model['hidden_channels'],
        num_layers=config_model['num_layers'],
        min_valid=config_model['min_valid'],
        use_edge_importance=config_model['no_edge_importance'],
        multi_scale_tcn=config_model['multi_scale_tcn'],
        root=config_model['root'],
        max_clips=args.max_clips,
    )


    extract_embeddings(
        index_json=skel_test_index,
        labels_json=labels,
        model_ckpt=model_path,
        out_dir=out_test,
        device=config_model['device'],
        hidden_channels=config_model['hidden_channels'],
        num_layers=config_model['num_layers'],
        min_valid=config_model['min_valid'],
        use_edge_importance=config_model['no_edge_importance'],
        multi_scale_tcn=config_model['multi_scale_tcn'],
        root=config_model['root'],
        max_clips=args.max_clips,
    )


if __name__ == "__main__":
    main()
