#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Train ONLY an MLP on frozen, pre-extracted clip embeddings.

Protocol (per common practice in the literature):
- TRAIN: 1 random clip PER VIDEO PER EPOCH (re-sampled every epoch).
- VAL/TEST: aggregate over ALL available clips of each video
            by averaging the LOGITS to get video-level predictions.
- Early stopping on VAL video-level BCE loss.

Inputs assume NPZ blobs + index.json like:
  {
    "blob": "/path/to/clip_xxx.npz",   # contains array "feat"
    "video_id": "...",
    "labels": [0/1, ..., 0/1],
    "start_sec": ..., "end_sec": ...,
    ...
  }

"""

import argparse, json, os, sys, random
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import warnings

from training.train_model import train_model
from dataset.video_dataset import TrainOneRandomClipPerEpoch, Video_Val_Test_Dataset
from head_classifier.mlp_head import MLP
from training.standardize import standardization
from utils import load_exclude_ids
from config import EMBEDDINGS_DIR, FINETUNED_EMBEDDINGS_DIR, MODELS_CONFIG, MODELS_DIR, LABELS_DIR, LABELS_SUBSET_DIR, EXCLUDED_IDS

# ---- TF32 warning μόνο σε CUDA. Σε MPS/CPU το κρύβουμε.
if torch.cuda.is_available():
    torch.set_float32_matmul_precision("high")
else:
    warnings.filterwarnings("ignore", message=".*TF32.*", category=UserWarning)


# -----------------------------
# Main
# -----------------------------

def main():
    ap = argparse.ArgumentParser(description="Train MLP on frozen embeddings: 1 random clip per video per epoch; eval on ALL clips.")
    # Dataset
    ap.add_argument("--dataset", type=str, default="lyra", help="lyra")

    # Model
    ap.add_argument("--model_name", type=str, default="slowfast50", choices=["slowfast50", "timesformer", "r21d", "resnet50", "videomae", "vitb16"])

    # Time window in which dataset been examined
    ap.add_argument("--time_window", type=str, default="3.69", choices=["3.69", "8.00"])

    # Subset or Whole Dataset
    ap.add_argument("--subset", type=bool, default=False)

    # Finetuned or Frozen Backbone Embeddings
    ap.add_argument("--embs", type=str, default="frozen", choices=["finetuned","frozen"])

    # Seed
    ap.add_argument("--seed", type=int, default=42, choices=[42, 123, 1337, 2024, 9999])

    # Runtime
    ap.add_argument("--device", type=str, default="cuda")

    args = ap.parse_args()

    # Set Seed
    rng = np.random.RandomState(args.seed)
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)


    config = MODELS_CONFIG[args.dataset].copy()
    config['dataset'] = args.dataset
    config['model_name'] = args.model_name
    config['device'] = torch.device(args.device)
    if args.subset == True:
      config['top_N_tags'] = 28

    # Fix parameters
    if args.subset == True:
       exclude_ids = load_exclude_ids(EXCLUDED_IDS)
       labels = json.load(open(LABELS_SUBSET_DIR))["labels"]
       dataset_folder = "subset"
    else:
       exclude_ids = None
       labels = json.load(open(LABELS_DIR))["labels"]
       dataset_folder = "whole_dataset"


    # Print training details
    print(config)


    # Define train and validation index
    if args.embs == "frozen":
       train_index = os.path.join(EMBEDDINGS_DIR, config['dataset'], config['model_name'], 'train')
       val_index = os.path.join(EMBEDDINGS_DIR, config['dataset'], config['model_name'], 'val')

       if args.subset == True:
          train_index = os.path.join(train_index, "index_danced_28.json")
          val_index = os.path.join(val_index, "index_danced_28.json")
       else:
          train_index = os.path.join(train_index, "index.json")
          val_index = os.path.join(val_index, "index.json")
    else:
       train_index = os.path.join(FINETUNED_EMBEDDINGS_DIR, config['dataset'], config['model_name'], 'train')
       val_index = os.path.join(FINETUNED_EMBEDDINGS_DIR, config['dataset'], config['model_name'], 'val')

       if args.subset == True:
          train_index = os.path.join(train_index, "index_danced_28.json")
          val_index = os.path.join(val_index, "index_danced_28.json")
       else:
          train_index = os.path.join(train_index, "index.json")
          val_index = os.path.join(val_index, "index.json")


    # ---------- TRAINING ----------
    if not train_index or not val_index:
        raise SystemExit("Training mode requires --train_index and --val_index")

    # Standardization (for some of the models): compute from ALL TRAIN clips ONCE
    mean_t = None
    std_t = None

    if config['model_name'] in {"vitb16", "videomae"}:
       config['standardize'] = True
       mean_t, std_t = standardization(train_index, True, exclude_ids, config['device'])
    else:
       config['standardize'] = False

    config['mean'] = mean_t
    config['std'] = std_t


    # Train Dataset/loader
    train_ds = TrainOneRandomClipPerEpoch(train_index, mean_t, std_t, seed=args.seed, exclude_ids=exclude_ids)
    D, C = train_ds.emb_dim, train_ds.C
    train_loader = DataLoader(train_ds, batch_size=config['batch_size'], shuffle=True, num_workers=0)

    # Validation Dataset/loader
    val_ds = Video_Val_Test_Dataset(val_index, mean=config['mean'], std=config['std'], exclude_ids=exclude_ids)
    val_loader = DataLoader(val_ds, batch_size=1, shuffle=False, num_workers=0)

    # Model / Optimizer
    mlp = MLP(D, config['mlp_hidden'], C, config['dropout']).to(config['device'])

    # Optimizer
    if config['optimizer'] == "AdamW":
       config['optimizer'] = torch.optim.AdamW(mlp.parameters(), lr=config['LR'], weight_decay=config['weight_decay'])
    else:
       raise NotImplementedError(
            'No optimizer implementation found for the given config.')

    # Scheduler
    if config['scheduler'] == "cosine_decay":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            config['optimizer'],
            T_max=config['epochs'],   # συνήθως ίσο με epochs
            eta_min=config.get('min_lr', 1e-4)
        )
    else:
       raise NotImplementedError(
            'No scheduler implementation found for the given config.')

    config['scheduler'] = scheduler


    # Define save path
    model_filename = f'{config["model_name"]}.pth'
    saved_models_dir = os.path.join(MODELS_DIR, config['dataset'], args.time_window, args.embs, dataset_folder, str(args.seed))
    Path(saved_models_dir).mkdir(parents=True, exist_ok=True)
    config['save_path'] = os.path.join(saved_models_dir, model_filename)

    # Train Model
    train_model(train_ds, train_loader, val_loader, mlp, labels, config, args.subset)


if __name__ == "__main__":
    main()
