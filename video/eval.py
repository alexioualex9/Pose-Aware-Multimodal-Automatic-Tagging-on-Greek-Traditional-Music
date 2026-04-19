#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Evaluate an MLP head **on embeddings**:
- TEST: aggregate logits over **ALL** clips per video (mean) → metrics
"""

import argparse, json
import numpy as np
import torch
from pathlib import Path
from torch.utils.data import DataLoader
import warnings
import os

from print_test_results import test_results
from video_scores import collect_scores
from global_metrics import compute_global_metrics
from dataset.video_dataset import Video_Val_Test_Dataset
from load_head import load_pt_head
from utils import load_exclude_ids, save_metrics
from config import EMBEDDINGS_DIR, FINETUNED_EMBEDDINGS_DIR, MODELS_CONFIG, MODELS_DIR, FINETUNED_MODELS_DIR, EVALUATIONS_DIR, LABELS_DIR, LABELS_SUBSET_DIR, EXCLUDED_IDS

try:
    from torch.serialization import add_safe_globals, safe_globals
except ImportError:
    from contextlib import contextmanager

    def add_safe_globals(*args, **kwargs):
        return None  # no-op on older torch

    @contextmanager
    def safe_globals(*args, **kwargs):
        yield


# κρύψε TF32 warnings όταν δεν είμαστε σε CUDA
if not torch.cuda.is_available():
    warnings.filterwarnings("ignore", message=".*TF32.*", category=UserWarning)



# -------------------- Main --------------------

def main():
    ap = argparse.ArgumentParser(description="Evaluate MLP head on embeddings: TEST=ALL clips (avg logits).")

    # Dataset
    ap.add_argument("--dataset", type=str, default="lyra", help="lyra")

    # Model
    ap.add_argument("--model_name", type=str, default="slowfast50", choices=["slowfast50", "timesformer", "r21d", "resnet50", "videomae", "vitb16"])

    # Subset (Only Danced Videos) or Whole Dataset
    ap.add_argument("--subset", type=bool, default=False)

    # Time window in which dataset been examined
    ap.add_argument("--time_window", type=str, default="3.69", choices=["3.69", "8.00"])

    # Frozen or Finetuned Embeddings
    ap.add_argument("--embs", type=str, default="frozen", choices=["frozen", "finetuned"])

    # Seed
    ap.add_argument("--seed", type=int, default=42, choices=[42, 123, 1337, 2024, 9999])

    # Runtime
    ap.add_argument("--device", default="cuda")

    # outputs
    ap.add_argument("--save_csv", default=None)
    ap.add_argument("--save_npy_prefix", default=None)
    args = ap.parse_args()


    config = MODELS_CONFIG[args.dataset].copy()
    config['dataset'] = args.dataset
    config['model_name'] = args.model_name
    config['device'] = torch.device(args.device)

    # Fix paths and parameters"
    if args.embs == "frozen":
       embs_folder = "frozen"
       test_index = os.path.join(EMBEDDINGS_DIR, config['dataset'], config['model_name'], 'test')
    else:
       embs_folder = "finetuned"
       test_index = os.path.join(FINETUNED_EMBEDDINGS_DIR, config['dataset'], config['model_name'], 'test')

    if args.subset:
       dataset_folder = "subset"
       exclude_ids = load_exclude_ids(EXCLUDED_IDS)
       labels = json.load(open(LABELS_SUBSET_DIR))["labels"]
       test_index = os.path.join(test_index, "index_danced_28.json")
    else:
       dataset_folder = "whole_dataset"
       exclude_ids = None
       labels = json.load(open(LABELS_DIR))["labels"]
       test_index = os.path.join(test_index, "index.json")

    # Define model path
    model_filename = f'{config["model_name"]}.pth'
    saved_models_dir = os.path.join(MODELS_DIR, config['dataset'], args.time_window, embs_folder, dataset_folder, str(args.seed))
    Path(saved_models_dir).mkdir(parents=True, exist_ok=True)
    config['model_path'] = os.path.join(saved_models_dir, model_filename)

    # Load trained head, labels, mean + std from standardization
    model, saved_labels, mean, std = load_pt_head(config)

    if saved_labels is not None and saved_labels != labels:
        print("[warn] labels in checkpoint differ from provided labels.json")

    mean_t = None if mean is None else mean.detach().to("cpu", dtype=torch.float32)
    std_t  = None if std  is None else std.detach().to("cpu", dtype=torch.float32)

    # Load Test Dataset
    ds = Video_Val_Test_Dataset(test_index, mean_t, std_t, exclude_ids=exclude_ids)
    loader = DataLoader(ds, batch_size=1, shuffle=False, num_workers=0)

    # Collect video-level scores of each video
    vids, Y, S = collect_scores(model=model, ds=ds, loader=loader, device=config['device'], time_window=args.time_window, dataset_folder=dataset_folder)

    # thresholds
    th = np.full(S.shape[1], 0.5, dtype=np.float32)
    P = (S >= th[None, :]).astype(np.int32)

    # Compute metrics
    m = compute_global_metrics(Y, S, threshold=0.5, label_names=labels)

    # Print test results
    test_results(m, config['model_name'], dataset="lyra", split="test", auc_kind="macro", decimals=2)

    # Save results
    output_dir = os.path.join(EVALUATIONS_DIR, config['dataset'], embs_folder, args.time_window, dataset_folder, str(args.seed))
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    output_filename = f'{config["model_name"]}.txt'
    output_file = os.path.join(output_dir, output_filename)

    save_metrics(m, output_file, config["model_name"], config["dataset"], "test")

if __name__ == "__main__":
    main()
