#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Train / Eval fusion models on PRECOMPUTED embeddings:
  - Video clip embeddings: blobs/<...>.npz with 'feat' (SlowFast)
  - Audio clip embeddings: <audio_emb_dir>/<video_id>.npz with:
        audio_clip_emb: [T, D_a]
        labels:         [C]
        clip_start:     [T]
        clip_end:       [T]

  -Skeleton embeddings: <skel_emb_dir>/<video_id>.npz with:
        clip_emb:   [N_s, D_s]
        clip_start: [N_s]  (seconds)
        clip_end:   [N_s]  (seconds)
        labels:     [C]
"""

import argparse
import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import warnings
if not torch.cuda.is_available():
    warnings.filterwarnings("ignore", message=".*TF32.*", category=UserWarning)

torch.set_float32_matmul_precision("high")


from training import train_transformer
from evaluate.eval_transformer import evaluate_transformer
from dataset.datasets import build_ds
from models.create_model import create_model
from loaders import load_ids_txt, load_label_names
from standardize import standardize_video_embeddings
from print_test_results import test_results
from utils import seed_everything, save_metrics, collate_av, collate_as, collate_vs, collate_avs
from config import (
                    MODELS_CONFIG,
                    MODELS_DIR,
                    AUDIO_EMBEDDINGS_DIR,
                    VIDEO_EMBEDDINGS_DIR,
                    SKELETON_EMBEDDINGS_DIR,
                    LABELS_DIR,
                    LABELS_SUBSET_DIR,
                    EXCLUDED_IDS,
                    EVALUATIONS_DIR,
                    )


transf_technique = {"simple": "self_attention",
                    "gated": "gated",
                    "cross_attention": "cross_attention",
                   }


comb_modalities = {"seq_transformer_av": "AV",
                   "seq_transformer_as_masked": "AS",
                   "seq_transformer_vs_masked": "VS",
                   "seq_transformer_avs_masked": "AVS",
                  }


# ============================================================
# Main
# ============================================================

def main():
    ap = argparse.ArgumentParser()

    ap.add_argument("--model_name", type=str, default="seq_transformer_avs", choices=[
        "seq_transformer_avs_masked",
        "seq_transformer_av",
        "seq_transformer_as_masked",
        "seq_transformer_vs_masked",
    ])
    ap.add_argument("--dataset", type=str, default="lyra")
    ap.add_argument("--subset", action="store_true")
    ap.add_argument("--time_window", default="3.69", choices=["3.69", "8.00"])
    ap.add_argument("--device", type=str, default="cpu", choices=["cuda", "mps", "cpu"])
    ap.add_argument("--standardize", action="store_true")
    ap.add_argument("--transformer", type=str, default="simple", choices=["simple", "gated", "cross_attention"])
    ap.add_argument("--seed", type=int, default=1337, choices=[10, 123, 1001, 2024, 9999])
    ap.add_argument("--eval_only", action="store_true")

    args = ap.parse_args()


    if args.subset:
       dataset_folder = "subset"
       index_filename = "index_danced_28.json"
       labels = load_label_names(LABELS_SUBSET_DIR)
       exclude_ids = load_ids_txt(EXCLUDED_IDS)
    else:
       dataset_folder = "whole_dataset"
       index_filename = "index.json"
       labels = load_label_names(LABELS_DIR)
       exclude_ids = None


    config = MODELS_CONFIG[args.dataset][dataset_folder][transf_technique[args.transformer]][comb_modalities[args.model_name]].copy()
    config['dataset'] = args.dataset
    config['model_name'] = args.model_name
    config['device'] = torch.device(args.device)
    config['transformer'] = args.transformer
    config['skeleton_model_name'] = "STGCN"
    config['video_model_name'] = "slowfast50"

    if args.time_window == "3.69":
       config['audio_model_name'] = "vgg_ish"
    else:
       config['audio_model_name'] = "ast"

    C = len(labels)
    audio_emb_dir = os.path.join(AUDIO_EMBEDDINGS_DIR, args.time_window)

    # Audio's path embeddings
    train_audio_emb_dir = os.path.join(audio_emb_dir, dataset_folder, "train")
    val_audio_emb_dir = os.path.join(audio_emb_dir, dataset_folder, "val")
    test_audio_emb_dir = os.path.join(audio_emb_dir, dataset_folder, "test")

    # Video's path indices
    train_index = os.path.join(VIDEO_EMBEDDINGS_DIR, config['video_model_name'], "train", index_filename)
    val_index = os.path.join(VIDEO_EMBEDDINGS_DIR, config['video_model_name'], "val", index_filename)
    test_index = os.path.join(VIDEO_EMBEDDINGS_DIR, config['video_model_name'], "test", index_filename)

    # Skeleton's path embeddings
    train_skel_emb_dir = os.path.join(SKELETON_EMBEDDINGS_DIR, args.time_window, config['skeleton_model_name'], str(2024), "train")
    val_skel_emb_dir = os.path.join(SKELETON_EMBEDDINGS_DIR, args.time_window, config['skeleton_model_name'], str(2024), "val")
    test_skel_emb_dir = os.path.join(SKELETON_EMBEDDINGS_DIR, args.time_window, config['skeleton_model_name'], str(2024), "test")

    # Set seed
    seed_everything(args.seed)

    # Eval-only
    if args.eval_only:

        # Fix model and output paths
        temp_path = os.path.join(MODELS_DIR, dataset_folder, args.time_window, config['transformer'], str(args.seed))
        output_dir = os.path.join(EVALUATIONS_DIR, config["dataset"], dataset_folder, "transformer", args.time_window, config['transformer'], str(args.seed))
        os.makedirs(output_dir, exist_ok=True)

        if config['model_name'] == "seq_transformer_av":
           model_path = os.path.join(temp_path, f"{config['audio_model_name']}_{config['video_model_name']}", f"best_{config['model_name']}.pt")
           out_dir = os.path.join(output_dir, f"{config['audio_model_name']}_{config['video_model_name']}.txt")
           modalities = ['a', 'v']
           collate = collate_av
        elif config['model_name'] == "seq_transformer_vs_masked":
           model_path = os.path.join(temp_path, f"{config['video_model_name']}_{config['skeleton_model_name']}", f"best_{config['model_name']}.pt")
           out_dir = os.path.join(output_dir, f"{config['video_model_name']}_{config['skeleton_model_name']}.txt")
           modalities = ['v', 's']
           collate = collate_vs
        elif config['model_name'] == "seq_transformer_as_masked":
           model_path = os.path.join(temp_path, f"{config['audio_model_name']}_{config['skeleton_model_name']}", f"best_{config['model_name']}.pt")
           out_dir = os.path.join(output_dir, f"{config['audio_model_name']}_{config['skeleton_model_name']}.txt")
           modalities = ['a', 's']
           collate = collate_as
        else:
           model_path = os.path.join(temp_path, f"{config['audio_model_name']}_{config['video_model_name']}_{config['skeleton_model_name']}", f"best_{config['model_name']}.pt")
           out_dir = os.path.join(output_dir, f"{config['audio_model_name']}_{config['video_model_name']}_{config['skeleton_model_name']}.txt")
           modalities = ['a', 'v', 's']
           collate = collate_avs

        # Load checkpoint
        ckpt = torch.load(model_path, map_location=str(config['device']))

        # Load mean/std if present
        mean_t = std_t = None
        if isinstance(ckpt, dict) and ckpt.get("mean_v") is not None:
            mean_t = torch.from_numpy(ckpt["mean_v"]).to(config['device'])
            std_t  = torch.from_numpy(ckpt["std_v"]).to(config['device'])
            print("[info] Loaded mean/std from checkpoint.")
        else:
            print("[info] No mean/std found in checkpoint (mean_t/std_t=None).")

        # Build Test Dataset (standardization happens INSIDE dataset via mean_t/std_t)
        test_ds = build_ds(test_index, test_audio_emb_dir, test_skel_emb_dir, config['model_name'], exclude_ids, "test", mean_t, std_t)
        test_loader = DataLoader(test_ds, batch_size=config['batch_size'], shuffle=False, num_workers=config['num_workers'], collate_fn=collate)

        # Create model
        dim_a = getattr(test_ds, "emb_dim_a", None)
        dim_v = getattr(test_ds, "emb_dim_v", None)
        dim_s = getattr(test_ds, "emb_dim_s", None)

        model = create_model(
            dim_a=dim_a, dim_v=dim_v, dim_s=dim_s,
            num_labels=C,
            config = config,
        ).to(config['device'])

        # Load model weights
        sd = ckpt["fusion_state_dict"] if isinstance(ckpt, dict) and "fusion_state_dict" in ckpt else ckpt
        model.load_state_dict(sd)
        print(f"[info] Loaded model from {model_path}")

        # Evaluate
        m = evaluate_transformer(test_loader, labels, model, config['model_name'], config['device'], args.subset)

        # Print test results
        test_results(m, audio_model_name=config['audio_model_name'], video_model_name=config['video_model_name'], skeleton_model_name=config['skeleton_model_name'], dataset=config['dataset'], modalities=modalities, split="test", auc_kind="macro", decimals=2)

        # Save results
        save_metrics(m, out_dir, modalities, config['audio_model_name'], config['video_model_name'], config['skeleton_model_name'], dataset=config['dataset'], split="test")

    # Training mode
    if not args.eval_only:

       # Standardize video embeddings (based on TRAIN video clips)
       mean_t = None
       std_t = None
       if args.standardize:
          mean_t, std_t = standardize_video_embeddings(train_index, args.standardize, exclude_ids)

       if config['model_name'] == "seq_transformer_av":
           collate = collate_av
       elif config['model_name'] == "seq_transformer_vs_masked":
           collate = collate_vs
       elif config['model_name'] == "seq_transformer_as_masked":
           collate = collate_as
       else:
           collate = collate_avs

       # Deterministic shuffle μέσω generator
       g = torch.Generator()
       g.manual_seed(args.seed)

       # Build Training Dataset
       train_ds = build_ds(train_index, train_audio_emb_dir, train_skel_emb_dir, config['model_name'], exclude_ids, "train", mean_t, std_t)
       train_loader = DataLoader(train_ds, batch_size=config['batch_size'], shuffle=True, num_workers=config['num_workers'], generator=g, collate_fn=collate)

       # Build Validation Dataset
       val_ds = build_ds(val_index, val_audio_emb_dir, val_skel_emb_dir, config['model_name'], exclude_ids, "val", mean_t, std_t)
       val_loader = DataLoader(val_ds, batch_size=config['batch_size'], shuffle=False, num_workers=config['num_workers'], collate_fn=collate)

       # Get dimensions of each modality's embeddings
       dim_a = getattr(train_ds, "emb_dim_a", None)
       dim_v = getattr(train_ds, "emb_dim_v", None)
       dim_s = getattr(train_ds, "emb_dim_s", None)

       # Create model
       model = create_model(
           dim_a=dim_a, dim_v=dim_v, dim_s=dim_s,
           num_labels=C,
           config=config,
       ).to(config['device'])


       # Train Model
       train_transformer(train_loader, val_loader, model, mean_t, std_t, config, args.subset, args.time_window, args.seed, args.transformer)


if __name__ == "__main__":
    main()
