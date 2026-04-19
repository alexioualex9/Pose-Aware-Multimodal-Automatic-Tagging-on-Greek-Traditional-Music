#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
late_train.py — Fusion over saved VIDEO-LEVEL logits (audio+video+skeleton).

"""

import argparse
from pathlib import Path
import os

from loaders import load_label_names
from print_test_results import test_results
from evaluate.eval_late_fusion import evaluate_late_fusion
from save_results import save_results
from utils import parse_modalities, save_metrics
from config import AUDIO_DIR, VIDEO_DIR, SKELETON_DIR, LABELS_DIR, LABELS_SUBSET_DIR, VIDEO_INDEX, EVALUATIONS_DIR

# -------------------------
# Main
# -------------------------
def main():
    ap = argparse.ArgumentParser(description="Fusion over saved VIDEO-LEVEL logits (audio+video+skeleton).")
    ap.add_argument("--modalities", type=str, default="a,v,s",
                    help="Subset of {a,v,s}. Examples: 'a,v', 'v,s', 'av', 's'.")
    ap.add_argument("--fusion", type=str, default="weighted",
                    choices=[
                        "mean", "weighted", "sum",
                    ])
    ap.add_argument("--video_model_name", type=str, default="slowfast50", choices=["slowfast50_finetuned", "slowfast50", "timesformer", "r21d", "resnet50", "videomae", "vitb16"])
    ap.add_argument("--skeleton_model_name", type=str, default="STGCN", choices=["STGCN", "CTRGCN"])
    ap.add_argument("--weights", type=str, default="equal", choices=["equal", "f1_macro"])
    ap.add_argument("--subset", action="store_true")
    ap.add_argument("--seed", type=int, default=42, choices=[42, 123, 1337, 2024, 9999])
    ap.add_argument("--time_window", default="3.69", choices=["3.69", "8.00"])

    args = ap.parse_args()


    if args.weights == "equal":
      w_a = w_v = w_s = 1
    elif args.weights == "f1_macro":
      w_a = 0.48; w_v = 0.28; w_s = 0.14;
    else:
      raise ValueError ("No implementation for this type of weights")

    # Modalities used for evaluation
    modalities = parse_modalities(args.modalities)

    # Fix paths and parameters
    if args.subset:
       label_names = load_label_names(LABELS_SUBSET_DIR)
       dataset_folder = "subset"
       index_filename = "index_danced_28.json"
    else:
       label_names = load_label_names(LABELS_DIR)
       dataset_folder = "whole_dataset"
       index_filename = "index.json"


    # Define audio model name from time window used
    if args.time_window == "3.69":
       audio_model_name = "vgg_ish"
    else:
       audio_model_name = "ast"


    # Define probabilities' paths
    video_dir = os.path.join(VIDEO_DIR, args.time_window, dataset_folder, str(args.seed), "test")
    audio_dir = os.path.join(AUDIO_DIR, args.time_window, dataset_folder, "test")
    skel_dir = os.path.join(SKELETON_DIR, args.time_window, str(args.seed), "test")

    # Define dirs for video index
    if args.video_model_name == "slowfast50":
       video_index = os.path.join(VIDEO_INDEX, "extract_video_embeddings/embeddings/lyra/slowfast50/test", index_filename)
    else:
       video_index = os.path.join(VIDEO_INDEX, "finetuning/embeddings/lyra/slowfast50/test", index_filename)

    # dirs
    if "a" in modalities:
       a_dir = Path(audio_dir)
    else:
      a_dir = None

    if "v" in modalities:
       v_dir = Path(video_dir)
    else:
       v_dir = None

    if "s" in modalities:
       s_dir = Path(skel_dir)
    else:
       s_dir = None

    if a_dir == None and v_dir == None and s_dir == None:
       return "No modalities for late fusion given in input"

    # Late fusion
    threshold = 0.5
    Y, S, P, m, used_vids, vids, miss_a, miss_v, miss_s = evaluate_late_fusion(
       a_dir=Path(a_dir) if a_dir else None,
       v_dir=Path(v_dir) if v_dir else None,
       s_dir=Path(s_dir) if s_dir else None,
       modalities=modalities,
       label_names=label_names,
       index_json=video_index,
       w_a=w_a, w_v=w_v, w_s=w_s,
       fusion=args.fusion,
       threshold=threshold
   )

    # Print and Save results
    output_dir = os.path.join(EVALUATIONS_DIR, "lyra", dataset_folder, "late_fusion", args.time_window, str(args.seed))
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    if "a" in modalities and not "v" in modalities and not "s" in modalities:
       out_dir = os.path.join(output_dir, f'{args.audio_model_name}.txt')
       test_results(m, audio_model_name=audio_model_name, dataset="lyra", split="test", modalities=modalities, auc_kind="macro", decimals=2)
    elif not "a" in modalities and "v" in modalities and not "s" in modalities:
       out_dir = os.path.join(output_dir, f'{args.video_model_name}.txt')
       test_results(m, video_model_name=args.video_model_name, dataset="lyra", split="test", modalities=modalities, auc_kind="macro", decimals=2)
    elif not "a" in modalities and not "v" in modalities and "s" in modalities:
       out_dir = os.path.join(output_dir, f'{args.skeleton_model_name}.txt')
       test_results(m, skeleton_model_name=args.skeleton_model_name, dataset="lyra", split="test", modalities=modalities, auc_kind="macro", decimals=2)
    elif "a" in modalities and "v" in modalities and not "s" in modalities:
       out_dir = os.path.join(output_dir, f'{audio_model_name}_{args.video_model_name}.txt')
       test_results(m, audio_model_name=audio_model_name, video_model_name=args.video_model_name, dataset="lyra", split="test", modalities=modalities, auc_kind="macro", decimals=2)
    elif not "a" in modalities and "v" in modalities and "s" in modalities:
       out_dir = os.path.join(output_dir, f'{args.video_model_name}_{args.skeleton_model_name}.txt')
       test_results(m, video_model_name=args.video_model_name, skeleton_model_name=args.skeleton_model_name, dataset="lyra", split="test", modalities=modalities, auc_kind="macro", decimals=2)
    elif "a" in modalities and not "v" in modalities and "s" in modalities:
       out_dir = os.path.join(output_dir, f'{audio_model_name}_{args.skeleton_model_name}.txt')
       test_results(m, audio_model_name=audio_model_name, skeleton_model_name=args.skeleton_model_name, dataset="lyra", split="test", modalities=modalities, auc_kind="macro", decimals=2)
    elif "a" in modalities and "v" in modalities and "s" in modalities:
       out_dir = os.path.join(output_dir, f'{audio_model_name}_{args.video_model_name}_{args.skeleton_model_name}.txt')
       test_results(m, audio_model_name=audio_model_name, video_model_name=args.video_model_name, skeleton_model_name=args.skeleton_model_name, dataset="lyra", split="test", modalities=modalities, auc_kind="macro", decimals=2)
    else:
       raise ValueError(f"Unexpected modalities combination: {modalities}")


    save_metrics(m, out_dir, modalities, audio_model_name, args.video_model_name, args.skeleton_model_name, dataset="lyra", split="test")

    print(f"\n[info] Modalities used: {modalities}")
    print(f"[info] Used videos:     {len(used_vids)}")
    if "s" in modalities and s_dir is not None:
        print(f"[info] Missing skeleton:{miss_s} (handled)")


if __name__ == "__main__":
    main()
