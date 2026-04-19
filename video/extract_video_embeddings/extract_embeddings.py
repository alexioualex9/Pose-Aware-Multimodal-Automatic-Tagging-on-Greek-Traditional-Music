# extract_embeddings.py
import argparse, json, random
from pathlib import Path
from typing import Callable, Dict, Any, Tuple
import sys
import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from video.extract_video_embeddings.transforms import build_video_transforms, build_image_transforms_train, build_image_transforms_eval
from video.extract_video_embeddings.save_embeddings import extract_emb
from video.extract_video_embeddings.feat_extraction import(
                   extract_clip_embedding_slowfast,
                   extract_clip_embedding_timesformer,
                   extract_clip_embedding_2d,
                   extract_clip_embedding_r21d,
                   extract_clip_embedding_videomae,
                  )
from video.extract_video_embeddings.emb_utils import (
            build_topN_labels_from_training_tsv,
            _build_vid2lab_from_tsv,
            load_training_ids,
            load_test_ids,
            split_train_val_ids,
            compute_num_frames,
            )
from video.extract_video_embeddings.load_model import (
            load_pretrained_slowfast_r50, make_backbone_return_embeddings_slowfast50, 
            load_resnet50_imagenet,
            load_pretrained_r2plus1d_18, make_backbone_return_embeddings_r21d,
            load_pretrained_timesformer, make_backbone_return_embeddings_timesformer,
            load_vitb16_imagenet,
            load_pretrained_videomae,
            )

import warnings
if not torch.cuda.is_available():
    warnings.filterwarnings("ignore", message=".*TF32.*", category=UserWarning)


# -----------------------------
# Main
# -----------------------------
def main():
    ap = argparse.ArgumentParser(description="Extract SlowFast-R50 embeddings for train/val/test in ONE run with a single labels.json.")
    # Dataset
    ap.add_argument("--dataset", type=str, default="lyra", help="lyra")

    # Video Model
    ap.add_argument("--model_name", type=str, default="slowfast50", choices=["slowfast50", "timesformer", "r21d", "resnet50", "vitb16", "videomae"])

    # Audio Model
    ap.add_argument("--audio_model_name", type=str, default="vgg_ish", choices=["ast", "vgg_ish"])

    # Splits
    ap.add_argument("--seed", type=int, default=42, help="Random seed for train/val split and train offsets.")

    # Runtime
    ap.add_argument("--dtype", choices=["fp16","bf16","fp32"], default="fp16", help="Precision to store embeddings.")

    # Device
    ap.add_argument("--device", default="cpu", help="mps, cuda or cpu")
    args = ap.parse_args()


    # RNG
    random.seed(args.seed)

    # Χρησιμοποίησε MPS αν υπάρχει (Mac GPU), αλλιώς CUDA, αλλιώς CPU
    if torch.backends.mps.is_available():
        device = "mps"
    elif torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"


    ######## SHOULD BE PLACED IN A config.py TYPE OF FILE. ########
    top_N = 30
    data_dir = "/data/datasets/mir_datasets/lyra"
    val_ratio = 0.1
    ###############################################################


    #config = MODELS_CONFIG[args.dataset].copy()
    config = dict()
    config['dataset'] = args.dataset
    config['data_dir'] = Path(data_dir)
    config['model_name'] = args.model_name
    config['audio_model_name'] = args.audio_model_name
    config['device'] = torch.device(device)
    config['video_template'] = "{id}.mp4"
    config['video_dir'] = config['data_dir'] / "videos"

    # Define save path
    ROOT_DIR = os.path.dirname(os.path.realpath(__file__))
    embeddings_filename = f'{config["model_name"]}'
    saved_models_dir = os.path.join(ROOT_DIR, "embeddings", config['dataset'])
    config['save_path'] = Path(os.path.join(saved_models_dir, embeddings_filename))
    config['save_path'].mkdir(parents=True, exist_ok=True)


    # Canonical labels.json under out_root
    labels_json_path = config['save_path'] / "labels.json"
    Path(labels_json_path)
    if labels_json_path.exists():
        labels = json.load(open(labels_json_path))["labels"]
        print(f"[info] Loaded canonical labels.json from {labels_json_path} ({len(labels)} labels)")
    else:
        labels = build_topN_labels_from_training_tsv(str(config['data_dir']), top_N=top_N)
        json.dump({"labels": labels}, open(labels_json_path, "w"), indent=2)
        print(f"[info] Built & saved canonical labels.json at {labels_json_path} ({len(labels)} labels)")
    C = len(labels)

    # Split IDs
    train_ids_all = load_training_ids(config['data_dir'])
    train_ids_resolved, val_ids_resolved = split_train_val_ids(train_ids_all, val_ratio, args.seed)
    test_ids_resolved = load_test_ids(config['data_dir'])

    # Label maps per table
    vid2lab_train = _build_vid2lab_from_tsv(config['data_dir'] / "split" / "training.tsv", labels)
    vid2lab_test  = _build_vid2lab_from_tsv(config['data_dir'] / "split" / "test.tsv", labels)

    # Number of Frames that should be used according to model used
    num_frames = compute_num_frames(config['model_name'])

    # Load Model
    processor = None
    wrapper = None
    forward_fn = None
    if config['model_name'] == "slowfast50":
        model = load_pretrained_slowfast_r50(device=config['device'])
        emb_dim = make_backbone_return_embeddings_slowfast50(model)
    elif config['model_name'] == "timesformer":
        processor, model = load_pretrained_timesformer("facebook/timesformer-base-finetuned-k400", device=config['device'])
        emb_dim = make_backbone_return_embeddings_timesformer(model)
    elif config['model_name'] == "r21d":
        model = load_pretrained_r2plus1d_18(device=config['device'])
        emb_dim = make_backbone_return_embeddings_r21d(model)
    elif config['model_name'] == "resnet50":
        model, emb_dim = load_resnet50_imagenet(device=config['device'])
        forward_fn = lambda x: model(x)
    elif config['model_name'] == "vitb16":
        model, emb_dim = load_vitb16_imagenet(device=config['device'])
        model.heads = nn.Identity()
        forward_fn = lambda x: model(x)
    elif config["model_name"] == "videomae":
        processor, model, emb_dim = load_pretrained_videomae("MCG-NJU/videomae-base", device=config['device'])

    else:
        raise NotImplementedError(
            'No model implementation found for the given config.')

    # Apply Transforms
    if config['model_name'] in {"slowfast50", "r21d"}:
       transform_eval  = build_video_transforms(config["model_name"], num_frames=num_frames, mode="eval", embs="frozen")
    elif config['model_name'] in {"resnet50", "vitb16"}:
       transform_eval  = build_image_transforms_eval(config['model_name'])
    else:
       transform_train = None
       transform_eval  = None

    # Load tables
    df_training = pd.read_csv(config['data_dir'] / "split" / "training.tsv", sep="\t", keep_default_na=False)
    df_test     = pd.read_csv(config['data_dir'] / "split" / "test.tsv",     sep="\t", keep_default_na=False)

    # Model in eval mode
    model.eval()
    for p in model.parameters():
       p.requires_grad = False

    # Initialize Extractors
    EXTRACTORS: Dict[str, Callable[..., Any]] = {
        "slowfast50": extract_clip_embedding_slowfast,
        "timesformer": extract_clip_embedding_timesformer,
        "resnet50": extract_clip_embedding_2d,
        "r21d": extract_clip_embedding_r21d,
        "vitb16": extract_clip_embedding_2d,
        "videomae": extract_clip_embedding_videomae,
    }

    # Decide which extractor you are going to use according to your model
    extract_fn = EXTRACTORS[config['model_name']]

    # Initialize Arguments
    EXTRACT_KWARGS: Dict[str, Dict[str, Any]] = {
       "slowfast50": {"model": model, "transform": transform_eval},
       "timesformer": {"model": model, "processor": processor, "num_frames": num_frames},
       "resnet50": {"forward_fn": forward_fn, "transform": transform_eval, "num_frames": num_frames},
       "r21d": {"model": model, "transform": transform_eval},
       "vitb16": {"forward_fn": forward_fn, "transform": transform_eval, "num_frames": num_frames},
       "videomae": {"model": model, "processor": processor, "num_frames": num_frames},

    }

    # Decide which arguments should be passed according to your model
    extract_kwargs = EXTRACT_KWARGS.get(config['model_name'], {"model": model})

    # Process each split to config['save_path']/{train,val,test}
    print("[info] Starting extraction for all splits...")

    extract_emb(
       split_name="train",
       ids=train_ids_resolved,
       df_source=df_training,
       vid2lab=vid2lab_train,
       labels=labels,
       out_dir=config['save_path'] / "train",
       extract_fn=extract_fn,
       extract_kwargs=extract_kwargs,
       dtype=args.dtype,
       num_frames=num_frames,
       config = config
    )

    extract_emb(
       split_name="val",
       ids=val_ids_resolved,
       df_source=df_training,
       vid2lab=vid2lab_train,
       labels=labels,
       out_dir=config['save_path'] / "val",
       extract_fn= extract_fn,
       extract_kwargs=extract_kwargs,
       dtype=args.dtype,
       num_frames=num_frames,
       config = config
    )

    extract_emb(
       split_name="test",
       ids=test_ids_resolved,
       df_source=df_test,
       vid2lab=vid2lab_test,
       labels=labels,
       out_dir=config['save_path'] / "test",
       extract_fn= extract_fn,
       extract_kwargs=extract_kwargs,
       dtype=args.dtype,
       num_frames=num_frames,
       config = config
    )

    print(f"[all done] Saved features to: {config['save_path']}")
    print(f"[info] emb_dim={emb_dim} | num_frames={num_frames} | dtype={args.dtype} | C={C} | out_root=config['save_path']")

if __name__ == "__main__":
    main()
