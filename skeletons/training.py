import os, random, json
from pathlib import Path
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np

from config import MODELS_CONFIG, MODELS_DIR, LABELS_SUBSET_DIR, OUT_DIR
from skeleton_dataset import (
                              SkeletonClipsDatasetCTv,
                              SkelAllClipsPerVideoCTv,
                             )

from build_model import return_model
from train_model import train_model
from utils import load_embed_stats, collate_videos


def train(args):


    # Set Seed
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    config = MODELS_CONFIG[args.dataset].copy()
    config['model_name'] = args.model_name
    config['dataset'] = args.dataset
    config['finetuning'] = args.finetuning
    config['device'] = torch.device(args.device)

    # Embeddings dir
    out_dir = os.path.join(OUT_DIR, config['dataset'], args.time_window, config['model_name'])

    # Define where the model is going to be stored
    config['save_dir'] = os.path.join(MODELS_DIR, args.time_window, config['model_name'], str(args.seed))
    Path(config['save_dir']).mkdir(parents=True, exist_ok=True)

    # Load labels
    labels = json.load(open(LABELS_SUBSET_DIR))["labels"]
    C_out = len(labels)

    # Define train's and validation's index.json path
    train_index = os.path.join(out_dir, "train", "skeleton_index.json")
    val_index = os.path.join(out_dir, "val", "skeleton_index.json")

    # Deterministic shuffle μέσω generator
    g = torch.Generator()
    g.manual_seed(args.seed)

    # Create train and validation datasets and loaders
    train_ds = SkeletonClipsDatasetCTv(train_index, min_valid=config['min_valid'])
    val_ds   = SkelAllClipsPerVideoCTv(val_index, min_valid=config['min_valid'])

    train_loader = DataLoader(train_ds, batch_size=config['batch_size'], shuffle=True, num_workers=0, generator=g)
    val_loader = DataLoader(val_ds, batch_size=1, shuffle=False, num_workers=0)

    # Get C_in
    ds0 = SkelAllClipsPerVideoCTv(val_index)
    C_in = ds0.C_in

    # Define model
    model = return_model(
       model_name=config['model_name'],
       C_out=C_out,
       C_in=C_in,
       config=config,
   )

    #print(model)

    # Training
    train_model(model, train_loader, val_loader, config)
