import os, sys
import json
from tqdm import tqdm
import torch
import numpy as np
from torch.utils.data import DataLoader
from pathlib import Path


from config import MODELS_CONFIG, LABELS_SUBSET_DIR, MODELS_DIR, OUT_DIR, EVALUATIONS_DIR
from STGCN_model import STGCNModel
from head_classifiers.mlp_head import MLPHead
from skeleton_dataset import SkelAllClipsPerVideoCTv
from compute_metrics import compute_global_metrics, test_results
from utils import save_metrics


def swap_np(x, i, j):
    x = x.copy()
    x[i], x[j] = x[j], x[i]
    return x

def swap_torch(x, i , j):
    x = x.clone()
    x[i], x[j] = x[j].clone(), x[i].clone()
    return x



@torch.no_grad()
def evaluate_video_level(model, test_loader, labels, config, threshold=0.5,  pool="topk", k=5):

    # Evaluate test set
    vids, Y_list, Z_list = [], [], []
    model.eval()
    for Xs, y, vid in tqdm(test_loader, desc="Eval (video-level)"):
        Xs = Xs.squeeze(0).to(config['device'])  # [Nv,C,T,V]
        y = y.squeeze(0).cpu().numpy()       # [Cout]
        y = swap_np(y, 20, 21)

        logits = model(Xs)             # [Nv,Cout]
        if pool == "mean":
          pooled = logits.mean(dim=0)
        elif pool == "topk":
           kk = min(k, logits.size(0))      # π.χ. top-5 clips ανά video
           vals, _ = torch.topk(logits, k=kk, dim=0)   # topk over clips
           pooled = vals.mean(dim=0)
        else:
            raise ValueError("pool must be 'mean' or 'topk'")

        z_np = torch.sigmoid(pooled).cpu().numpy().astype(np.float32)
        z = swap_np(z_np, 20, 21)

        vids.append(vid[0])
        Y_list.append(y[None, :])
        Z_list.append(z[None, :])

    Y = np.concatenate(Y_list, axis=0)
    Z = np.concatenate(Z_list, axis=0)

    # Compute metrics and print results
    m = compute_global_metrics(Y, Z, threshold, labels)
    test_results(m, model_name=config['model_name'], dataset=config['dataset'], auc_kind="macro", decimals=2)

    return m


def evaluate(args):

    config = MODELS_CONFIG[args.dataset].copy()
    config['model_name'] = args.model_name
    config['dataset'] = args.dataset
    config['finetuning'] = args.finetuning
    config['device'] = torch.device(args.device)

    # Embeddings dir
    out_dir = os.path.join(OUT_DIR, config['dataset'], args.time_window, config['model_name'])

    # Load labels
    labels = json.load(open(LABELS_SUBSET_DIR, "r"))["labels"]
    C_out = len(labels)

    # Define test's index.json path
    test_index = os.path.join(out_dir, "test", "skeleton_index.json")

    # Create test dataset and loader
    test_ds = SkelAllClipsPerVideoCTv(test_index, min_valid=config['min_valid'])
    test_loader = DataLoader(test_ds, batch_size=1, shuffle=False, num_workers=0)

    # Get C_in
    C_in = test_ds.C_in

    # Define STGCN model
    model = STGCNModel(
         num_class=C_out,
         in_channels=C_in,
         hidden_channels=config['hidden_channels'],
         num_layers=config['num_layers'],
         use_edge_importance=config['no_edge_importance'],
         multi_scale_tcn=config['multi_scale_tcn'],
         root=config['root'],
    ).to(config['device'])

    # Model Path
    model_path = os.path.join(MODELS_DIR, args.time_window, config['model_name'], str(args.seed), 'best.pt')

    # Load Model
    sd = torch.load(model_path, map_location=config['device'])
    model.load_state_dict(sd)

    # Evaluate Model
    m = evaluate_video_level(model, test_loader, labels, config, pool=args.pool, k=args.k)

    # Save results
    output_dir = os.path.join(EVALUATIONS_DIR, config['dataset'], args.time_window, str(args.seed))

    Path(output_dir).mkdir(parents=True, exist_ok=True)
    output_filename = f'{config["model_name"]}.txt'
    output_file = os.path.join(output_dir, output_filename)

    save_metrics(m, output_file, config["model_name"], config["dataset"], "test")

