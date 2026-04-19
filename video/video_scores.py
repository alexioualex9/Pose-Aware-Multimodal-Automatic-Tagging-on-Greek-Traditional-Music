import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader
from pathlib import Path

def swap_np(x, i, j):
    x = x.copy()
    x[i], x[j] = x[j], x[i]
    return x

def swap_torch(x, i , j):
    x = x.clone()
    x[i], x[j] = x[j].clone(), x[i].clone()
    return x

@torch.no_grad()
def collect_scores(model, ds, loader, device, store_data=False, time_window="3.69", dataset_folder="whole_dataset", seed=42):
    # store by vid to guarantee alignment independent of loader order
    gt_by_vid = {}
    probs_by_vid = {}

    for Xs, y, vid in loader:
        vid0 = vid[0]

        Xs = Xs.squeeze(0).to(device)              # [Nv, D]
        logits = model(Xs)                         # [Nv, C]
        s = torch.sigmoid(logits.mean(dim=0))      # [C]

        s_np = s.detach().cpu().numpy().astype(np.float32).reshape(-1)

        if y.dim() == 2:
           y0 = y.squeeze(0)   # [C]
        else:
            y0 = y

        # FIX: swap labels to match canonical labels.json order
        if dataset_folder == "whole_dataset":
           y0 = swap_torch(y0, 22, 23)
           s_np = swap_np(s_np, 22, 23)

        else:
           y0 = swap_torch(y0, 20, 21)
           s_np = swap_np(s_np, 20, 21)

        y_np = y0.detach().cpu().numpy().astype(np.int32).reshape(-1)

        gt_by_vid[vid0] = y_np
        probs_by_vid[vid0] = s_np

        if store_data:
            # Save canonical probs + canonical labels
            video_probs = s_np
            labels = y_np

            out_dir = Path("save_video_probs", time_window, dataset_folder, str(seed), "test")
            out_dir.mkdir(parents=True, exist_ok=True)
            out_path = out_dir / f"{vid0}.npz"

            payload = dict(
                labels=labels,
                video_probs=video_probs,
                video_id=vid0,
                agg_video=np.asarray("mean_probs_videoonly_chunks"),
            )
            np.savez_compressed(out_path, **payload)

    # Make ordering deterministic (matches metrics script behavior)
    vids = sorted(gt_by_vid.keys())
    Y = np.stack([gt_by_vid[v] for v in vids], axis=0).astype(np.int32)
    S = np.stack([probs_by_vid[v] for v in vids], axis=0).astype(np.float32)

    return vids, Y, S




@torch.no_grad()
def video_level_val_loss(model: nn.Module, loader: DataLoader, criterion, device: str, subset) -> float:
    losses = []
    model.eval()
    for Xs, y, _ in loader:
        Xs = Xs.squeeze(0).to(device)     # [Nv, D]
        y  = y.squeeze(0).to(device)      # [C]
        y = y.float()

        logits = model(Xs)                   # [Nv, C]
        video_logits = logits.mean(dim=0)    # [C]

        if subset:
           y = swap_torch(y, 20, 21)
           video_logits  = swap_torch(video_logits, 20, 21)        # [C]
        else:
           y = swap_torch(y, 22, 23)
           video_logits  = swap_torch(video_logits, 22, 23)        # [C]

        loss = criterion(video_logits, y)
        losses.append(float(loss.detach().cpu()))
    return float(np.mean(losses)) if losses else float("inf")
