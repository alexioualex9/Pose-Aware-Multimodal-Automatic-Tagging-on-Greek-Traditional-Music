import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader

def swap_np(x, i, j):
    x = x.copy()
    if x.ndim == 1:          # [C]
        x[i], x[j] = x[j], x[i]
    elif x.ndim == 2:        # [B, C]
        x[:, [i, j]] = x[:, [j, i]]
    else:
        raise ValueError(f"Unsupported shape {x.shape}")
    return x

def swap_torch(x: torch.Tensor, i: int, j: int) -> torch.Tensor:
    # x: [B,C] ή [C]
    x = x.clone()
    if x.dim() == 1:
        x[i], x[j] = x[j].clone(), x[i].clone()
    elif x.dim() == 2:
        x[:, [i, j]] = x[:, [j, i]]
    else:
        raise ValueError(f"Expected 1D or 2D tensor, got {x.shape}")
    return x

@torch.no_grad()
def collect_video_scores_av(model: nn.Module, loader: DataLoader, device: str, subset):
    vids, Y_list, S_list = [], [], []
    model.eval()
    for emb_a, emb_v, y, valid_mask, vid in loader:
        emb_a = emb_a.to(device)
        emb_v = emb_v.to(device)
        emb_s = emb_a
        mask_s = emb_a
        valid_mask = valid_mask.to(device)

        logits = model(emb_a, emb_v, valid_mask=valid_mask)

#        out = model(emb_a, emb_v, emb_s, mask_s, valid_mask=valid_mask)
#        logits = out[0] if isinstance(out, (tuple, list)) else out
        s = torch.sigmoid(logits).cpu().numpy()

        if subset:
          y = swap_torch(y, 20, 21)
          s = swap_np(s, 20, 21)
        else:
          y = swap_torch(y, 22, 23)
          s = swap_np(s, 22, 23)

        vids.append(vid[0] if isinstance(vid, (list, tuple)) else vid)
        Y_list.append(y.cpu().numpy())
        S_list.append(s)
    Y = np.concatenate(Y_list, 0)
    S = np.concatenate(S_list, 0)
    return vids, Y, S


@torch.no_grad()
def collect_video_scores_avs(model: nn.Module, loader: DataLoader, device: str, subset):
    vids, Y_list, S_list = [], [], []
    model.eval()
    for emb_a, emb_v, emb_s, mask_s, y, valid_mask, vid in loader:
        emb_a = emb_a.to(device)
        emb_v = emb_v.to(device)
        emb_s = emb_s.to(device)
        mask_s = mask_s.to(device)
        valid_mask = valid_mask.to(device)

        #logits, _ = model(emb_a, emb_v, emb_s, mask_s, valid_mask=valid_mask)

        out = model(emb_a, emb_v, emb_s, mask_s, valid_mask=valid_mask)
        logits = out[0] if isinstance(out, (tuple, list)) else out



        s = torch.sigmoid(logits).cpu().numpy()

        if subset:
          y = swap_torch(y, 20, 21)
          s = swap_np(s, 20, 21)
        else:
          y = swap_torch(y, 22, 23)
          s = swap_np(s, 22, 23)

        #vids.append(vid[0] if isinstance(vid, (list, tuple)) else vid)
        vids.extend(list(vid) if isinstance(vid, (list, tuple)) else [vid])
        Y_list.append(y.cpu().numpy())
        S_list.append(s)
    Y = np.concatenate(Y_list, 0)
    S = np.concatenate(S_list, 0)
    return vids, Y, S


@torch.no_grad()
def collect_video_scores_as(model: nn.Module, loader: DataLoader, device: str, subset):
    vids, Y_list, S_list = [], [], []
    model.eval()
    for emb_a, emb_s, mask_s, y, valid_mask, vid in loader:
        emb_a = emb_a.to(device)
        emb_v = emb_a
        emb_s = emb_s.to(device)
        mask_s = mask_s.to(device)
        valid_mask = valid_mask.to(device)

        logits = model(emb_a, emb_s, mask_s, valid_mask=valid_mask)
#        out = model(emb_a, emb_s, mask_s, valid_mask=valid_mask)
#        logits = out[0] if isinstance(out, (tuple, list)) else out

        s = torch.sigmoid(logits).cpu().numpy()

        if subset:
          y = swap_torch(y, 20, 21)
          s = swap_np(s, 20, 21)
        else:
          y = swap_torch(y, 22, 23)
          s = swap_np(s, 22, 23)

        vids.append(vid[0] if isinstance(vid, (list, tuple)) else vid)
        Y_list.append(y.cpu().numpy())
        S_list.append(s)
    Y = np.concatenate(Y_list, 0)
    S = np.concatenate(S_list, 0)
    return vids, Y, S


@torch.no_grad()
def collect_video_scores_vs(model: nn.Module, loader: DataLoader, device: str, subset):
    vids, Y_list, S_list = [], [], []
    model.eval()
    for emb_v, emb_s, mask_s, y, valid_mask, vid in loader:
        emb_v = emb_v.to(device)
        emb_s = emb_s.to(device)
        mask_s = mask_s.to(device)
        valid_mask = valid_mask.to(device)
        emb_a = emb_v

        logits = model(emb_v, emb_s, mask_s, valid_mask=valid_mask)

#        out = model(emb_a, emb_v, emb_s, mask_s, valid_mask=valid_mask)
#        logits = out[0] if isinstance(out, (tuple, list)) else out
        s = torch.sigmoid(logits).cpu().numpy()

        if subset:
          y = swap_torch(y, 20, 21)
          s = swap_np(s, 20, 21)
        else:
          y = swap_torch(y, 22, 23)
          s = swap_np(s, 22, 23)

        vids.append(vid[0] if isinstance(vid, (list, tuple)) else vid)
        Y_list.append(y.cpu().numpy())
        S_list.append(s)
    Y = np.concatenate(Y_list, 0)
    S = np.concatenate(S_list, 0)
    return vids, Y, S
