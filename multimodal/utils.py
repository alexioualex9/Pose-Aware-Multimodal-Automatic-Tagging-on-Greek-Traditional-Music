import numpy as np
import torch
import torch.nn as nn
import random
from typing import List
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc

def seed_everything(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def return_active_mod(model_name):
    if model_name == "seq_transformer_av":
       return ("audio", "video")
    elif model_name == "seq_transformer_as_masked":
       return ("audio", "skeleton")
    elif model_name == "seq_transformer_vs_masked":
       return ("video", "skeleton")
    elif model_name == "seq_transformer_avs_masked":
       return ("audio", "video", "skeleton")

def inp_mods(model_name, dim_a, dim_v, dim_s):
    if model_name == "seq_transformer_av":
       return {"audio": dim_a, "video": dim_v}
    elif model_name == "seq_transformer_as_masked":
       return {"audio": dim_a, "skeleton": dim_s}
    elif model_name == "seq_transformer_vs_masked":
       return {"video": dim_v, "skeleton": dim_s}
    elif model_name == "seq_transformer_avs_masked":
       return {"audio": dim_a, "video": dim_v, "skeleton": dim_s}

# -------------------------
# Arg helpers
# -------------------------
def parse_modalities(mod_str: str) -> List[str]:
    s = mod_str.strip().lower()
    if "," in s:
        toks = [t.strip() for t in s.split(",") if t.strip()]
    elif " " in s:
        toks = [t.strip() for t in s.split(" ") if t.strip()]
    else:
        toks = list(s)

    toks = [t for t in toks if t in ("a", "v", "s")]
    out = []
    for t in ("a", "v", "s"):
        if t in toks and t not in out:
            out.append(t)
    if not out:
        raise ValueError("No valid modalities parsed. Use --modalities like 'a,v,s' or 'av' or 'v,s'.")
    return out


#######################
# Collate Audio + Video
#######################
def collate_av(batch):
    # batch items: (emb_a [T,Da], emb_v [T,Dv], y [C], vid str, L int)
    emb_a_list, emb_v_list, y_list, vid_list, L_list = zip(*batch)

    B = len(batch)
    Da = emb_a_list[0].shape[-1]
    Dv = emb_v_list[0].shape[-1]
    C  = y_list[0].shape[-1]

    Tmax = max(int(L) for L in L_list)

    A = torch.zeros(B, Tmax, Da, dtype=torch.float32)
    V = torch.zeros(B, Tmax, Dv, dtype=torch.float32)
    Y = torch.stack([y.float() for y in y_list], dim=0)
    lengths = torch.tensor(L_list, dtype=torch.long)

    valid_mask = torch.zeros(B, Tmax, dtype=torch.bool)  # True = valid
    for b, (a, v, L) in enumerate(zip(emb_a_list, emb_v_list, L_list)):
        L = int(L)
        A[b, :L] = a[:L].float()
        V[b, :L] = v[:L].float()
        valid_mask[b, :L] = True

    return A, V, Y, valid_mask, vid_list


##########################
# Collate Audio + Skeleton
##########################
def collate_as(batch):
    # batch items: (emb_a [L,Da], emb_s [L,Ds], mask_s [L], y [C], vid str, L int)
    emb_a_list, emb_s_list, mask_s_list, y_list, vid_list, L_list = zip(*batch)

    B = len(batch)
    Da = emb_a_list[0].shape[-1]
    Ds = emb_s_list[0].shape[-1]
    C  = y_list[0].shape[-1]

    Tmax = max(int(L) for L in L_list)

    A = torch.zeros(B, Tmax, Da, dtype=torch.float32)
    S = torch.zeros(B, Tmax, Ds, dtype=torch.float32)
    MS = torch.zeros(B, Tmax, dtype=torch.float32)  # 1.0 where skeleton exists, else 0.0 (even within valid clips)
    Y = torch.stack([y.float() for y in y_list], dim=0)

    valid_mask = torch.zeros(B, Tmax, dtype=torch.bool)  # True where timestep is real (not padding)

    for b, (a, s, ms, L) in enumerate(zip(emb_a_list, emb_s_list, mask_s_list, L_list)):
        L = int(L)
        if L <= 0:
            continue

        A[b, :L] = a[:L].float()
        S[b, :L] = s[:L].float()
        MS[b, :L] = ms[:L].float()
        valid_mask[b, :L] = True

    return A, S, MS, Y, valid_mask, vid_list



##########################
# Collate Video + Skeleton
##########################
def collate_vs(batch):
    # batch items: (emb_v [L,Dv], emb_s [L,Ds], mask_s [L], y [C], vid str, L int)
    emb_v_list, emb_s_list, mask_s_list, y_list, vid_list, L_list = zip(*batch)

    B = len(batch)
    Dv = emb_v_list[0].shape[-1]
    Ds = emb_s_list[0].shape[-1]
    C  = y_list[0].shape[-1]

    Tmax = max(int(L) for L in L_list)

    V = torch.zeros(B, Tmax, Dv, dtype=torch.float32)
    S = torch.zeros(B, Tmax, Ds, dtype=torch.float32)
    MS = torch.zeros(B, Tmax, dtype=torch.float32)  # 1.0 where skeleton exists, else 0.0 (even within valid clips)
    Y = torch.stack([y.float() for y in y_list], dim=0)

    valid_mask = torch.zeros(B, Tmax, dtype=torch.bool)  # True where timestep is real (not padding)

    for b, (v, s, ms, L) in enumerate(zip(emb_v_list, emb_s_list, mask_s_list, L_list)):
        L = int(L)
        if L <= 0:
            continue

        V[b, :L] = v[:L].float()
        S[b, :L] = s[:L].float()
        MS[b, :L] = ms[:L].float()
        valid_mask[b, :L] = True

    return V, S, MS, Y, valid_mask, vid_list



############################
# Collate Audio + Video + Skeleton
############################
def collate_avs(batch):
    """
    Expects each dataset item to be:
      (emb_a [L,Da], emb_v [L,Dv], emb_s [L,Ds], mask_s [L] or [L,1], y [C], vid str, L int)

    Returns:
      A: [B,Tmax,Da]
      V: [B,Tmax,Dv]
      S: [B,Tmax,Ds]
      MS: [B,Tmax] float (1.0 skeleton exists, else 0.0)
      Y: [B,C]
      valid_mask: [B,Tmax] bool (True real clip, False padding)
      vid_list: tuple/list of strings
    """
    emb_a_list, emb_v_list, emb_s_list, mask_s_list, y_list, vid_list, L_list = zip(*batch)

    B = len(batch)
    Da = emb_a_list[0].shape[-1]
    Dv = emb_v_list[0].shape[-1]
    Ds = emb_s_list[0].shape[-1]
    C  = y_list[0].shape[-1]

    Tmax = max(int(L) for L in L_list)

    A = torch.zeros(B, Tmax, Da, dtype=torch.float32)
    V = torch.zeros(B, Tmax, Dv, dtype=torch.float32)
    S = torch.zeros(B, Tmax, Ds, dtype=torch.float32)
    MS = torch.zeros(B, Tmax, dtype=torch.float32)     # skeleton exists mask (even within valid clips)
    Y = torch.stack([y.float() for y in y_list], dim=0)

    valid_mask = torch.zeros(B, Tmax, dtype=torch.bool)

    for b, (a, v, s, ms, L) in enumerate(zip(emb_a_list, emb_v_list, emb_s_list, mask_s_list, L_list)):
        L = int(L)
        if L <= 0:
            continue

        A[b, :L] = a[:L].float()
        V[b, :L] = v[:L].float()
        S[b, :L] = s[:L].float()

        # ms may be [L] or [L,1]
        if ms.dim() > 1:
            ms = ms.squeeze(-1)
        MS[b, :L] = ms[:L].float()

        valid_mask[b, :L] = True

    return A, V, S, MS, Y, valid_mask, vid_list



# -------------------------
# Math helpers
# -------------------------
def sigmoid_np(x: np.ndarray) -> np.ndarray:
    x = np.clip(x, -50, 50)
    return 1.0 / (1.0 + np.exp(-x))

def softmax(x: np.ndarray, axis: int = -1) -> np.ndarray:
    x = x - np.max(x, axis=axis, keepdims=True)
    e = np.exp(x)
    return e / np.sum(e, axis=axis, keepdims=True)


@torch.no_grad()
def infer_as_logits(model_as: nn.Module, emb_a: torch.Tensor, emb_s: torch.Tensor, mask_s: torch.Tensor, device: str) -> np.ndarray:
    logits = model_as(emb_a.unsqueeze(0).to(device), emb_s.unsqueeze(0).to(device), mask_s.unsqueeze(0).to(device))
    return logits.squeeze(0).detach().cpu().numpy().astype(np.float32)


@torch.no_grad()
def infer_vs_logits(model_vs: nn.Module, emb_v: torch.Tensor, emb_s: torch.Tensor, mask_s: torch.Tensor, device: str) -> np.ndarray:
    logits = model_vs(emb_v.unsqueeze(0).to(device), emb_s.unsqueeze(0).to(device), mask_s.unsqueeze(0).to(device))
    return logits.squeeze(0).detach().cpu().numpy().astype(np.float32)


@torch.no_grad()
def infer_avs_logits(model_avs: nn.Module, emb_a: torch.Tensor, emb_v: torch.Tensor, emb_s: torch.Tensor, mask_s: torch.Tensor, device: str) -> np.ndarray:
    logits = model_avs(
        emb_a.unsqueeze(0).to(device),
        emb_v.unsqueeze(0).to(device),
        emb_s.unsqueeze(0).to(device),
        mask_s.unsqueeze(0).to(device),
    )
    return logits.squeeze(0).detach().cpu().numpy().astype(np.float32)


# Save metrics in a .txt file
def save_metrics(metrics, out_dir, modalities, audio_model_name, video_model_name, skeleton_model_name, dataset, split):

    # TXT classification report
    if "a" in modalities and not "v" in modalities and not "s" in modalities:
       with open(out_dir, "w", encoding="utf-8") as f:
           f.write(f'Evaluation of model "{audio_model_name}" on "{dataset}" {split} set:\n')
           f.write(f'ROC-AUC score: {metrics["roc_macro"]}\n')
           f.write(f'PR-AUC score: {metrics["pr_macro"]}\n')
           f.write(f'F1 score: {metrics["f1_macro"]}\n')
           f.write(metrics["report"])

    if not "a" in modalities and "v" in modalities and not "s" in modalities:
       with open(out_dir, "w", encoding="utf-8") as f:
           f.write(f'Evaluation of model "{video_model_name}" on "{dataset}" {split} set:\n')
           f.write(f'ROC-AUC score: {metrics["roc_macro"]}\n')
           f.write(f'PR-AUC score: {metrics["pr_macro"]}\n')
           f.write(f'F1 score: {metrics["f1_macro"]}\n')
           f.write(metrics["report"])

    if not "a" in modalities and not "v" in modalities and "s" in modalities:
       with open(out_dir, "w", encoding="utf-8") as f:
           f.write(f'Evaluation of model "{skeleton_model_name}" on "{dataset}" {split} set:\n')
           f.write(f'ROC-AUC score: {metrics["roc_macro"]}\n')
           f.write(f'PR-AUC score: {metrics["pr_macro"]}\n')
           f.write(f'F1 score: {metrics["f1_macro"]}\n')
           f.write(metrics["report"])

    if "a" in modalities and "v" in modalities and not "s" in modalities:
       with open(out_dir, "w", encoding="utf-8") as f:
           f.write(f'Evaluation of model "{audio_model_name}" + "{video_model_name}" on "{dataset}" {split} set:\n')
           f.write(f'ROC-AUC score: {metrics["roc_macro"]}\n')
           f.write(f'PR-AUC score: {metrics["pr_macro"]}\n')
           f.write(f'F1 score: {metrics["f1_macro"]}\n')
           f.write(metrics["report"])

    if not "a" in modalities and "v" in modalities and "s" in modalities:
       with open(out_dir, "w", encoding="utf-8") as f:
           f.write(f'Evaluation of model "{video_model_name}" + "{skeleton_model_name}" on "{dataset}" {split} set:\n')
           f.write(f'ROC-AUC score: {metrics["roc_macro"]}\n')
           f.write(f'PR-AUC score: {metrics["pr_macro"]}\n')
           f.write(f'F1 score: {metrics["f1_macro"]}\n')
           f.write(metrics["report"])

    if "a" in modalities and not "v" in modalities and "s" in modalities:
       with open(out_dir, "w", encoding="utf-8") as f:
           f.write(f'Evaluation of model "{audio_model_name}" + "{skeleton_model_name}" on "{dataset}" {split} set:\n')
           f.write(f'ROC-AUC score: {metrics["roc_macro"]}\n')
           f.write(f'PR-AUC score: {metrics["pr_macro"]}\n')
           f.write(f'F1 score: {metrics["f1_macro"]}\n')
           f.write(metrics["report"])

    if "a" in modalities and "v" in modalities and "s" in modalities:
       with open(out_dir, "w", encoding="utf-8") as f:
           f.write(f'Evaluation of model "{audio_model_name}" + "{video_model_name}" + {skeleton_model_name} on "{dataset}" {split} set:\n')
           f.write(f'ROC-AUC score: {metrics["roc_macro"]}\n')
           f.write(f'PR-AUC score: {metrics["pr_macro"]}\n')
           f.write(f'F1 score: {metrics["f1_macro"]}\n')
           f.write(metrics["report"])


#########################
# Plot ROC curve
#########################
def macro_roc_curve(Y, S, n_grid=1001):
    """
    Y: (N, C) ground truth (0/1)
    S: (N, C) scores (probabilities from sigmoid)
    Returns: fpr_grid, tpr_macro, auc_macro, auc_per_class
    """
    Y = np.asarray(Y)
    S = np.asarray(S)
    N, C = Y.shape

    fpr_grid = np.linspace(0.0, 1.0, n_grid)

    tprs_interp = []
    auc_per_class = {}

    for c in range(C):
        y_c = Y[:, c]
        s_c = S[:, c]

        # χρειάζεται και 0 και 1 για ROC
        if len(np.unique(y_c)) < 2:
            continue

        fpr_c, tpr_c, _ = roc_curve(y_c, s_c)
        auc_c = auc(fpr_c, tpr_c)
        auc_per_class[c] = auc_c

        # interpolate TPR σε κοινό grid FPR
        tpr_interp = np.interp(fpr_grid, fpr_c, tpr_c)
        tpr_interp[0] = 0.0
        tprs_interp.append(tpr_interp)

    if len(tprs_interp) == 0:
        raise ValueError("No valid classes for ROC (each class needs both positives and negatives).")

    tpr_macro = np.mean(np.vstack(tprs_interp), axis=0)
    tpr_macro[-1] = 1.0

    auc_macro = np.mean(list(auc_per_class.values()))
    return fpr_grid, tpr_macro, auc_macro, auc_per_class


def plot_macro_roc_av(Y, S, title="Late Fusion Macro ROC", save_path=None):
    fpr, tpr_macro, auc_macro, auc_per_class = macro_roc_curve(Y, S)

    plt.figure()
    plt.plot(fpr, tpr_macro, label=f"macro ROC (macro AUC={auc_macro:.3f})")
    plt.plot([0, 1], [0, 1], linestyle="--", label="random")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("AV Macro ROC")
    plt.grid(True, alpha=0.3)
    plt.legend()

    if save_path:
        plt.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.show()

    return auc_macro, auc_per_class


