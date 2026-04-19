from pathlib import Path
import numpy as np
import sys
import os
import torch

from head_classifier.mlp_head import MLP

sys.path.append(os.path.join(
    os.path.dirname(__file__),
    '/Users/alexandrosalexiou/Desktop/Σχολή/Thesis/VIDEO_MODELS/extract_video_embeddings'
))

from config import MODELS_CONFIG

def load_pt_head(config):

    path = config['model_path']
    device = config['device']

    if not Path(path).exists():
        raise FileNotFoundError(f"Checkpoint not found: {path}")
    try:
        ckpt = torch.load(path, map_location=device, weights_only=False)
    except TypeError:
        ckpt = torch.load(path, map_location=device)
    except pickle.UnpicklingError:
        add_safe_globals([np.core.multiarray._reconstruct])
        with safe_globals([np.core.multiarray._reconstruct]):
            ckpt = torch.load(path, map_location=device, weights_only=True)
    except Exception:
        add_safe_globals([np.core.multiarray._reconstruct])
        with safe_globals([np.core.multiarray._reconstruct]):
            ckpt = torch.load(path, map_location=device, weights_only=True)

    d = int(ckpt["emb_dim"])
    C = int(ckpt["num_classes"])
    model = MLP(d, MODELS_CONFIG['lyra']['mlp_hidden'], C).to(device)
    model.load_state_dict(ckpt["state_dict"]); model.eval()
    labels = ckpt.get("labels", None)
    mean = ckpt.get("mean", None)
    std = ckpt.get("std", None)
    standardize = bool(ckpt.get("standardize", False))
    return model, labels, (mean if standardize else None), (std if standardize else None)
