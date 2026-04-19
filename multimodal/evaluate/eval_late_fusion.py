# Late Fusion evaluation 
import numpy as np
from metrics.global_metrics import compute_global_metrics
from loaders import load_video_ids_from_index, load_modal_probs_for_vid
from fuse_probs import fuse_probs
from utils import plot_macro_roc_av

def evaluate_late_fusion(a_dir, v_dir, s_dir,
                         modalities, label_names, index_json,
                         w_a, w_v, w_s, fusion, threshold):

    vids = load_video_ids_from_index(index_json)

    """
    from pathlib import Path
    print("example vid:", vids[0])
    print("s file exists?", (Path(s_dir) / f"{vids[0]}.npz").exists())
    """


    Y_list, S_list, used_vids = [], [], []
    missing_a = 0
    missing_v = 0
    missing_skel = 0

    for vid in vids:
        y, pa, pv, ps, has_s = load_modal_probs_for_vid(
            vid=vid,
            a_dir=a_dir,
            v_dir=v_dir,
            s_dir=s_dir,
            modalities=modalities
        )

        if y is None:
            # missing required modality
            if "a" in modalities and pa is None:
                missing_a += 1
            if "v" in modalities and pv is None:
                missing_v += 1
            continue
        if "s" in modalities and s_dir is not None and not has_s:
            missing_skel += 1

        probs_list, weights = [], []
        if "a" in modalities:
            probs_list.append(pa); weights.append(w_a)
        if "v" in modalities:
            probs_list.append(pv); weights.append(w_v)

        # Keep the same number of clips
        Ns = [p.shape[0] for p in probs_list]
        if "a" in modalities and "v" in modalities and "s" not in modalities:
            if len(set(Ns)) != 1:
               Nmin = min(Ns)
               probs_list = [p[:Nmin] for p in probs_list]

        if "s" in modalities and ps is not None:
            probs_list.append(ps); weights.append(w_s)

        fused_prob = fuse_probs(probs_list, weights, fusion)  # probs in [0,1]

        used_vids.append(vid)
        Y_list.append(y.astype(np.int32))
        S_list.append(fused_prob.astype(np.float32))

    if not used_vids:
        raise RuntimeError("No videos evaluated (check dirs / index_json / required flags).")

    Y = np.stack(Y_list, axis=0)
    S = np.stack(S_list, axis=0)
    P = (S >= threshold).astype(np.int8)

    m = compute_global_metrics(Y, S, threshold, label_names)

    auc_macro, auc_per_class = plot_macro_roc_av(
         Y, S,
         title="AV (macro-average ROC)",
         save_path="roc_av_macro.png"  # προαιρετικό
    )

    return Y, S, P, m, used_vids, vids, missing_a, missing_v, missing_skel
