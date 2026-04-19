import numpy as np
from typing import Tuple

##########################
# UTILS
#########################
def select_best_spread_by_quality(skels, M: int):
    n = len(skels)
    if n == 0 or n <= M:
        return skels

    times = np.array([s["t"] for s in skels], dtype=np.float32)
    q = np.array([s.get("q", 0.0) for s in skels], dtype=np.float32)

    order = np.argsort(times)
    sk = [skels[i] for i in order]
    t = times[order]
    q = q[order]

    t_min, t_max = float(t[0]), float(t[-1])
    if t_max == t_min:
        top = np.argsort(q)[::-1][:M]
        sel = [sk[i] for i in top]
        sel.sort(key=lambda s: s["t"])
        return sel

    edges = np.linspace(t_min, t_max, M + 1)
    chosen = []
    for b in range(M):
        a, bb = edges[b], edges[b + 1]
        if b == M - 1:
            bb += 1e-6
        cand = [i for i, tt in enumerate(t) if (a <= tt < bb)]
        if not cand:
            continue
        best = cand[int(np.argmax(q[cand]))]
        chosen.append(best)

    if len(chosen) < M:
        remaining = [i for i in range(n) if i not in chosen]
        remaining = sorted(remaining, key=lambda i: q[i], reverse=True)
        for i in remaining:
            if len(chosen) >= M:
                break
            chosen.append(i)

    chosen = sorted(chosen, key=lambda i: t[i])
    return [sk[i] for i in chosen]


def select_spread_skeletons(skels, max_skeletons):
    if max_skeletons <= 0:
        return skels
    n = len(skels)
    if n <= max_skeletons:
        return skels

    times = np.array([s["t"] for s in skels], dtype=np.float32)
    order = np.argsort(times)
    skels_sorted = [skels[i] for i in order]

    idx = np.linspace(0, n - 1, max_skeletons).round().astype(int)
    idx = np.unique(idx)

    skels_sel = [skels_sorted[i] for i in idx]
    skels_sel.sort(key=lambda x: x["t"])
    return skels_sel


def sample_or_pad_frames(frames_TVC: np.ndarray, T: int) -> Tuple[np.ndarray, int, int]:
    """
    frames_TVC: [N,V,3]
    returns out: [T,V,3], valid_len, pad
    """
    N = frames_TVC.shape[0]
    if N == 0:
        raise ValueError("Empty frames_TVC")
    if N >= T:
        idx = np.linspace(0, N - 1, T).round().astype(int)
        out = frames_TVC[idx]
        pad = 0
        valid_len = T
    else:
        pad_len = T - N
        pad = int(pad_len)
        pad_block = np.repeat(frames_TVC[-1][None, :, :], pad_len, axis=0)
        out = np.concatenate([frames_TVC, pad_block], axis=0)
        valid_len = int(N)
    return out.astype(np.float32), valid_len, pad


def build_label_filter(labels_30):
    remove_set = {"genres--Laiko", "instruments--Piano"}
    new_labels = []
    keep_indices = []
    for i, lab in enumerate(labels_30):
        if lab not in remove_set:
            new_labels.append(lab)
            keep_indices.append(i)
    return new_labels, keep_indices