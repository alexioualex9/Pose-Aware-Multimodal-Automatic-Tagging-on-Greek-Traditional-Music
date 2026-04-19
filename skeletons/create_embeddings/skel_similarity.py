import numpy as np

#########################
# Skeleton Similarity
#########################
def _skeleton_vec(kp: np.ndarray, joint_ids, conf_thresh: float, y_weight: float = 1.0):
    xy = kp[:, :2]
    c = kp[:, 2]
    pts = []
    for j in joint_ids:
        if j < 0 or j >= kp.shape[0]:
            continue
        if c[j] < conf_thresh:
            continue
        x, y = float(xy[j, 0]), float(xy[j, 1])
        if not (np.isfinite(x) and np.isfinite(y)):
            continue
        pts.append([x, y * y_weight])
    if len(pts) < max(2, len(joint_ids) // 2):
        return None
    return np.asarray(pts, dtype=np.float32).reshape(-1)

def _pair_dist(v1: np.ndarray, v2: np.ndarray):
    n = min(v1.size, v2.size)
    if n == 0:
        return np.inf
    d = v1[:n] - v2[:n]
    return float(np.sqrt(np.mean(d * d)))

def filter_by_self_similarity_medoid(
    skels,
    conf_thresh: float = 0.2,
    joint_ids=(5, 6, 11, 12),
    y_weight: float = 1.0,
    tukey_k: float = 1.5,
    min_keep: int = 8,
):
    n = len(skels)
    if n == 0 or n <= min_keep:
        return skels, {"kept": [True] * n, "medoid": 0 if n else None}

    vecs = []
    valid_idx = []
    for i, s in enumerate(skels):
        v = _skeleton_vec(s["keypoints"], joint_ids, conf_thresh, y_weight=y_weight)
        vecs.append(v)
        if v is not None:
            valid_idx.append(i)

    if len(valid_idx) < min_keep:
        return skels, {"kept": [True] * n, "medoid": None}

    m = len(valid_idx)
    D = np.zeros((m, m), dtype=np.float32)
    for a in range(m):
        for b in range(a + 1, m):
            da = _pair_dist(vecs[valid_idx[a]], vecs[valid_idx[b]])
            D[a, b] = da
            D[b, a] = da

    mean_d = D.mean(axis=1)
    medoid_local = int(np.argmin(mean_d))
    medoid_i = valid_idx[medoid_local]
    v_med = vecs[medoid_i]

    d_to_medoid = np.full((n,), np.inf, dtype=np.float32)
    for i in range(n):
        if vecs[i] is None:
            continue
        d_to_medoid[i] = _pair_dist(vecs[i], v_med)

    finite = np.isfinite(d_to_medoid)
    if finite.sum() < min_keep:
        return skels, {"kept": [True] * n, "medoid": medoid_i, "d_to_medoid": d_to_medoid.tolist()}

    vals = d_to_medoid[finite]
    med = float(np.median(vals))
    q1 = float(np.percentile(vals, 25))
    q3 = float(np.percentile(vals, 75))
    iqr = max(q3 - q1, 1e-6)
    thr = med + tukey_k * iqr

    kept = [bool(np.isfinite(d_to_medoid[i]) and (d_to_medoid[i] <= thr)) for i in range(n)]

    if sum(kept) < min_keep:
        order = np.argsort(d_to_medoid)
        kept = [False] * n
        cnt = 0
        for i in order:
            if not np.isfinite(d_to_medoid[i]):
                continue
            kept[int(i)] = True
            cnt += 1
            if cnt >= min_keep:
                break

    sk_f = [s for s, k in zip(skels, kept) if k]
    return sk_f, {"kept": kept, "medoid": medoid_i, "thr": thr, "d_to_medoid": d_to_medoid.tolist()}
