import numpy as np

############################# 
# Quality scoring utilities
#############################
BONES = [
    (5, 7), (7, 9),      # L arm
    (6, 8), (8, 10),     # R arm
    (11, 13), (13, 15),  # L leg
    (12, 14), (14, 16),  # R leg
    (5, 6),              # shoulders
    (11, 12),            # hips
    (5, 11), (6, 12),    # torso sides
]

def per_frame_conf_sum(kp: np.ndarray, conf_thresh: float) -> float:
    c = kp[:, 2]
    return float(c[c >= conf_thresh].sum())

def bone_lengths(kp: np.ndarray, bone_threshold: float, bones=BONES):
    xy = kp[:, :2]
    c = kp[:, 2]
    lengths = []
    for a, b in bones:
        if a >= kp.shape[0] or b >= kp.shape[0]:
            lengths.append(np.nan)
            continue
        if c[a] < bone_threshold or c[b] < bone_threshold:
            lengths.append(np.nan)
            continue
        d = float(np.linalg.norm(xy[a] - xy[b]))
        lengths.append(d if np.isfinite(d) else np.nan)
    return np.array(lengths, dtype=np.float32)

def bone_plausibility_penalty(bl: np.ndarray, bl_ref: np.ndarray, eps: float = 1e-6) -> float:
    m = np.isfinite(bl) & np.isfinite(bl_ref) & (bl_ref > eps) & (bl > eps)
    if not np.any(m):
        return 1.0
    r = np.log((bl[m] + eps) / (bl_ref[m] + eps))
    return float(np.mean(np.abs(r)))

def lr_consistency_penalty(kp: np.ndarray, conf_thresh: float, eps: float = 1e-6) -> float:
    xy = kp[:, :2]
    c = kp[:, 2]
    pairs = [(5, 6), (7, 8), (9, 10), (11, 12), (13, 14), (15, 16)]
    num = 0.0
    den = 0.0
    for l, r in pairs:
        if l >= kp.shape[0] or r >= kp.shape[0]:
            continue
        if c[l] < conf_thresh or c[r] < conf_thresh:
            continue
        w = float(min(c[l], c[r]))
        den += w
        if xy[l, 0] > xy[r, 0]:
            num += w
    if den < eps:
        return 0.5
    return float(num / den)

def jitter_penalty(prev_kp: np.ndarray, kp: np.ndarray, conf_thresh: float, eps: float = 1e-6) -> float:
    x0, y0, c0 = prev_kp[:, 0], prev_kp[:, 1], prev_kp[:, 2]
    x1, y1, c1 = kp[:, 0], kp[:, 1], kp[:, 2]
    m = (
        (c0 >= conf_thresh) & (c1 >= conf_thresh)
        & np.isfinite(x0) & np.isfinite(y0) & np.isfinite(x1) & np.isfinite(y1)
    )
    if not np.any(m):
        return 0.5
    d = np.sqrt((x1[m] - x0[m]) ** 2 + (y1[m] - y0[m]) ** 2)
    return float(np.median(d))

def compute_quality_scores(skels, conf_thresh: float, w_bone: float, w_jitter: float, w_lr: float):
    """
    Adds to each s in skels:
      - conf_sum, bone_pen, lr_pen, jitter, q (higher is better)
    """
    if not skels:
        return None, 0.0

    bone_threshold = 0.03
    bl_all = [bone_lengths(s["keypoints"], bone_threshold) for s in skels]
    bl_all = np.stack(bl_all, axis=0)
    bl_ref = np.nanmedian(bl_all, axis=0)

    prev = None
    jitters = []
    for s in skels:
        kp = s["keypoints"]
        s["conf_sum"] = per_frame_conf_sum(kp, conf_thresh)

        bl = bone_lengths(kp, conf_thresh)
        s["bone_pen"] = bone_plausibility_penalty(bl, bl_ref)

        s["lr_pen"] = lr_consistency_penalty(kp, conf_thresh)

        if prev is None:
            s["jitter"] = 0.0
        else:
            s["jitter"] = jitter_penalty(prev, kp, conf_thresh)
        prev = kp
        jitters.append(s["jitter"])

    jitters = np.array(jitters, dtype=np.float32)
    j_med = float(np.median(jitters[np.isfinite(jitters)])) if np.any(np.isfinite(jitters)) else 0.0
    j_scale = j_med if j_med > 1e-6 else 1.0

    for s in skels:
        j_pen = float(s["jitter"] / j_scale)
        s["q"] = float(np.log1p(s["conf_sum"]) - w_bone * s["bone_pen"] - w_jitter * j_pen - w_lr * s["lr_pen"])

    return bl_ref, j_med
