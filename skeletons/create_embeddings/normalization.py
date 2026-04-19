import numpy as np

#############################
# Normalization
#############################
def normalize_single_skeleton_coco17(
    kp: np.ndarray,
    conf_thresh: float = 0.5,
    eps: float = 1e-6,
    lhip_idx: int = 11,
    rhip_idx: int = 12,
    lsho_idx: int = 5,
    rsho_idx: int = 6,
    # filtering / robustness
    min_scale: float = 0.15,
    max_abs_xy: float = 50.0,
    min_shoulder_conf: float = 0.0,
    # controls
    already_centered: bool = False,
    do_scale: bool = False,
    clamp_xy: bool = True,
    hard_reject_factor: float = 4.0,
    return_scale: bool = False,
):
    """
    Robust skeleton normalization for COCO17.

    Returns (kp_norm, reason) or (kp_norm, reason, scale)
    kp_norm: [J,3] or None if rejected.
    """
    if kp is None:
        return (None, "bad_shape", None) if return_scale else (None, "bad_shape")
    kp = np.asarray(kp)
    if kp.ndim != 2 or kp.shape[1] < 2:
        return (None, "bad_shape", None) if return_scale else (None, "bad_shape")

    kp = kp.astype(np.float32, copy=True)

    # ensure [J,3]
    if kp.shape[1] == 2:
        conf = np.ones((kp.shape[0], 1), dtype=np.float32)
        kp = np.concatenate([kp, conf], axis=1)

    xy = kp[:, :2].copy()
    c = kp[:, 2].copy()

    finite = np.isfinite(xy[:, 0]) & np.isfinite(xy[:, 1])
    valid = finite & (c >= conf_thresh)

    if not np.any(valid):
        return (None, "no_valid_joints", None) if return_scale else (None, "no_valid_joints")

    J = xy.shape[0]

    def safe_idx(i: int) -> bool:
        return 0 <= int(i) < J

    # optional: require shoulder confidence
    if min_shoulder_conf > 0.0:
        if not (safe_idx(lsho_idx) and safe_idx(rsho_idx)):
            return (None, "no_shoulder_idx", None) if return_scale else (None, "no_shoulder_idx")
        if not (c[lsho_idx] >= min_shoulder_conf and c[rsho_idx] >= min_shoulder_conf):
            return (None, "low_shoulder_conf", None) if return_scale else (None, "low_shoulder_conf")

    # centering
    if not already_centered:
        if safe_idx(lhip_idx) and safe_idx(rhip_idx) and valid[lhip_idx] and valid[rhip_idx]:
            root = 0.5 * (xy[lhip_idx] + xy[rhip_idx])
        elif safe_idx(lhip_idx) and valid[lhip_idx]:
            root = xy[lhip_idx]
        elif safe_idx(rhip_idx) and valid[rhip_idx]:
            root = xy[rhip_idx]
        else:
            root = xy[valid].mean(axis=0)
        xy = xy - root

    # scale estimation (optional)
    scale = 1.0
    if do_scale:
        scale_val: Optional[float] = None
        if safe_idx(lsho_idx) and safe_idx(rsho_idx) and valid[lsho_idx] and valid[rsho_idx]:
            scale_val = float(np.linalg.norm(xy[lsho_idx] - xy[rsho_idx]))
            if not np.isfinite(scale_val):
                scale_val = None
        if scale_val is None:
            radii = np.linalg.norm(xy[valid], axis=1)
            scale_val = float(np.mean(radii)) if radii.size > 0 else 1.0

        if scale_val < eps:
            return (None, "scale_zero", None) if return_scale else (None, "scale_zero")
        if scale_val < min_scale:
            return (None, "scale_too_small", None) if return_scale else (None, "scale_too_small")

        scale = scale_val
        xy = xy / scale

    # outlier handling (or clamp)
    max_abs = float(np.max(np.abs(xy[valid])))
    if max_abs > max_abs_xy:
        if max_abs > hard_reject_factor * max_abs_xy:
            return (None, "xy_outlier_hard", None) if return_scale else (None, "xy_outlier_hard")
        if clamp_xy:
            xy = np.clip(xy, -max_abs_xy, max_abs_xy)
            reason = "ok_clamped"
        else:
            return (None, "xy_outlier", None) if return_scale else (None, "xy_outlier")
    else:
        reason = "ok"

    out = kp.copy()
    out[:, :2] = xy
    if return_scale:
        return out, reason, float(scale)
    return out, reason