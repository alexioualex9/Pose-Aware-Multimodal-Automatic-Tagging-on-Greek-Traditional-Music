import glob
import os
from collections import Counter
import json
import numpy as np
from create_embeddings.normalization import normalize_single_skeleton_coco17

##########################
# Load Skeletons
##########################

def clip_has_min_frames_from_name(clip_dir_name: str, min_frames: int = 75) -> bool:
    try:
        start_f, end_f = clip_dir_name.split("_")
        start_f = int(start_f)
        end_f = int(end_f)
        n_frames = end_f - start_f + 1
        return n_frames >= int(min_frames)
    except Exception:
        return False

def should_reject_no_scale_pose(
    arr_norm: np.ndarray,
    conf_thresh: float,
    min_shoulder_conf: float,
    min_shoulder_dist_abs: float,
    lsho_idx: int = 5,
    rsho_idx: int = 6,
):
    """
    arr_norm: [J,3] AFTER centering, NO scaling.
    Reject if shoulders unreliable/collapsed.
    """
    if arr_norm is None or arr_norm.ndim != 2 or arr_norm.shape[1] < 3:
        return True, "bad_shape"

    xy = arr_norm[:, :2]
    c = arr_norm[:, 2]
    J = xy.shape[0]
    if not (0 <= lsho_idx < J and 0 <= rsho_idx < J):
        return True, "no_shoulder_idx"

    if min_shoulder_conf > 0.0:
        if not (c[lsho_idx] >= min_shoulder_conf and c[rsho_idx] >= min_shoulder_conf):
            return True, "low_shoulder_conf"

    if (c[lsho_idx] >= conf_thresh) and (c[rsho_idx] >= conf_thresh):
        d = float(np.linalg.norm(xy[lsho_idx] - xy[rsho_idx]))
        if not np.isfinite(d):
            return True, "shoulder_dist_nan"
        if d < float(min_shoulder_dist_abs):
            return True, "shoulder_too_close"
    else:
        return True, "shoulder_below_conf_thresh"

    return False, "ok"


def load_all_skeletons_for_video(
    name,
    json_dir,
    reject_stats: Counter,
    conf_thresh=0.35,
    min_valid_joints=12,
    min_scale=0.15,
    max_abs_xy=200.0,
    min_shoulder_conf=0.35,
    min_shoulder_dist_abs=12,
    min_clip_frames: int = 30,
    already_centered: bool = True,
    do_scale: bool = False,
):
    """
    Loads ALL clip-jsons under:
      merged_results/<name>/<start_end>/<start_end>.json
    but SKIPS clip folders with < min_clip_frames inferred from folder name.

    Returns list of dicts:
      {"t": float, "keypoints": np.ndarray[17,3]} for all surviving frames across clips.
    """
    vid_dir = os.path.join(json_dir, name)
    if not os.path.isdir(vid_dir):
        return []

    all_jsons = glob.glob(os.path.join(vid_dir, "*/*.json"))
    skels = []

    for jpath in all_jsons:
        clip_dir = os.path.basename(os.path.dirname(jpath))
        if not clip_has_min_frames_from_name(clip_dir, min_frames=min_clip_frames):
            if reject_stats is not None:
                reject_stats["clip_too_short"] += 1
            continue

        with open(jpath, "r") as f:
            data = json.load(f)

        for d in data:
            arr = np.array(d.get("keypoints", []), dtype=np.float32)

            if arr.ndim == 1:
                if arr.size % 3 != 0:
                    if reject_stats is not None:
                        reject_stats["bad_flat_shape"] += 1
                    continue
                arr = arr.reshape(-1, 3)

            if arr.ndim != 2 or arr.shape[1] < 2:
                if reject_stats is not None:
                    reject_stats["bad_shape"] += 1
                continue

            if arr.shape[1] == 2:
                conf = np.ones((arr.shape[0], 1), dtype=np.float32)
                arr = np.concatenate([arr, conf], axis=1)

            x = arr[:, 0]
            y = arr[:, 1]
            c = arr[:, 2]
            valid = (c >= conf_thresh) & np.isfinite(x) & np.isfinite(y)

            if int(valid.sum()) < int(min_valid_joints):
                if reject_stats is not None:
                    reject_stats["min_valid_joints"] += 1
                continue

            arr_norm, reason = normalize_single_skeleton_coco17(
                arr,
                conf_thresh=conf_thresh,
                already_centered=already_centered,
                do_scale=do_scale,
                clamp_xy=True,
                max_abs_xy=max_abs_xy,
                min_scale=min_scale,
            )
            if arr_norm is None:
                if reject_stats is not None:
                    reject_stats[reason] += 1
                continue

            # shoulder geometry filter
            rej, rreason = should_reject_no_scale_pose(
                arr_norm,
                conf_thresh=conf_thresh,
                min_shoulder_conf=min_shoulder_conf,
                min_shoulder_dist_abs=min_shoulder_dist_abs,
            )
            if rej:
                if reject_stats is not None:
                    reject_stats[rreason] += 1
                continue

            if arr_norm.shape[0] != 17:
                if reject_stats is not None:
                    reject_stats["not_17_joints"] += 1
                continue

            skels.append({
                "t": float(d.get("t", 0.0)),
                "keypoints": arr_norm,
            })

    skels.sort(key=lambda x: x["t"])
    return skels
