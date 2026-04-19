import json
import os
import subprocess

import numpy as np

from config import (
    ALPHAPOSE_DIR,
    A_EXPERIMENT_CONFIG,
    A_CHECKPOINT,
)


def normalize_skeleton(p, hip_j1=11, hip_j2=12):
    """
    Normalize a skeleton by translating it to the hip-centered coordinate system.

    The function recenters the skeleton so that the midpoint of the two hip
    joints lies at the origin of the 2D image plane. Only the first two
    coordinates (x, y) are translated; any additional coordinate(s), such as
    confidence or depth, remain unchanged.

    Parameters
    ----------
    p : array-like
        Skeleton joint coordinates. Expected shape is either:
        - (J, 3): one row per joint, or
        - (3J,): flattened representation that can be reshaped to (-1, 3).

    hip_j1 : int, optional
        Index of the first hip joint. Default is 11.

    hip_j2 : int, optional
        Index of the second hip joint. Default is 12.

    Returns
    -------
    list
        Skeleton coordinates after translation normalization, returned as a
        nested Python list.

    Notes
    -----
    This function performs only translation-based normalization.
    It does not apply scaling, rotation, or temporal smoothing.
    """
    arr = np.asarray(p, dtype=float)

    if arr.ndim == 1:
        arr = arr.reshape(-1, 3)

    hip_midpoint = (arr[hip_j1, :2] + arr[hip_j2, :2]) * 0.5
    arr[:, :2] -= hip_midpoint

    return arr.tolist()


def apply_alphapose(vid_path):
    """
    Run AlphaPose on a video file and return the resulting pose detections.

    This function executes the AlphaPose inference script for a given video,
    stores the output in the directory containing the input video, and loads
    the generated JSON file with the detected poses.

    Parameters
    ----------
    vid_path : str
        Path to the input video file.

    Returns
    -------
    list
        Parsed contents of ``alphapose-results.json``, typically a list of pose
        detections produced by AlphaPose.

    Raises
    ------
    FileNotFoundError
        If the expected AlphaPose JSON output file is not found.

    subprocess.CalledProcessError
        If the AlphaPose subprocess fails during execution.

    Notes
    -----
    The output JSON is expected to be written to:

        .../<segment_dir>/alphapose-results.json

    where ``segment_dir`` is the parent directory of ``vid_path``.
    """
    segment_dir = os.path.dirname(vid_path)
    os.makedirs(segment_dir, exist_ok=True)

    cmd = [
        "python",
        "scripts/demo_inference.py",
        "--cfg",
        A_EXPERIMENT_CONFIG,
        "--checkpoint",
        A_CHECKPOINT,
        "--video",
        vid_path,
        "--outdir",
        segment_dir,
    ]

    subprocess.run(cmd, cwd=ALPHAPOSE_DIR, check=True)

    json_path = os.path.join(segment_dir, "alphapose-results.json")
    if not os.path.exists(json_path):
        raise FileNotFoundError(f"Expected output JSON not found: {json_path}")

    with open(json_path, "r", encoding="utf-8") as f:
        poses = json.load(f)

    return poses