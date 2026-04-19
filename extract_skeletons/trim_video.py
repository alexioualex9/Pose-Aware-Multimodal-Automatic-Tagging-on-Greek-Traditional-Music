import os

import ffmpeg

from extract_skeletons.config import ALPHAPOSE_OUTPUT_DIR, frames_dir, fps, lyra_dir
from extract_skeletons.utils import read_scene_frames


def trim_video(vid):
    """
    Split a video into retained dancing segments and return segment metadata.

    The scene annotations are treated as frame indices (ground truth). Trimming
    is performed using time values in seconds derived from those frame indices.

    Parameters
    ----------
    vid : str
        Input video filename, including the ``.mp4`` suffix.

    Returns
    -------
    list of dict
        Metadata for each retained segment. Each dictionary contains:
        - ``seg_idx``
        - ``orig_idx``
        - ``start_frame``
        - ``end_frame``
        - ``start_sec``
        - ``end_sec``
        - ``duration_sec``

    Notes
    -----
    Segments shorter than 5 seconds are discarded.

    The current implementation preserves the original duration logic:

        duration_frames = end_frame - start_frame

    This matches the behavior of your previous code.
    """
    name = vid[:-4]
    start_frames, end_frames = read_scene_frames(name)

    input_file = os.path.join(lyra_dir, vid)
    segment_info = []

    # Prepare AlphaPose output root for this video.
    video_out_root = os.path.join(ALPHAPOSE_OUTPUT_DIR, name)
    os.makedirs(video_out_root, exist_ok=True)

    for orig_idx, (start_frame, end_frame) in enumerate(
        zip(start_frames, end_frames),
        start=1,
    ):
        duration_frames = end_frame - start_frame
        duration_sec = duration_frames / fps

        if duration_sec < 5.0:
            continue

        seg_idx = len(segment_info) + 1
        start_sec = start_frame / fps
        end_sec = end_frame / fps

        output_file = os.path.join(frames_dir, f"{seg_idx}_trimmed_{vid}")

        (
            ffmpeg
            .input(input_file, ss=start_sec, t=duration_sec)
            .output(output_file)
            .overwrite_output()
            .run(quiet=True)
        )

        os.makedirs(
            os.path.join(video_out_root, str(seg_idx)),
            exist_ok=True,
        )

        segment_info.append(
            {
                "seg_idx": seg_idx,
                "orig_idx": orig_idx,
                "start_frame": start_frame,
                "end_frame": end_frame,
                "start_sec": start_sec,
                "end_sec": end_sec,
                "duration_sec": duration_sec,
            }
        )

    return segment_info
