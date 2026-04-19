#!/usr/bin/env python3
"""
Main pipeline for extracting normalized pose sequences from dancing videos.

Pipeline overview
-----------------
1. Clean temporary working directories.
2. Retrieve the list of dancing videos from the Lyra metadata.
3. Trim each video into retained dancing segments.
4. Run ByteTrack on each segment.
5. Select the best detection per frame.
6. Crop the detected subject and store the crops as a temporary video.
7. Run AlphaPose on the cropped segment video.
8. Map local detections back to the global video timeline.
9. Save the merged, normalized pose results as JSON.
"""

import json
import os

import pandas as pd

from extract_skeletons.alphapose import apply_alphapose, normalize_skeleton
from extract_skeletons.bytetrack import bytetrack, get_results_bytetrack, select_best_per_frame
from extract_skeletons.clean_dirs import delete_folders, delete_files, delete_folders2
from extract_skeletons.crop_frame import crop_frame, store_frames
from extract_skeletons.read_video import read_frames
from extract_skeletons.trim_video import trim_video
from extract_skeletons.utils import video_dancing, get_vids_server, prop_type
from extract_skeletons.config import (
    ALPHAPOSE_OUTPUT_DIR,
    BYTETRACK_DIR,
    COLUMNS,
    fps,
    frames_dir,
)


def main():
    """
    Execute the full pose-extraction pipeline on all dancing videos.
    """

    delete_folders()

    danced_vids = video_dancing()
    mp4_files = get_vids_server()

    for vid in mp4_files:
        name = vid[:-4]

        if name not in danced_vids:
            continue

        # Trim the video into retained dancing segments.
        segment_info = trim_video(vid)

        if not segment_info:
            delete_files(name)
            delete_folders2()
            continue

        # Run ByteTrack on each retained segment.
        bytetrack(vid, len(segment_info))

        txt_files = sorted(get_results_bytetrack())

        if len(txt_files) != len(segment_info):
            print(
                f"Warning: {vid} -> {len(segment_info)} retained segments, "
                f"but {len(txt_files)} ByteTrack result files were found."
            )

        for meta, txt in zip(segment_info, txt_files):
            seg_idx = meta["seg_idx"]
            seg_start_frame = meta["start_frame"]
            seg_end_frame = meta["end_frame"]

            # 1) Load and clean ByteTrack output.
            path = os.path.join(
                BYTETRACK_DIR,
                "YOLOX_outputs",
                "yolox_x_mix_det",
                "track_vis",
                txt,
            )
            df = pd.read_csv(path, header=None, names=COLUMNS)
            df = prop_type(df)

            # 2) Load all frames for this segment.
            frames = read_frames(
                os.path.join(frames_dir, f"{seg_idx}_trimmed_{vid}")
            )

            # 3) Select one best detection per frame.
            best = select_best_per_frame(df, len(frames))
            if best.empty:
                continue

            best_valid = best.dropna(subset=["x", "y", "w", "h"]).copy()
            best_valid[["x", "y", "w", "h"]] = best_valid[
                ["x", "y", "w", "h"]
            ].astype(int)

            # 4) Crop the detected subject from each valid frame.
            crops = []
            for _, det in best_valid.iterrows():
                frame_idx = int(det["frame"]) - 1
                bbox = (
                    int(det["x"]),
                    int(det["y"]),
                    int(det["w"]),
                    int(det["h"]),
                )
                crops.append(crop_frame(frames[frame_idx], bbox))

            if not crops:
                continue

            # 5) Store the cropped sequence as a temporary video.
            crop_video = store_frames(crops, name, seg_idx)

            # 6) Run AlphaPose on the cropped video.
            poses = apply_alphapose(crop_video)

            # 7) Merge local detections into the global video timeline.
            scene_keypoints = []
            for det_row, pose in zip(best_valid.itertuples(), poses):
                local_frame = int(det_row.frame)

                # local_frame is 1-based inside the trimmed segment
                global_frame = seg_start_frame + (local_frame - 1)
                t_sec = global_frame / fps

                norm_kp = normalize_skeleton(
                    pose["keypoints"],
                    hip_j1=11,
                    hip_j2=12,
                )

                scene_keypoints.append(
                    {
                        "frame": global_frame,
                        "t": round(t_sec, 3),
                        "keypoints": norm_kp,
                    }
                )

            # 8) Store merged JSON using frame-accurate segment naming.
            out_dir = os.path.join(
                ALPHAPOSE_OUTPUT_DIR,
                "merged_results",
                name,
                f"{seg_start_frame}_{seg_end_frame}",
            )
            os.makedirs(out_dir, exist_ok=True)

            out_path = os.path.join(
                out_dir,
                f"{seg_start_frame}_{seg_end_frame}.json",
            )

            with open(out_path, "w", encoding="utf-8") as fout:
                json.dump(scene_keypoints, fout, indent=2)

        delete_files(name)
        delete_folders2()

if __name__ == "__main__":
    main()
