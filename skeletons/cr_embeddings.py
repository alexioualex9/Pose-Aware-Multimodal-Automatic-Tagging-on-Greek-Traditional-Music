import json
import os
import numpy as np
import torch
from tqdm import tqdm
from collections import Counter
from config import DATA_DIR, SKELETONS_CONFIG, LABELS_DIR, LABELS_SUBSET_DIR, VIDEO_DIR, OUT_DIR
from create_embeddings.load_skeletons import load_all_skeletons_for_video
from create_embeddings.interpolation import interpolate_short_gaps
from create_embeddings.skel_similarity import filter_by_self_similarity_medoid
from create_embeddings.quality_scores import compute_quality_scores
from create_embeddings.utils import (build_label_filter,
                   select_spread_skeletons,
                   select_best_spread_by_quality,
                   sample_or_pad_frames,
                  )

def build_npz(args):


    video_index = os.path.join(VIDEO_DIR, "extract_video_embeddings/embeddings/lyra/slowfast50", args.set, "index.json")
    out_dir = os.path.join(OUT_DIR, args.dataset, args.time_window, args.model_name, args.set)

    os.makedirs(out_dir, exist_ok=True)

    config = SKELETONS_CONFIG[args.dataset].copy()
    config['model_name'] = args.model_name
    config['dataset'] = args.dataset
    config['device'] = torch.device(args.device)

    # Load Labels
    labels = json.load(open(LABELS_SUBSET_DIR))["labels"]
    labels_full = json.load(open(LABELS_DIR))["labels"]
    _, keep_indices = build_label_filter(labels_full)

    # Read video index info
    with open(video_index, "r") as f:
        video_rows = json.load(f)

    cache = {}
    out_index = []
    eps = 1e-3

    reject_stats_global = Counter()
    clip_skipped_no_frames = 0
    clip_skipped_after_sim = 0

    for r in tqdm(video_rows, desc="Building skeleton clips (C=6, ST-GCN-like)"):
        vid = r["video_id"]
        start_s = float(r["start_sec"])
        end_s = float(r["end_sec"])
        labels_vec = r.get("labels", [])

        if labels_vec:
            if keep_indices is not None:
                labels_vec = [labels_vec[i] for i in keep_indices]
        else:
            labels_vec = [0] * len(labels)

        # load skeletons once per video_id (robust) — skipping short clip folders
        if vid not in cache:
            reject_stats_local = Counter()
            cache[vid] = load_all_skeletons_for_video(
                vid,
                DATA_DIR,
                reject_stats=reject_stats_local,
                conf_thresh= config['conf_thresh'],
                min_valid_joints= config['min_valid_joints'],
                min_scale=config['min_scale'],
                max_abs_xy=config['max_abs_xy'],
                min_shoulder_conf= config['min_shoulder_conf'],
                min_shoulder_dist_abs= config['min_shoulder_dist_abs'],
                min_clip_frames= config['min_clip_frames'],
                already_centered= config['already_centered'],  # NEW
                do_scale= config['do_scale'],                  # NEW
            )
            reject_stats_global.update(reject_stats_local)

        skels = cache[vid]
        if not skels:
            continue

        # clip window by time
        clip_skels = [s for s in skels if (start_s - eps) <= s["t"] < (end_s + eps)]
        if len(clip_skels) == 0:
            clip_skipped_no_frames += 1
            continue

        # self-similarity filter (medoid/Tukey)
        clip_skels, _ = filter_by_self_similarity_medoid(
            clip_skels,
            conf_thresh= config['conf_thresh'],
            joint_ids=(5, 6, 11, 12),
            y_weight= config['y_weight'],
            tukey_k= config['sim_tukey_k'],
            min_keep= config['sim_min_keep'],
        )
        if len(clip_skels) < max(2, config['sim_min_keep']):
            clip_skipped_after_sim += 1
            continue

        # interpolate short gaps
        kps_seq = [s["keypoints"] for s in clip_skels]
        kps_interp = interpolate_short_gaps(
            kps_seq,
            conf_thresh= config['conf_thresh'],
            max_gap= config['interp_max_gap'],
            interpolate_conf=False,
        )
        for s, kp_new in zip(clip_skels, kps_interp):
            s["keypoints"] = kp_new

        # selection:
        clip_skels = select_spread_skeletons(clip_skels, config['max_skeletons'])

        compute_quality_scores(
            clip_skels,
            conf_thresh= config['conf_thresh'],
            w_bone= config['w_bone'],
            w_jitter= config['w_jitter'],
            w_lr= config['w_lr'],
        )

        clip_skels = select_best_spread_by_quality(clip_skels, M=config['T'])

        frames = np.stack([s["keypoints"] for s in clip_skels], axis=0).astype(np.float32)  # [Tsel,V,3]
        frames, valid_len, pad = sample_or_pad_frames(frames, config['T'])                       # [T,V,3]

        # pos + vel
        pos = frames.copy()  # [T,V,3] (x,y,conf)
        vel = np.zeros_like(pos, dtype=np.float32)
        vel[1:, :, :2] = pos[1:, :, :2] - pos[:-1, :, :2]
        vel[:, :, 2] = pos[:, :, 2]  # or 0.0 if you want pure motion

        data = np.concatenate(
            [pos.transpose(2, 0, 1), vel.transpose(2, 0, 1)],
            axis=0
        ).astype(np.float32)  # [6,T,V]


        # Define name of skeleton embedding
        start_sample = r["start_sample"]
        end_sample   = r["end_sample"]
        fname = f"{vid}_s{start_sample}_e{end_sample}.npz"
        out_path = os.path.join(out_dir, fname)


        # Save embeddings
        np.savez_compressed(
            out_path,
            data=data,
            valid_len=np.int32(valid_len),
            pad=np.int32(pad),
        )

        # Save info in index.json file
        out_index.append({
            "blob": str(os.path.abspath(out_path)),
            "video_id": vid,
            "start_sec": start_s,
            "end_sec": end_s,
            "labels": labels_vec,
            "T": int(config['T']),
            "C": int(data.shape[0]),
            "V": int(data.shape[2]),
            "valid_len": int(valid_len),
            "pad": int(pad),
            "has_skeleton": 1,

            # store config for traceability
            "conf_thresh": float(config['conf_thresh']),
            "min_valid_joints": int(config['min_valid_joints']),
            "max_skeletons": int(config['max_skeletons']),
            "min_scale": float(config['min_scale']),
            "max_abs_xy": float(config['max_abs_xy']),
            "min_shoulder_conf": float(config['min_shoulder_conf']),
            "min_shoulder_dist_abs": float(config['min_shoulder_dist_abs']),
            "min_clip_frames": int(config['min_clip_frames']),
            "sim_tukey_k": float(config['sim_tukey_k']),
            "sim_min_keep": int(config['sim_min_keep']),
            "interp_max_gap": int(config['interp_max_gap']),
            "w_bone": float(config['w_bone']),
            "w_jitter": float(config['w_jitter']),
            "w_lr": float(config['w_lr']),
            "already_centered": bool(config['already_centered']),
            "do_scale": bool(config['do_scale']),
        })

    out_index_path = os.path.join(out_dir, "skeleton_index.json")
    with open(out_index_path, "w") as f:
        json.dump(out_index, f, indent=2, ensure_ascii=False)

    print("\n================ STATS ================")
    print(f"[done] Saved {len(out_index)} clips to: {out_index_path}")
    print(f"[info] Clips skipped (no frames in time window after filtering): {clip_skipped_no_frames}")
    print(f"[info] Clips skipped (too few after self-similarity): {clip_skipped_after_sim}")
    print("[reject reasons] (frame/clip-level):")
    for k, v in reject_stats_global.most_common():
        print(f"  {k:24s}: {v}")
    print("=======================================")
