import argparse
from pathlib import Path

import torch
from scenedetect import AdaptiveDetector, detect

from .config import InferenceConfig
from .inference import classify_scenes, load_model, predict_scene_clips
from .utils import write_results
from .metadata import fetch_danced_video_ids, list_video_files
from .transforms import build_video_transform


def parse_args():
    parser = argparse.ArgumentParser(description="Dance scene detection pipeline.")
    parser.add_argument("--video-dir", type=Path, required=True, help="Directory with input videos.")
    parser.add_argument("--model-path", type=Path, required=True, help="Path to trained model.")
    parser.add_argument("--output-dir", type=Path, required=True, help="Directory for output text files.")
    return parser.parse_args()


def main():
    args = parse_args()
    config = InferenceConfig()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    transform = build_video_transform(config)

    danced_video_ids = fetch_danced_video_ids(config.metadata_url)
    video_files = list_video_files(args.video_dir, config.video_extension)
    model = load_model(args.model_path, device)

    processed_count = 0

    for video_path in video_files:
        if video_path.stem not in danced_video_ids:
            continue

        processed_count += 1
        print(f"[{processed_count}] Processing: {video_path}")

        scene_list = detect(str(video_path), AdaptiveDetector())
        scene_predictions = predict_scene_clips(
            video_path=video_path,
            scene_list=scene_list,
            model=model,
            transform=transform,
            clip_duration=config.clip_duration,
            threshold=config.threshold,
            device=device,
        )

        scene_labels = classify_scenes(scene_predictions)
        write_results(
            scene_labels=scene_labels,
            scene_list=scene_list,
            output_dir=args.output_dir,
            video_name=video_path.name,
        )


if __name__ == "__main__":
    main()