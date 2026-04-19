from pathlib import Path


def write_results(scene_labels, scene_list, output_dir: Path, video_name: str) -> Path:
    """Write frame ranges of detected dance scenes to a text file."""
    output_dir.mkdir(parents=True, exist_ok=True)

    output_path = output_dir / f"results_of_video_{video_name}.txt"

    with output_path.open("w", encoding="utf-8") as file:
        for label, (start_timecode, end_timecode) in zip(scene_labels, scene_list):
            if label == 1:
                start_frame = start_timecode.get_frames()
                end_frame = end_timecode.get_frames() - 1
                file.write(f"{start_frame} - {end_frame}\n")

    return output_path