from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from pytorchvideo.data.encoded_video import EncodedVideo


def load_model(model_path: Path, device: torch.device):
    """Load a trained model and move it to the requested device."""
    model = torch.load(model_path, map_location=device)
    model = model.to(device)
    model.eval()
    return model


def predict_clip(video_data: dict, model, device: torch.device, threshold: float) -> int:
    """Predict whether a clip contains dance."""
    with torch.no_grad():
        inputs = video_data["video"].to(device)
        logits = model(inputs[None, ...])

        probabilities = F.softmax(logits, dim=1)
        confidence, predicted_class = probabilities.max(dim=1)

        if confidence.item() < threshold:
            return 0

        return int(predicted_class.item())


def predict_scene_clips(
    video_path: Path,
    scene_list,
    model,
    transform,
    clip_duration: float,
    threshold: float,
    device: torch.device,
) -> list[np.ndarray]:
    """
    Predict dance labels for each clip inside each detected scene.
    Returns one numpy array per scene.
    """
    encoded_video = EncodedVideo.from_path(str(video_path))
    scene_predictions: list[np.ndarray] = []

    for start_timecode, end_timecode in scene_list:
        scene_duration = int(np.floor((end_timecode - start_timecode).get_seconds()))

        if scene_duration <= 1:
            scene_predictions.append(np.array([0], dtype=int))
            continue

        current_start = start_timecode.get_seconds()
        clip_predictions: list[int] = []

        for _ in range(scene_duration):
            clip = encoded_video.get_clip(
                start_sec=current_start,
                end_sec=current_start + clip_duration,
            )

            if clip.get("video") is None:
                break

            clip = transform(clip)
            clip_label = predict_clip(clip, model, device, threshold)
            clip_predictions.append(clip_label)

            current_start += clip_duration

        if clip_predictions:
            scene_predictions.append(np.array(clip_predictions, dtype=int))
        else:
            scene_predictions.append(np.array([0], dtype=int))

    return scene_predictions


def classify_scenes(scene_predictions: list[np.ndarray]) -> list[int]:
    """Convert per-clip predictions into one label per scene."""
    scene_labels: list[int] = []

    for predictions in scene_predictions:
        if len(predictions) <= 1:
            scene_labels.append(0)
            continue

        positives = int((predictions == 1).sum())
        label = 1 if positives > len(predictions) // 2 else 0
        scene_labels.append(label)

    return scene_labels