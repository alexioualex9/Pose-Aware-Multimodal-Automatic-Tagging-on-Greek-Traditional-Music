#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from pytorchvideo.data.encoded_video import EncodedVideo
from pytorchvideo.transforms import (
    ApplyTransformToKey,
    Normalize,
    ShortSideScale,
    UniformTemporalSubsample,
)
from sklearn.metrics import accuracy_score, classification_report
from torchvision.transforms import CenterCrop, Compose, Lambda


@dataclass(frozen=True)
class TrainingConfig:
    mean: tuple[float, float, float] = (0.45, 0.45, 0.45)
    std: tuple[float, float, float] = (0.225, 0.225, 0.225)
    frames_per_second: int = 64
    num_frames: int = 32
    sampling_rate: int = 2
    crop_size: int = 224
    side_size: int = 256

    dataset_size: int = 230
    val_size: int = 40
    test_size: int = 30

    num_classes: int = 2
    dropout: float = 0.1
    hidden_dim: int = 32

    special_two_clip_ids: tuple[int, ...] = (12, 95, 97, 147)

    @property
    def clip_duration(self) -> float:
        return (self.num_frames * self.sampling_rate) / self.frames_per_second


class DanceDetectionModel(nn.Module):
    """Fine-tuning wrapper for MViT."""

    def __init__(
        self,
        num_classes: int = 2,
        hidden_dim: int = 32,
        dropout: float = 0.1,
        pretrained: bool = True,
        freeze_backbone: bool = True,
    ) -> None:
        super().__init__()

        self.backbone = torch.hub.load(
            "facebookresearch/pytorchvideo",
            model="mvit_base_32x3",
            pretrained=pretrained,
        )

        if freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False

            # Keep the backbone classification head trainable, όπως στο αρχικό σου script
            for param in self.backbone.head.parameters():
                param.requires_grad = True

        self.classifier = nn.Sequential(
            nn.Linear(400, hidden_dim),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        features = self.backbone(inputs)
        logits = self.classifier(features)
        return logits


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Fine-tune an MViT model for dance-scene detection."
    )

    parser.add_argument("--video-dir", type=Path, required=True, help="Directory with input .mp4 files.")
    parser.add_argument("--labels-file", type=Path, required=True, help="Text file with one label per line.")
    parser.add_argument("--output-dir", type=Path, required=True, help="Directory for checkpoints and logs.")

    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--patience", type=int, default=3)
    parser.add_argument("--seed", type=int, default=0)

    parser.add_argument(
        "--mode",
        choices=["train", "validate", "test", "full"],
        default="full",
        help="train: train+validate, validate: validation only, test: test only, full: train+validate+test",
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=None,
        help="Optional checkpoint (.pt) to resume from or evaluate.",
    )
    parser.add_argument(
        "--device",
        choices=["auto", "cpu", "cuda"],
        default="auto",
        help="Computation device.",
    )
    parser.add_argument(
        "--unfreeze-backbone",
        action="store_true",
        help="Train the whole backbone instead of freezing it.",
    )

    return parser.parse_args()


def resolve_device(device_arg: str) -> torch.device:
    if device_arg == "cpu":
        return torch.device("cpu")
    if device_arg == "cuda":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def set_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def build_transform(config: TrainingConfig):
    return ApplyTransformToKey(
        key="video",
        transform=Compose(
            [
                UniformTemporalSubsample(config.num_frames),
                Lambda(lambda x: x / 255.0),
                Normalize(config.mean, config.std),
                ShortSideScale(size=config.side_size),
                CenterCrop(config.crop_size),
            ]
        ),
    )


def numeric_sort_key(path: Path):
    try:
        return int(path.stem)
    except ValueError:
        return path.stem


def load_video_paths(video_dir: Path) -> list[Path]:
    return sorted(video_dir.glob("*.mp4"), key=numeric_sort_key)


def load_labels(labels_file: Path) -> list[int]:
    with labels_file.open("r", encoding="utf-8") as file:
        return [int(line.strip()) for line in file if line.strip()]


def split_indices(
    dataset_size: int,
    val_size: int,
    test_size: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    indices = np.arange(dataset_size)
    rng.shuffle(indices)

    test_indices = indices[:test_size]
    val_indices = indices[test_size:test_size + val_size]
    train_indices = indices[test_size + val_size:]

    return train_indices, val_indices, test_indices


def get_num_clips_for_video(video_path: Path, config: TrainingConfig, fallback_idx: int) -> int:
    try:
        video_id = int(video_path.stem)
    except ValueError:
        video_id = fallback_idx + 1

    return 2 if video_id in config.special_two_clip_ids else 3


def run_step(
    clip_data: dict,
    label: int,
    model: nn.Module,
    loss_function: nn.Module,
    device: torch.device,
    optimizer: torch.optim.Optimizer | None = None,
) -> tuple[float, int]:
    inputs = clip_data["video"].to(device).unsqueeze(0)
    targets = torch.tensor([label], dtype=torch.long, device=device)

    if optimizer is None:
        with torch.no_grad():
            logits = model(inputs)
            loss = loss_function(logits, targets)
    else:
        optimizer.zero_grad()
        logits = model(inputs)
        loss = loss_function(logits, targets)
        loss.backward()
        optimizer.step()

    prediction = int(torch.argmax(logits, dim=1).item())
    return float(loss.item()), prediction


def run_split(
    split_name: str,
    indices: np.ndarray,
    video_paths: list[Path],
    labels: list[int],
    model: nn.Module,
    transform,
    config: TrainingConfig,
    loss_function: nn.Module,
    device: torch.device,
    optimizer: torch.optim.Optimizer | None = None,
) -> dict:
    is_training = optimizer is not None
    model.train(is_training)

    total_loss = 0.0
    total_clips = 0
    gold_labels: list[int] = []
    predicted_labels: list[int] = []

    for dataset_idx in indices:
        video_path = video_paths[int(dataset_idx)]
        label = labels[int(dataset_idx)]

        num_clips = get_num_clips_for_video(video_path, config, int(dataset_idx))
        encoded_video = EncodedVideo.from_path(str(video_path))

        start_time = 0.0
        for _ in range(num_clips):
            end_time = start_time + config.clip_duration
            clip_data = encoded_video.get_clip(start_sec=start_time, end_sec=end_time)

            if clip_data.get("video") is None:
                break

            clip_data = transform(clip_data)
            loss_value, prediction = run_step(
                clip_data=clip_data,
                label=label,
                model=model,
                loss_function=loss_function,
                device=device,
                optimizer=optimizer,
            )

            total_loss += loss_value
            total_clips += 1
            gold_labels.append(label)
            predicted_labels.append(prediction)

            start_time = end_time

    average_loss = total_loss / total_clips if total_clips > 0 else 0.0
    accuracy = accuracy_score(gold_labels, predicted_labels) if gold_labels else 0.0
    report = (
        classification_report(gold_labels, predicted_labels, digits=4, zero_division=0)
        if gold_labels
        else "No clips were processed."
    )

    print(f"\n[{split_name}]")
    print(f"Processed clips: {total_clips}")
    print(f"Average loss: {average_loss:.6f}")
    print(f"Accuracy: {accuracy:.6f}")
    print(report)

    return {
        "loss": average_loss,
        "accuracy": accuracy,
        "report": report,
        "gold": gold_labels,
        "predictions": predicted_labels,
    }


def save_checkpoint(
    checkpoint_path: Path,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    best_val_loss: float,
) -> None:
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "epoch": epoch,
            "best_val_loss": best_val_loss,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
        },
        checkpoint_path,
    )


def load_checkpoint(
    checkpoint_path: Path,
    model: nn.Module,
    optimizer: torch.optim.Optimizer | None,
    device: torch.device,
) -> tuple[int, float]:
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])

    if optimizer is not None and "optimizer_state_dict" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

    epoch = int(checkpoint.get("epoch", 0))
    best_val_loss = float(checkpoint.get("best_val_loss", float("inf")))
    return epoch, best_val_loss


def main() -> None:
    args = parse_args()
    config = TrainingConfig()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    with (args.output_dir / "training_config.json").open("w", encoding="utf-8") as file:
        json.dump(asdict(config), file, indent=2)

    set_seed(args.seed)
    device = resolve_device(args.device)
    transform = build_transform(config)

    video_paths = load_video_paths(args.video_dir)
    labels = load_labels(args.labels_file)

    if len(video_paths) != len(labels):
        raise ValueError(
            f"Mismatch between number of videos ({len(video_paths)}) and labels ({len(labels)})."
        )

    if len(video_paths) != config.dataset_size:
        raise ValueError(
            f"Expected dataset_size={config.dataset_size}, but found {len(video_paths)} videos."
        )

    train_indices, val_indices, test_indices = split_indices(
        dataset_size=config.dataset_size,
        val_size=config.val_size,
        test_size=config.test_size,
        seed=args.seed,
    )

    model = DanceDetectionModel(
        num_classes=config.num_classes,
        hidden_dim=config.hidden_dim,
        dropout=config.dropout,
        pretrained=True,
        freeze_backbone=not args.unfreeze_backbone,
    ).to(device)

    optimizer = torch.optim.Adam(
        (param for param in model.parameters() if param.requires_grad),
        lr=args.learning_rate,
    )
    loss_function = nn.CrossEntropyLoss().to(device)

    start_epoch = 0
    best_val_loss = float("inf")
    best_checkpoint_path = args.output_dir / "best_model.pt"

    if args.checkpoint is not None:
        start_epoch, best_val_loss = load_checkpoint(
            checkpoint_path=args.checkpoint,
            model=model,
            optimizer=optimizer if args.mode in {"train", "full"} else None,
            device=device,
        )
        print(f"Loaded checkpoint from: {args.checkpoint}")

    if args.mode in {"train", "full"}:
        epochs_without_improvement = 0

        for epoch in range(start_epoch, start_epoch + args.epochs):
            print(f"\n========== Epoch {epoch + 1} ==========")

            train_metrics = run_split(
                split_name="TRAIN",
                indices=train_indices,
                video_paths=video_paths,
                labels=labels,
                model=model,
                transform=transform,
                config=config,
                loss_function=loss_function,
                device=device,
                optimizer=optimizer,
            )

            val_metrics = run_split(
                split_name="VALIDATION",
                indices=val_indices,
                video_paths=video_paths,
                labels=labels,
                model=model,
                transform=transform,
                config=config,
                loss_function=loss_function,
                device=device,
                optimizer=None,
            )

            if val_metrics["loss"] < best_val_loss:
                best_val_loss = val_metrics["loss"]
                epochs_without_improvement = 0
                save_checkpoint(
                    checkpoint_path=best_checkpoint_path,
                    model=model,
                    optimizer=optimizer,
                    epoch=epoch + 1,
                    best_val_loss=best_val_loss,
                )
                print("Validation improvement: checkpoint saved.")
            else:
                epochs_without_improvement += 1
                print(f"No improvement for {epochs_without_improvement} epoch(s).")

            if epochs_without_improvement >= args.patience:
                print("Early stopping triggered.")
                break

        if best_checkpoint_path.exists():
            load_checkpoint(
                checkpoint_path=best_checkpoint_path,
                model=model,
                optimizer=None,
                device=device,
            )
            print(f"Loaded best checkpoint from: {best_checkpoint_path}")

    elif args.mode == "validate":
        run_split(
            split_name="VALIDATION",
            indices=val_indices,
            video_paths=video_paths,
            labels=labels,
            model=model,
            transform=transform,
            config=config,
            loss_function=loss_function,
            device=device,
            optimizer=None,
        )

    if args.mode in {"test", "full"}:
        run_split(
            split_name="TEST",
            indices=test_indices,
            video_paths=video_paths,
            labels=labels,
            model=model,
            transform=transform,
            config=config,
            loss_function=loss_function,
            device=device,
            optimizer=None,
        )


if __name__ == "__main__":
    main()