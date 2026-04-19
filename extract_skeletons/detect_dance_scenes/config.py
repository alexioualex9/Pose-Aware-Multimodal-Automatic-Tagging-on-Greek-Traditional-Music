from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class InferenceConfig:
    mean: tuple[float, float, float] = (0.45, 0.45, 0.45)
    std: tuple[float, float, float] = (0.225, 0.225, 0.225)
    frames_per_second: int = 64
    num_frames: int = 32
    sampling_rate: int = 2
    crop_size: int = 224
    side_size: int = 256
    threshold: float = 0.5

    metadata_url: str = "https://raw.githubusercontent.com/pxaris/lyra-dataset/refs/heads/main/data/raw.tsv"
    video_extension: str = ".mp4"

    @property
    def clip_duration(self) -> float:
        return (self.num_frames * self.sampling_rate) / self.frames_per_second