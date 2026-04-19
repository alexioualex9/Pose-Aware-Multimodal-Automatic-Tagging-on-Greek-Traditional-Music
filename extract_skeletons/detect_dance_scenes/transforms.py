from torchvision.transforms import Compose, Lambda, CenterCrop
from pytorchvideo.transforms import ApplyTransformToKey, Normalize, ShortSideScale, UniformTemporalSubsample

from .config import InferenceConfig


def build_video_transform(config: InferenceConfig):
    """Build the preprocessing pipeline for video clips."""
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