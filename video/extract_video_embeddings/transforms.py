import torch
from torchvision import transforms
from torchvision.models import resnet50, ResNet50_Weights
from torchvision.models import vit_b_16, ViT_B_16_Weights
from torchvision.models.video import R2Plus1D_18_Weights
from torchvision.transforms import Compose, Lambda, RandomCrop, RandomHorizontalFlip
from torchvision.transforms._transforms_video import CenterCropVideo, NormalizeVideo
from torchvision.transforms import InterpolationMode
from pytorchvideo.transforms import (
    ApplyTransformToKey,
    RandomShortSideScale,
    ShortSideScale,
    UniformTemporalSubsample,
)


class PackPathway(torch.nn.Module):
    """Prepare inputs for SlowFast: fast pathway T, slow pathway T/alpha."""
    def __init__(self, alpha: int = 4):
        super().__init__()
        self.alpha = alpha
    def forward(self, frames):
        # frames: (C, T, H, W)
        fast = frames
        idx = torch.linspace(0, frames.shape[1]-1, max(frames.shape[1] // self.alpha, 1)).long()
        slow = torch.index_select(frames, 1, idx)
        return [slow, fast]


def build_video_transforms(model_name, num_frames: int, mode: str, embs: str = "frozen"):
    """
    TRAIN: RandomShortSideScale(256..320) -> RandomCrop(256) -> Flip
    EVAL : ShortSideScale(256) -> CenterCrop(256)
    Normalize: video mean/std (0.45 / 0.225)
    """

    if mode == "train":
        spatial = Compose([
            RandomShortSideScale(min_size=256, max_size=320),
            RandomCrop(224),
            RandomHorizontalFlip(p=0.5),
        ])
    else:
        spatial = Compose([
            ShortSideScale(256),
            CenterCropVideo(224),
        ])


    if model_name == "slowfast50":
        pack = PackPathway(alpha=4)
        mean, std = [0.45]*3, [0.225]*3
        return ApplyTransformToKey(
           key="video",
           transform=Compose([
               UniformTemporalSubsample(num_frames),   # N frames uniformly in [start,end]
               Lambda(lambda x: x / 255.0),
               NormalizeVideo(mean, std),
               spatial,
               pack,
           ]))

    elif model_name == "r21d":
       if embs == "finetuned":
           r21d_weights = R2Plus1D_18_Weights.KINETICS400_V1
           mean, std = list(r21d_weights.transforms().mean), list(r21d_weights.transforms().std)
           return ApplyTransformToKey(
               key="video",
               transform=Compose([
                   UniformTemporalSubsample(num_frames),
                   Lambda(lambda x: x / 255.0),
                   NormalizeVideo(mean, std),
                   spatial,
               ]),
           )
       elif embs == "frozen":
          weights = R2Plus1D_18_Weights.KINETICS400_V1
          return weights.transforms()
       else:
          raise ValueError(f"Unsupported type of embeddings: {embs}")

    else:
       raise ValueError(f"Unsupported model_name: {model_name}")


def build_image_transforms_train(model_name):
    """
    TRAIN: Resize(short=256) -> RandomCrop(224) -> Flip
    Normalize: mean/std
    """

    # Load Weights according to model used
    if model_name == "resnet50":
       weights = ResNet50_Weights.DEFAULT
    elif model_name == "vitb16":
       weights = ViT_B_16_Weights.DEFAULT
    else:
       raise ValueError(f"Unsupported model_name: {model_name} for image based transform in train mode")

    # Get mean and std
    preset = weights.transforms()
    mean = preset.mean
    std  = preset.std

    # Define Transform
    t = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize(256, interpolation=InterpolationMode.BILINEAR),
        transforms.RandomCrop(224),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
        ])
    return t


def build_image_transforms_eval(model_name):
    if model_name == "resnet50":
       weights = ResNet50_Weights.DEFAULT
    elif model_name == "vitb16":
       weights = ViT_B_16_Weights.DEFAULT
    else:
       raise ValueError(f"Unsupported model_name: {model_name} for image based transform in eval mode")

    return weights.transforms()
