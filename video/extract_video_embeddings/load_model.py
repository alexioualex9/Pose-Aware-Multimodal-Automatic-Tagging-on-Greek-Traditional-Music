import numpy as np
import torch
import torch.nn as nn
from transformers import TimesformerModel, AutoImageProcessor
from torchvision.models import resnet50, ResNet50_Weights, vit_b_16, ViT_B_16_Weights
from torchvision.models.video import r2plus1d_18
from transformers import VideoMAEImageProcessor, VideoMAEModel

try:
    from torchvision.models.video import R2Plus1D_18_Weights
    _HAS_R2_WEIGHTS = True
except ImportError:
    _HAS_R2_WEIGHTS = False


try:
    import decord
    _HAS_DECORD = True
except Exception:
    _HAS_DECORD = False
    import cv2

# -----------------------------
# SlowFast50
# -----------------------------
def load_pretrained_slowfast_r50(device="cuda"):
    model = torch.hub.load("facebookresearch/pytorchvideo", "slowfast_r50", pretrained=True)
    model = model.eval().to(device)
    return model

def make_backbone_return_embeddings_slowfast50(model: torch.nn.Module) -> int:
    last = model.blocks[-1]
    in_dim = last.proj.in_features
    last.proj = nn.Identity()
    return in_dim


# -----------------------------
# TimesFormer
# -----------------------------
def load_pretrained_timesformer(
    model_name: str,
    device: str = "cuda",
):
    """
    Φορτώνει TimeSformer από HuggingFace:
      - TimesformerModel (χωρίς ταξινομητή)
      - AutoImageProcessor (responsible για resize/crop/normalize)
    """
    print(f"[info] Loading HF TimeSformer model: {model_name}")
    processor = AutoImageProcessor.from_pretrained(model_name)
    model = TimesformerModel.from_pretrained(model_name)
    model = model.eval().to(device)
    return processor, model


def make_backbone_return_embeddings_timesformer(model: nn.Module) -> int:
    """
    Το TimesformerModel ήδη ΔΕΝ έχει classification head (σε αντίθεση με ForVideoClassification).
    Άρα απλά θα παίρνουμε CLS embedding από last_hidden_state.
    """
    # Hidden size από config (συνήθως 768)
    if hasattr(model, "config") and hasattr(model.config, "hidden_size"):
        return int(model.config.hidden_size)
    else:
        raise RuntimeError("Could not infer hidden_size from TimesformerModel.config.")


# -----------------------------
# Resnet50
# -----------------------------
def load_resnet50_imagenet(device="cuda"):
    weights = ResNet50_Weights.IMAGENET1K_V2
    model = resnet50(weights=weights)
    model.fc = nn.Identity()   # 2048-d pooled features
    model.eval().to(device)
    return model, 2048


# -----------------------------
# R(2+1)D
# -----------------------------
def load_pretrained_r2plus1d_18(device="cuda"):
    if _HAS_R2_WEIGHTS:
        weights = R2Plus1D_18_Weights.KINETICS400_V1
        model = r2plus1d_18(weights=weights)
    else:
        model = r2plus1d_18(pretrained=True)
    return model.eval().to(device)

def make_backbone_return_embeddings_r21d(model):
    in_dim = model.fc.in_features
    model.fc = nn.Identity()
    return in_dim


# -----------------------------
# VitB16
# -----------------------------
def load_vitb16_imagenet(device="cuda"):
    weights = ViT_B_16_Weights.IMAGENET1K_V1
    model = vit_b_16(weights=weights)
    # Replace classification head with identity
    if hasattr(model.heads, "head"):
        model.heads.head = nn.Identity()
    else:
        model.heads = nn.Identity()
    model.eval().to(device)
    return model, 768


# -----------------------------
# VideoMAE
# -----------------------------

class VideoMAEWrapper(nn.Module):
    """
    Wrapper για VideoMAE:
    - Διαβάζει frames (decord/cv2) ως HWC uint8 RGB
    - Καλεί σωστά τον VideoMAEImageProcessor (συμβατότητα νέου/παλιού API)
    - Κανονικοποιεί σε (B, T, C, H, W) με C=3
    - Επιστρέφει embedding από **το 9ο block** (hidden_states[9]) με **mean-pooling στα patch tokens**
    """
    def __init__(
        self,
        model_name: str = "MCG-NJU/videomae-base",
        device: str = "cuda",
        layer_idx: int = -1,       # 1-based index του block
        use_cls: bool = False,     # False => mean-pool των patch tokens
    ):
        super().__init__()
        if not _HAS_HF:
            raise RuntimeError("transformers not installed. `pip install transformers`")
        self.processor = VideoMAEImageProcessor.from_pretrained(model_name)
        # ΠΡΟΣΟΧΗ: output_hidden_states=True για να πάρουμε ενδιάμεσα layers
        self.model = VideoMAEModel.from_pretrained(
            model_name,
            output_hidden_states=True
        ).eval().to(device)

        self.device = device
        self.hidden = int(self.model.config.hidden_size)
        self.want_c = int(getattr(self.model.config, "num_channels", 3))

        # Ρυθμίσεις επιλογής layer/pooling
        self.layer_idx = int(layer_idx)   # 1..num_blocks (12 για base)
        self.use_cls = bool(use_cls)

    @staticmethod
    def _uniform_indices(n_total: int, n_wanted: int) -> np.ndarray:
        if n_total <= 0 or n_wanted <= 0:
            return np.zeros((0,), dtype=np.int32)
        if n_wanted >= n_total:
            return np.arange(n_total, dtype=np.int32)
        return np.linspace(0, n_total - 1, num=n_wanted).astype(np.int32)

    @torch.no_grad()
    def forward_clip(
        self,
        video_path: str,
        start_sec: float,
        end_sec: float,
        num_frames: int,
        dtype: str = "fp16",
        mode: str = "eval",
    ):
        # 1) σύνολο frames & fps
        if _HAS_DECORD:
            decord.bridge.set_bridge("native")
            vr = decord.VideoReader(video_path)
            total = len(vr)
            try:
                fps = float(vr.get_avg_fps())
            except Exception:
                fps = None
        else:
            cap = cv2.VideoCapture(video_path)
            total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            fps = cap.get(cv2.CAP_PROP_FPS)
            cap.release()

        if total <= 0:
            return torch.zeros(self.hidden)

        # 2) sec -> frame indices (uniform στο [start,end])
        if fps and fps > 0:
            s_idx = int(round(max(0.0, start_sec) * fps))
            e_idx = int(round(max(start_sec, end_sec) * fps))
            s_idx = max(0, min(s_idx, max(total - 1, 0)))
            e_idx = max(s_idx + 1, min(e_idx, total))
            seg_len = max(1, e_idx - s_idx)
            idxs = s_idx + self._uniform_indices(seg_len, num_frames)
        else:
            # fallback: uniform σε όλο το video
            idxs = self._uniform_indices(total, num_frames)

        if idxs.size == 0:
            return torch.zeros(self.hidden)

        # 3) διάβασμα frames ως HWC uint8 RGB
        if _HAS_DECORD:
            idxs_clip = np.clip(idxs, 0, total - 1).tolist()
            batch = vr.get_batch(idxs_clip)            # (T,H,W,3) uint8
            frames = batch.asnumpy()
        else:
            cap = cv2.VideoCapture(video_path)
            tot = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            out = []
            for i in idxs:
                j = int(np.clip(i, 0, max(tot - 1, 0)))
                cap.set(cv2.CAP_PROP_POS_FRAMES, j)
                ok, frame = cap.read()
                if not ok:
                    cap.release()
                    return torch.zeros(self.hidden)
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  # HWC uint8
                out.append(frame)
            cap.release()
            frames = np.stack(out, 0) if out else None
            if frames is None:
                return torch.zeros(self.hidden)

        # safety: dtype / channels
        if frames.dtype != np.uint8:
            vmax = float(frames.max()) if frames.size else 255.0
            if vmax <= 1.0:
                frames = (frames * 255.0).clip(0, 255).astype(np.uint8)
            else:
                frames = np.clip(frames, 0, 255).astype(np.uint8)
        if frames.ndim != 4:  # (T,H,W,C)
            return torch.zeros(self.hidden)
        if frames.shape[-1] == 1:
            frames = np.repeat(frames, 3, axis=3)
        elif frames.shape[-1] > 3:
            frames = frames[..., :3]

        # 4) processor call: νέο/παλιό API
        frames_list = [np.ascontiguousarray(fr) for fr in frames]  # len=T, κάθε ένα HWC uint8
        try:
            # ΝΕΟ API (videos=...)
            batch = self.processor(videos=[frames_list], return_tensors="pt")
        except TypeError:
            # ΠΑΛΙΟ API (positional images)
            batch = self.processor([frames_list], return_tensors="pt")

        pv = batch["pixel_values"]  # (B,T,C,H,W) ή (B,C,T,H,W) ή (T,C,H,W)

        # 5) Κανονικοποίηση σε (B, T, C, H, W) + C=3
        if pv.ndim == 4:  # (T,C,H,W) -> (1,T,C,H,W)
            pv = pv.unsqueeze(0)

        if pv.ndim != 5:
            return torch.zeros(self.hidden)

        # Αν είναι (B,C,T,H,W) -> (B,T,C,H,W)
        if pv.shape[1] == self.want_c and pv.shape[2] != self.want_c:
            pv = pv.permute(0, 2, 1, 3, 4)
        # Αν είναι (B,T,H,W,C) -> (B,T,C,H,W)
        elif pv.shape[-1] == self.want_c and pv.shape[2] != self.want_c:
            pv = pv.permute(0, 1, 4, 2, 3)

        # Τελικός έλεγχος καναλιών στον άξονα 2
        if pv.shape[2] != self.want_c:
            if pv.shape[2] == 1 and self.want_c == 3:
                pv = pv.repeat(1, 1, 3, 1, 1)
            elif pv.shape[2] > 3 and self.want_c == 3:
                pv = pv[:, :, :3]
            else:
                return torch.zeros(self.hidden)

        pixel_values = pv.to(self.device)  # (B,T,C,H,W), C=3

        # 6) Forward (με hidden states)
        use_amp = dtype in ("fp16", "bf16") and self.device.startswith("cuda")
        amp_dtype = torch.bfloat16 if dtype == "bf16" else torch.float16
        ctx = torch.autocast(device_type="cuda", dtype=amp_dtype) if use_amp else torch.cuda.amp.autocast(False)
        with ctx:
            out = self.model(pixel_values=pixel_values)

        # 7) Πάρε το τελευταιο block (1-based → hidden_states[-1])
        hs = out.hidden_states  # len = num_blocks+1, hs[0] = embeddings πριν τα blocks
        if not (1 <= self.layer_idx < len(hs)):
            # safety: αν layer_idx εκτός ορίων, πάρε το τελευταίο block
            layer_idx = len(hs) - 1
        else:
            layer_idx = self.layer_idx

        tokens = hs[layer_idx]                 # (B, N_tokens, D)
        # 8) Pooling: mean πάνω στα patch tokens (αγνοούμε το CLS)
        if self.use_cls:
            emb = tokens[:, 0, :].squeeze(0)     # CLS
        else:
            emb = tokens[:, 1:, :].mean(dim=1).squeeze(0)  # mean των patch tokens

        # 9) dtype cast & return
        target_dtype = {
            "fp16": torch.float16,
            "bf16": torch.bfloat16
        }.get(dtype, torch.float32)
        return emb.to(target_dtype)


def load_pretrained_videomae(model_name: str = "MCG-NJU/videomae-base", device="cuda"):
    processor = VideoMAEImageProcessor.from_pretrained(model_name)
    model = VideoMAEModel.from_pretrained(model_name, output_hidden_states=True)
    model = model.eval().to(device)
    emb_dim = int(model.config.hidden_size)
    return processor, model, emb_dim

