import sys
import os
import torch
import torch.nn as nn
import numpy as np
from PIL import Image
from pytorchvideo.data.encoded_video import EncodedVideo

from video.extract_video_embeddings.emb_utils import sample_frame_indices, sample_frame_indices_2d, _uniform_subsample_cthw


########################
# Slowfast50
########################
@torch.no_grad()
def extract_clip_embedding_slowfast(
    model,
    transform,
    video_path,
    start_sec,
    end_sec,
    device="cuda",
    dtype="fp16",
):

    video = EncodedVideo.from_path(video_path)
    clip = video.get_clip(start_sec=float(start_sec), end_sec=float(end_sec))
    out = transform(clip)

    if isinstance(out, dict):
        if "video" in out:
            inputs = out["video"]              # [slow, fast]
        elif "slow" in out and "fast" in out:
            inputs = [out["slow"], out["fast"]]
        else:
            raise RuntimeError(f"SlowFast transform returned dict keys={list(out.keys())}, expected video or slow/fast")
    elif isinstance(out, (list, tuple)):
        inputs = list(out)  # assume [slow, fast]
    else:
        raise RuntimeError(f"Unexpected transform output type: {type(out)}")


    inputs = [t.unsqueeze(0).to(device) for t in inputs]  # add batch dim

    use_amp = dtype in ("fp16", "bf16")
    amp_dtype = torch.bfloat16 if dtype == "bf16" else torch.float16
    if use_amp:
        with torch.autocast(device_type=("cuda" if str(device).startswith("cuda") else "cpu"), dtype=amp_dtype):
            feats = model(inputs)
    else:
        feats = model(inputs)

    if feats.ndim > 2:
        feats = torch.flatten(feats, 1)
    feats = feats.squeeze(0).detach().cpu()
    if dtype == "fp16":
        feats = feats.to(torch.float16)
    elif dtype == "bf16":
        emb = emb.to(torch.bfloat16)
    else:
        feats = feats.to(torch.float32)
    return feats  # [D]



########################
# TimesFormer
########################
@torch.no_grad()
def extract_clip_embedding_timesformer(
    model: nn.Module,
    processor,
    video_path: str,
    start_sec: float,
    end_sec: float,
    num_frames: int = 8,
    device: str = "cuda",
    dtype: str = "fp16",
):
    """
    - Διαβάζει τις εικόνες [C,T,H,W] από EncodedVideo.get_clip
    - Uniform temporal subsample σε num_frames
    - Μετατρέπει σε λίστα από HWC uint8 frames
    - HuggingFace transform -> pixel_values [1, T, C, H, W]
    - forward στο TimesformerModel
    - Παίρνουμε CLS embedding από last_hidden_state[:,0,:]  -> [D]
    """
    video = EncodedVideo.from_path(video_path)
    clip = video.get_clip(start_sec=float(start_sec), end_sec=float(end_sec))
    vt = clip["video"]  # (C, T, H, W)

    if vt is None or vt.ndim != 4:
        raise RuntimeError(f"Unexpected video tensor shape for {video_path}: {None if vt is None else vt.shape}")

    vt = torch.as_tensor(vt)  # (C,T,H,W)
    C, T, H, W = vt.shape
    if T <= 0:
        raise RuntimeError(f"Empty video for {video_path}")

    idxs = sample_frame_indices(T, num_frames)
    if idxs.size == 0:
        raise RuntimeError(f"No frame indices for {video_path}")

    frames = vt[:, idxs]  # (C, n, H, W)
    n = frames.shape[1]
    frames_list = []
    for j in range(n):
       frame = frames[:, j].permute(1, 2, 0).cpu().numpy()  # HWC

       # force uint8 0..255
       if frame.dtype != np.uint8:
           maxv = frame.max()
           if maxv <= 1.0:
               frame = (frame * 255.0).round()
           frame = np.clip(frame, 0, 255).astype(np.uint8)

       frames_list.append(np.ascontiguousarray(frame))


    # Transform: try new API (videos=...), else fallback in older.
    try:
        batch = processor(frames_list, return_tensors="pt")
    except TypeError:
        batch = processor([frames_list], return_tensors="pt")

    pixel_values = batch["pixel_values"]  # αναμενόμενο: (1, T, C, H, W) ή (1, C, T, H, W)

    if pixel_values.ndim != 5:
        raise RuntimeError(f"Unexpected pixel_values shape: {pixel_values.shape}")

    # If (1, C, T, H, W) instead of (1, T, C, H, W) then correct it:
    if pixel_values.shape[1] == 3 and pixel_values.shape[2] != 3:
        pixel_values = pixel_values.permute(0, 2, 1, 3, 4)  # (1, T, C, H, W)

    pixel_values = pixel_values.to(device)

    use_amp = dtype in ("fp16", "bf16") and device == "cuda"
    amp_dtype = torch.bfloat16 if dtype == "bf16" else torch.float16

    if use_amp:
        with torch.autocast(
            device_type=("cuda" if str(device).startswith("cuda") else "cpu"),
            dtype=amp_dtype,
        ):
            out = model(pixel_values=pixel_values)
    else:
        out = model(pixel_values=pixel_values)

    # CLS embedding from last_hidden_state
    last = out.last_hidden_state  # (B, N_tokens, D)
    emb = last[:, 0, :]           # (B, D)
    emb = emb.squeeze(0).detach().cpu()

    if dtype == "fp16":
        emb = emb.to(torch.float16)
    elif dtype == "bf16":
        emb = emb.to(torch.bfloat16)
    else:
        emb = emb.to(torch.float32)

    return emb  # [D]



########################
# ResNet50 + VitB/16
########################
@torch.no_grad()
def extract_clip_embedding_2d(
    forward_fn,
    transform,
    video_path,
    start_sec,
    end_sec,
    num_frames=16,
    device="cuda",
    dtype="fp16",
    mode="eval",
    batch=64,
):
    """
    Decode [start,end], TSN-sample frames, apply transforms per-frame,
    forward ResNet-50, mean-pool over time -> [2048]
    """
    ev = EncodedVideo.from_path(video_path)
    clip = ev.get_clip(start_sec=float(start_sec), end_sec=float(end_sec))
    vt = clip["video"]  # (C, T, H, W) uint8
    if vt is None or getattr(vt, "ndim", None) != 4:
        raise RuntimeError("Bad clip decode (vt is None or not 4D)")
    C, T, H, W = vt.shape
    idx = sample_frame_indices_2d(T, num_frames, mode)
    frames = vt[:, idx]  # (C, n, H, W) where n <= num_frames (or == if T>=num_frames)
    n = frames.shape[1]

    feats = []
    use_amp = dtype in ("fp16", "bf16") and str(device).startswith("cuda")
    amp_dtype = (torch.bfloat16 if dtype == "bf16" else torch.float16)
    autocast_ctx = (
        torch.autocast(device_type="cuda", dtype=amp_dtype)
        if use_amp else
        torch.autocast(device_type="cuda", enabled=False)
    )

    with autocast_ctx:
        for i in range(0, n, batch):
            chunk = frames[:, i:i+batch]
            imgs = []
            for j in range(chunk.shape[1]):
                img = chunk[:, j].permute(1, 2, 0).cpu().numpy()
                # PIL needs HWC uint8 (usually)
                if img.dtype != np.uint8:
                   # If 0..1 float, scale to 0..255
                   if img.max() <= 1.0:
                       img = (img * 255.0)
                   img = np.clip(img, 0, 255).astype(np.uint8)

                # safety: ensure 3 channels
                if img.ndim != 3 or img.shape[2] != 3:
                  raise RuntimeError(f"Bad frame shape for PIL: {img.shape}, dtype={img.dtype}")

                pil = Image.fromarray(img, mode="RGB")
                imgs.append(transform(pil))  # Tensor [3,224,224]

            if not imgs:
               break
            xb = torch.stack(imgs, dim=0).to(device)                   # [B,3,224,224]
            f = forward_fn(xb)                                         # [B,2048]
            feats.append(f.detach().cpu())

    if not feats:
        raise RuntimeError("No features produced (empty feats list)")

    feat = torch.cat(feats, dim=0).mean(dim=0)                         # [D]

    if dtype == "fp16":
        feat = feat.to(torch.float16)
    elif dtype == "bf16":
        emb = emb.to(torch.bfloat16)
    else:
        feat = feat.to(torch.float32)
    return feat



########################
# R(2+1)D
########################
@torch.no_grad()
def extract_clip_embedding_r21d(
    model,
    transform,            # R2Plus1D_18_Weights.KINETICS400_V1.transforms()
    video_path,
    start_sec,
    end_sec,
    num_frames=16, # canonical for these weights
    device="cuda",
    dtype="fp16",
):
    video = EncodedVideo.from_path(video_path)
    clip = video.get_clip(start_sec=float(start_sec), end_sec=float(end_sec))
    vt = clip["video"]  # (C,T,H,W)

    if vt is None or vt.ndim != 4:
        raise RuntimeError(f"Bad clip decode: {video_path}")

    vt = torch.as_tensor(vt)

    # (optional but good): ensure uint8 0..255 because weights transforms rescale internally
    if vt.dtype != torch.uint8:
        # if float 0..1 -> scale; if already 0..255 float -> just clamp
        mx = float(vt.max())
        if mx <= 1.0:
            vt = (vt * 255.0).round()
        vt = vt.clamp(0, 255).to(torch.uint8)

    # 1) enforce fixed T
    vt = _uniform_subsample_cthw(vt, num_frames)  # (C,16,H,W)

    # 2) weights.transforms expects (T,C,H,W)
    vt = vt.permute(1, 0, 2, 3)                   # (16,3,H,W)

    # 3) apply official preprocessing -> outputs (C,T,112,112)
    vt = transform(vt)                              # (3,16,112,112)

    if vt.ndim != 4 or vt.shape[0] != 3:
        raise RuntimeError(f"After transform expected (3,T,112,112), got {tuple(vt.shape)}")

    # 4) model expects (B,C,T,H,W)
    inputs = vt.unsqueeze(0).to(device)

    dev = device.type if isinstance(device, torch.device) else str(device)
    use_amp = dtype in ("fp16", "bf16") and dev == "cuda"
    amp_dtype = torch.bfloat16 if dtype == "bf16" else torch.float16

    if use_amp:
        with torch.autocast(device_type="cuda", dtype=amp_dtype):
            feats = model(inputs)
    else:
        feats = model(inputs)

    feats = feats.flatten(1).squeeze(0).detach().cpu()

    if dtype == "fp16":
        feats = feats.to(torch.float16)
    elif dtype == "bf16":
        feats = feats.to(torch.bfloat16)
    else:
        feats = feats.to(torch.float32)

    return feats




########################
# VideoMAE
########################
@torch.no_grad()
def extract_clip_embedding_videomae(
    model,
    processor,
    video_path: str,
    start_sec: float,
    end_sec: float,
    num_frames: int = 16,
    device: str = "cuda",
    dtype: str = "fp16",
    layer_idx: int = -1,     # -1 = last block
    use_cls: bool = False,   # False = mean pool patch tokens
):
    transform = None
    video = EncodedVideo.from_path(video_path)
    clip = video.get_clip(start_sec=float(start_sec), end_sec=float(end_sec))

    # clip["video"] expected (C,T,H,W)
    if transform is not None:
        clip = transform(clip)

    vt = clip["video"]
    if vt is None or vt.ndim != 4:
        raise RuntimeError(f"Bad clip decode: {video_path}")

    vt = torch.as_tensor(vt)  # (C,T,H,W)
    C, T, H, W = vt.shape
    if T <= 0:
        raise RuntimeError(f"Empty clip: {video_path}")

    # Uniform temporal subsample
    if transform is None:
        if T >= num_frames:
            idx = torch.linspace(0, T - 1, steps=num_frames).long()
            vt = vt.index_select(1, idx)
        else:
            # repeat-last to reach num_frames
            pad = vt[:, -1:].repeat(1, num_frames - T, 1, 1)
            vt = torch.cat([vt, pad], dim=1)

    # Frames list HWC uint8
    frames_list = []
    for j in range(vt.shape[1]):
        frame = vt[:, j].permute(1, 2, 0).cpu().numpy()  # HWC
        if frame.dtype != np.uint8:
            # Convert to 0..255 uint8
            mx = float(frame.max()) if frame.size else 255.0
            if mx <= 1.0:
                frame = (frame * 255.0).round().clip(0, 255).astype(np.uint8)
            else:
                frame = frame.round().clip(0, 255).astype(np.uint8)
        frames_list.append(np.ascontiguousarray(frame))

    # processor -> pixel_values
    try:
        batch = processor(videos=[frames_list], return_tensors="pt")
    except TypeError:
        batch = processor([frames_list], return_tensors="pt")

    pv = batch["pixel_values"]  # (B,T,C,H,W) ή (B,C,T,H,W)
    if pv.ndim == 4:
        pv = pv.unsqueeze(0)
    if pv.ndim != 5:
        raise RuntimeError(f"Unexpected pixel_values shape: {pv.shape}")

    # (B,C,T,H,W) -> (B,T,C,H,W) if needed
    if pv.shape[1] == 3 and pv.shape[2] != 3:
        pv = pv.permute(0, 2, 1, 3, 4)

    pv = pv.to(device)

    use_amp = dtype in ("fp16", "bf16") and str(device).startswith("cuda")
    amp_dtype = torch.bfloat16 if dtype == "bf16" else torch.float16

    if use_amp:
        with torch.autocast(device_type="cuda", dtype=amp_dtype):
            out = model(pixel_values=pv)
    else:
        out = model(pixel_values=pv)

    #hs = out.last_hidden_state  # tuple(len = num_layers+1)
    #tokens = hs[layer_idx]  # (B, N, D)
    tokens = out.last_hidden_state

    if use_cls:
        emb = tokens[:, 0, :]
    else:
        emb = tokens.mean(dim=1)

    emb = emb.squeeze(0).detach().cpu()

    # Cast output dtype
    if dtype == "fp16":
        emb = emb.to(torch.float16)
    elif dtype == "bf16":
        emb = emb.to(torch.bfloat16)
    else:
        emb = emb.to(torch.float32)

    return emb  # [D]

