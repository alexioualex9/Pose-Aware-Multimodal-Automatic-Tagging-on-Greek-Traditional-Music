from pathlib import Path
from typing import Dict, Optional
import numpy as np


def _sec_to_ms(sec: float) -> int:
    return int(round(float(sec) * 1000.0))


class SkeletonVideoStore:
    """
    Loads per-video skeleton embeddings from:
      skel_emb_dir/<vid>.npz

    Expected keys:
      - clip_emb:   [N_s, D_s]
      - clip_start: [N_s] seconds
      - clip_end:   [N_s] seconds

    Provides alignment by (start_ms, end_ms) exact match after rounding.
    Missing -> None.
    """

    def __init__(self, skel_emb_dir: str):
        self.skel_emb_dir = Path(skel_emb_dir)
        self._cache: Dict[str, dict] = {}

    def has_video(self, vid: str) -> bool:
        return (self.skel_emb_dir / f"{vid}.npz").exists()

    def load_video(self, vid: str) -> Optional[dict]:
        if vid in self._cache:
            return self._cache[vid]

        p = self.skel_emb_dir / f"{vid}.npz"
        if not p.exists():
            self._cache[vid] = None
            return None

        z = np.load(p, allow_pickle=False)
        if "clip_emb" not in z or "clip_start" not in z or "clip_end" not in z:
            raise KeyError(f"[skeleton] {p} missing keys. Expected: clip_emb, clip_start, clip_end. Found: {list(z.files)}")
        clip_emb = np.asarray(z["clip_emb"], dtype=np.float32)
        clip_start = np.asarray(z["clip_start"], dtype=np.float32).reshape(-1)
        clip_end = np.asarray(z["clip_end"], dtype=np.float32).reshape(-1)

        if clip_emb.ndim != 2:
            raise RuntimeError(f"[skeleton] {p} clip_emb must be [N,D], got {clip_emb.shape}")
        if clip_start.shape[0] != clip_emb.shape[0] or clip_end.shape[0] != clip_emb.shape[0]:
            raise RuntimeError(f"[skeleton] {p} time arrays mismatch with clip_emb: "
                               f"clip_emb N={clip_emb.shape[0]} start={clip_start.shape} end={clip_end.shape}")

        # build mapping + store ms arrays for fallback
        start_ms = np.array([_sec_to_ms(s) for s in clip_start], dtype=np.int32)
        end_ms   = np.array([_sec_to_ms(e) for e in clip_end], dtype=np.int32)

        mapping = {}
        for i in range(clip_emb.shape[0]):
            mapping[(int(start_ms[i]), int(end_ms[i]))] = int(i)

        out = {
            "path": str(p),
            "clip_emb": clip_emb,   # [N_s, D_s]
            "clip_start": clip_start,
            "clip_end": clip_end,
            "start_ms": start_ms,
            "end_ms": end_ms,
            "map": mapping,
            "D_s": int(clip_emb.shape[1]),
            "N_s": int(clip_emb.shape[0]),
        }
        self._cache[vid] = out
        return out

    def get_clip_emb(self, vid: str, start_sec: float, end_sec: float, tol_ms: int = 50) -> Optional[np.ndarray]:
        info = self.load_video(vid)
        if info is None:
            return None

        """
        key = (_sec_to_ms(start_sec), _sec_to_ms(end_sec))
        idx = info["map"].get(key, None)
        if idx is None:
            return None
        return info["clip_emb"][idx]
        """
        s = _sec_to_ms(start_sec)
        e = _sec_to_ms(end_sec)

        # 1) exact match (fast)
        idx = info["map"].get((s, e), None)
        if idx is not None:
            return info["clip_emb"][idx]

        # 2) fallback: nearest by start/end within tol
        ds = np.abs(info["start_ms"] - s)
        de = np.abs(info["end_ms"] - e)
        score = ds + de
        j = int(score.argmin())
        if ds[j] <= tol_ms and de[j] <= tol_ms:
            return info["clip_emb"][j]

        return None
