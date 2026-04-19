from pathlib import Path
import numpy as np


# Save test results
def save_results(Y, S, P, used_vids, modalities, save_npz, threshold, fusion, w_a, w_v, w_s):
    if save_npz:
        out_path = Path(save_npz)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            out_path,
            video_id=np.asarray(used_vids),
            Y=Y.astype(np.int8),
            S=S.astype(np.float32),
            P=P,
            threshold=np.asarray(threshold, dtype=np.float32),
            fusion=np.asarray(fusion),
            modalities=np.asarray("".join(modalities)),
            w_a=np.asarray(w_a, dtype=np.float32),
            w_v=np.asarray(w_v, dtype=np.float32),
            w_s=np.asarray(w_s, dtype=np.float32),
        )
        print(f"\n[done] Saved fused outputs to: {out_path}")