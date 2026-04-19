import numpy as np

###########################
# Interpolation
###########################
def interpolate_short_gaps(
    kps_seq,
    conf_thresh: float = 0.2,
    max_gap: int = 3,
    interpolate_conf: bool = False,
):
    if not kps_seq:
        return kps_seq

    T = len(kps_seq)
    J = kps_seq[0].shape[0]
    out = [kp.copy() for kp in kps_seq]

    for j in range(J):
        conf = np.array([out[t][j, 2] for t in range(T)], dtype=np.float32)
        valid = conf >= conf_thresh

        t = 0
        while t < T:
            if valid[t]:
                t += 1
                continue

            t0 = t - 1
            t1 = t
            while t1 < T and not valid[t1]:
                t1 += 1
            gap_len = t1 - t

            if (
                gap_len <= max_gap
                and t0 >= 0
                and t1 < T
                and valid[t0]
                and valid[t1]
            ):
                p0 = out[t0][j, :2]
                p1 = out[t1][j, :2]
                c0 = out[t0][j, 2]
                c1 = out[t1][j, 2]

                for k in range(1, gap_len + 1):
                    alpha = k / (gap_len + 1)
                    out[t0 + k][j, 0:2] = (1 - alpha) * p0 + alpha * p1
                    if interpolate_conf:
                        out[t0 + k][j, 2] = (1 - alpha) * c0 + alpha * c1

            t = t1

    return out