import numpy as np
import torch
from collections import deque


COCO17_EDGES = [
    (0, 1), (0, 2),
    (1, 3), (2, 4),
    (0, 5), (0, 6),
    (5, 7), (7, 9),
    (6, 8), (8, 10),
    (5, 11), (6, 12),
    (11, 13), (13, 15),
    (12, 14), (14, 16),
]

def _bfs_dist(num_joints: int, edges, root: int):
    g = [[] for _ in range(num_joints)]
    for a, b in edges:
        g[a].append(b)
        g[b].append(a)
    dist = [-1] * num_joints
    dist[root] = 0
    q = deque([root])
    while q:
        u = q.popleft()
        for v in g[u]:
            if dist[v] == -1:
                dist[v] = dist[u] + 1
                q.append(v)
    for i in range(num_joints):
        if dist[i] == -1:
            dist[i] = 999
    return dist


def normalize_digraph(A: np.ndarray, eps: float = 1e-6):
    K, V, _ = A.shape
    out = np.zeros_like(A, dtype=np.float32)
    for k in range(K):
        Ak = A[k]
        Dl = np.sum(Ak, axis=1)
        Dn = np.diag(1.0 / (Dl + eps))
        out[k] = Dn @ Ak
    return out


def build_coco17_A_subsets(root: int = 11) -> torch.Tensor:
    """
    A: [K=3, V=17, V=17]
      K0: self + same-distance neighbors
      K1: centripetal  (towards root)
      K2: centrifugal  (away from root)
    """
    V = 17
    edges = COCO17_EDGES
    dist = _bfs_dist(V, edges, root=root)

    A0 = np.eye(V, dtype=np.float32)
    A1 = np.zeros((V, V), dtype=np.float32)
    A2 = np.zeros((V, V), dtype=np.float32)

    for i, j in edges:
        di, dj = dist[i], dist[j]
        if di == dj:
            A0[i, j] = 1.0
            A0[j, i] = 1.0
        elif di < dj:
            A1[j, i] = 1.0  # j -> i
            A2[i, j] = 1.0  # i -> j
        else:
            A1[i, j] = 1.0
            A2[j, i] = 1.0

    A = np.stack([A0, A1, A2], axis=0)
    A = normalize_digraph(A)
    return torch.tensor(A, dtype=torch.float32)