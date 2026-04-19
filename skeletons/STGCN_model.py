import torch
import torch.nn as nn
from collections import deque
from model_config import COCO17_EDGES, _bfs_dist, normalize_digraph, build_coco17_A_subsets


# -------------------- ST-GCN-like Model ---------------------

class SpatialGraphConv(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, A: torch.Tensor, use_edge_importance: bool = False):
        super().__init__()
        assert A.dim() == 3, "A must be [K,V,V]"
        self.register_buffer("A", A)   # [K,V,V]
        self.K = A.size(0)

        self.conv = nn.Conv2d(in_channels, out_channels * self.K, kernel_size=1, bias=False)

        if use_edge_importance:
            self.edge_importance = nn.Parameter(torch.ones_like(A))  # [K,V,V]
        else:
            self.edge_importance = None

        self.bn = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        # x: [N,C,T,V]
        N, C, T, V = x.shape
        A = self.A
        if self.edge_importance is not None:
            A = A * self.edge_importance

        x = self.conv(x)                 # [N, K*Cout, T, V]
        x = x.view(N, self.K, -1, T, V)  # [N,K,Cout,T,V]

        out = 0
        for k in range(self.K):
            out = out + torch.einsum("nctv,vw->nctw", x[:, k], A[k])

        out = self.bn(out)
        return out


class MultiScaleTCN(nn.Module):
    def __init__(self, channels, stride=1):
        super().__init__()
        self.b1 = nn.Sequential(
            nn.ReLU(inplace=False),
            nn.Conv2d(channels, channels, kernel_size=(9, 1), padding=(4, 0), stride=(stride, 1), bias=False),
            nn.BatchNorm2d(channels),
        )
        self.b2 = nn.Sequential(
            nn.ReLU(inplace=False),
            nn.Conv2d(channels, channels, kernel_size=(3, 1), padding=(1, 0), stride=(stride, 1), bias=False),
            nn.BatchNorm2d(channels),
        )
        self.fuse = nn.Sequential(
            nn.ReLU(inplace=False),
            nn.Conv2d(2 * channels, channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(channels),
        )

    def forward(self, x):
        y1 = self.b1(x)
        y2 = self.b2(x)
        y = torch.cat([y1, y2], dim=1)
        return self.fuse(y)


class STGCNBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        A: torch.Tensor,
        stride: int = 1,
        residual: bool = True,
        use_edge_importance: bool = False,
        multi_scale_tcn: bool = False,
    ):
        super().__init__()
        self.gcn = SpatialGraphConv(in_channels, out_channels, A, use_edge_importance=use_edge_importance)

        if multi_scale_tcn:
            self.tcn = MultiScaleTCN(out_channels, stride=stride)
        else:
            self.tcn = nn.Sequential(
                nn.ReLU(inplace=False),
                nn.Conv2d(out_channels, out_channels, kernel_size=(9, 1), padding=(4, 0), stride=(stride, 1), bias=False),
                nn.BatchNorm2d(out_channels),
            )

        if not residual:
            self.residual = lambda x: 0
        elif in_channels == out_channels and stride == 1:
            self.residual = lambda x: x
        else:
            self.residual = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=(stride, 1), bias=False),
                nn.BatchNorm2d(out_channels),
            )

        self.relu = nn.ReLU(inplace=False)

    def forward(self, x):
        res = self.residual(x)
        x = self.gcn(x)
        x = self.tcn(x)
        x = x + res
        return self.relu(x)


class STGCNModel(nn.Module):
    def __init__(
        self,
        num_class: int,
        in_channels: int = 6,
        hidden_channels: int = 64,
        num_layers: int = 6,
        use_edge_importance: bool = False,
        multi_scale_tcn: bool = False,
        root: int = 11,
    ):
        super().__init__()
        A = build_coco17_A_subsets(root=root)  # [3,17,17]
        self.register_buffer("A", A)

        layers = []
        c_in = in_channels
        c_out = hidden_channels

        for _ in range(num_layers):
            layers.append(STGCNBlock(
                in_channels=c_in,
                out_channels=c_out,
                A=self.A,
                stride=1,
                residual=True,
                use_edge_importance=use_edge_importance,
                multi_scale_tcn=multi_scale_tcn,
            ))
            c_in = c_out

        self.stgcn = nn.Sequential(*layers)
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(c_out, num_class)

    def forward(self, x):
        # x: [N,C,T,V]
        x = self.stgcn(x)
        x = self.pool(x)       # [N,C,1,1]
        x = x.flatten(1)       # [N,C]
        return self.fc(x)      # [N,num_class]


    def forward_features(self, x):
        """
        Returns:
          emb:   [N,D] (D=hidden_channels)
          logits:[N,C]
        """
        h = self.stgcn(x)
        h = self.pool(h).flatten(1)   # [N,D]
        logits = self.fc(h)           # [N,C]
        return h, logits