import torch.nn as nn

# Model (MLP head)
# -----------------------------

class MLP(nn.Module):
    def __init__(self, d_in: int, d_hid: int, C: int, p_drop: float = 0.3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_in, d_hid),
            nn.GELU(),
            nn.Dropout(p_drop),
            nn.Linear(d_hid, C),
        )
    def forward(self, x):
        return self.net(x)  # logits