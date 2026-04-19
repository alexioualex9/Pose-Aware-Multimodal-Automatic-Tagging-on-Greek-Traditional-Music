import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class EmbCLSFusionTransformerVSMasked(nn.Module):
    def __init__(
        self,
        dim_v,
        dim_s,
        num_labels,
        d_model=256,
        nhead=4,
        num_layers=1,
        dropout=0.2,
        max_clips=96,
        keep_missing_s_in_final=True,
    ):
        super().__init__()

        self.dim_v, self.dim_s = dim_v, dim_s
        self.d_model = d_model
        self.max_clips = max_clips
        self.keep_missing_s_in_final = keep_missing_s_in_final

        self.video_proj = nn.Linear(dim_v, d_model)
        self.skel_proj = nn.Linear(dim_s, d_model)

        self.video_ln = nn.LayerNorm(d_model)
        self.skel_ln = nn.LayerNorm(d_model)

        self.mod_emb = nn.Embedding(2, d_model)   # 0=v, 1=s
        self.clip_pos_emb = nn.Embedding(max_clips, d_model)

        self.cls_token = nn.Parameter(torch.zeros(1, 1, d_model))
        self.missing_skel_token = nn.Parameter(torch.zeros(1, 1, d_model))

        enc = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=4 * d_model,
            dropout=dropout,
            batch_first=True,
            activation="gelu",
        )
        self.transformer = nn.TransformerEncoder(enc, num_layers=num_layers)

        self.fc1 = nn.Linear(d_model, d_model)
        self.fc2 = nn.Linear(d_model, num_labels)
        self.dropout = nn.Dropout(dropout)
        self.out_ln = nn.LayerNorm(d_model)

        nn.init.trunc_normal_(self.cls_token, std=0.02)
        nn.init.trunc_normal_(self.missing_skel_token, std=0.02)
        nn.init.trunc_normal_(self.clip_pos_emb.weight, std=0.02)
        nn.init.trunc_normal_(self.mod_emb.weight, std=0.02)

    def forward(self, emb_v, emb_s, mask_s, valid_mask):
        """
        emb_v: [B, T, Dv]
        emb_s: [B, T, Ds]
        mask_s: [B, T] bool/float   (1 if skeleton exists at timestep)
        valid_mask: [B, T] bool     (1 if timestep is real, not padding)
        """
        B, T, Dv = emb_v.shape
        _, T2, Ds = emb_s.shape
        assert T == T2 and Dv == self.dim_v and Ds == self.dim_s

        device = emb_v.device
        valid_mask = valid_mask.to(device).bool()

        if mask_s.dtype == torch.bool:
            mask_s = mask_s.to(device)
        else:
            mask_s = (mask_s > 0.5).to(device)

        skel_present = mask_s & valid_mask
        skel_present_f = skel_present.float().unsqueeze(-1)

        v = F.gelu(self.video_ln(self.video_proj(emb_v)))
        s = F.gelu(self.skel_ln(self.skel_proj(emb_s)))

        missing_s = self.missing_skel_token.expand(B, T, -1)
        s = skel_present_f * s + (1.0 - skel_present_f) * missing_s

        # interleaved [v1,s1,v2,s2,...]
        tokens = torch.stack([v, s], dim=2).reshape(B, 2 * T, self.d_model)

        clip_idx = torch.arange(T, device=device).repeat_interleave(2)
        if T > self.max_clips:
            clip_idx = clip_idx % self.max_clips
        pos = self.clip_pos_emb(clip_idx).unsqueeze(0)

        mod_idx = torch.tensor([0, 1], device=device).repeat(T)
        mod = self.mod_emb(mod_idx).unsqueeze(0)

        x = tokens + pos + mod
        cls = self.cls_token.expand(B, -1, -1)
        x = torch.cat([cls, x], dim=1)

        pad = torch.zeros((B, 1 + 2 * T), dtype=torch.bool, device=device)

        pad_clips = ~valid_mask
        pad[:, 1::2] |= pad_clips   # video
        pad[:, 2::2] |= pad_clips   # skeleton

        if not self.keep_missing_s_in_final:
            pad[:, 2::2] |= ~skel_present

        x = self.transformer(x, src_key_padding_mask=pad)

        h = self.out_ln(x[:, 0, :])
        z = F.gelu(self.fc1(h))
        z = self.dropout(z)
        return self.fc2(z)






class GMUVideoSkeletonMissingFusion(nn.Module):
    """
    Original simplified bimodal GMU with missing-modality handling
    for Video + Skeleton fusion.

    For each clip:
        h_v = tanh(W_v v)
        h_s = tanh(W_s s)                  if skeleton is present
            = missing_skel_token          if skeleton is missing

        z   = sigmoid(W_z [v ; s_hat])
        h   = z * h_v + (1 - z) * h_s

    where:
        s_hat = s                         if skeleton is present
              = 0                         if skeleton is missing

    Notes:
        - The gate follows the original GMU formulation:
          it is computed from raw modality inputs, not hidden states.
        - Missing skeleton is handled with:
            (1) a learned missing token in the transformed branch
            (2) a zero vector in the raw gate-input branch
    """

    def __init__(self, dim_v: int, dim_s: int, d_model: int):
        super().__init__()

        self.dim_v = dim_v
        self.dim_s = dim_s
        self.d_model = d_model

        # Modality-specific transformed states
        self.video_proj = nn.Linear(dim_v, d_model, bias=True)
        self.skel_proj = nn.Linear(dim_s, d_model, bias=True)

        # Learned token for missing skeleton in hidden/transformed branch
        self.missing_skel_token = nn.Parameter(torch.zeros(1, 1, d_model))

        # Learned token for missing skeleton in raw gate-input branch
        self.missing_skel_gate_token = nn.Parameter(torch.zeros(1, 1, dim_s))

        # Original GMU gate from raw inputs
        self.gate_proj = nn.Linear(dim_v + dim_s, 1, bias=True)

        nn.init.trunc_normal_(self.missing_skel_token, std=0.02)
        nn.init.trunc_normal_(self.missing_skel_gate_token, std=0.02)

    def forward(
        self,
        emb_v: torch.Tensor,
        emb_s: torch.Tensor,
        mask_s: torch.Tensor,
        valid_mask: torch.Tensor,
    ):
        """
        Args:
            emb_v:      [B, T, dim_v]
            emb_s:      [B, T, dim_s]
            mask_s:     [B, T] bool or {0,1}, True where skeleton is available
            valid_mask: [B, T] bool, True where clip is valid (not padding)

        Returns:
            h: [B, T, d_model]
            z: [B, T, 1]
        """
        B, T, Dv = emb_v.shape
        Bs, Ts, Ds = emb_s.shape

        assert B == Bs and T == Ts, f"Video/skeleton shape mismatch: {emb_v.shape} vs {emb_s.shape}"
        assert Dv == self.dim_v, f"Expected video dim {self.dim_v}, got {Dv}"
        assert Ds == self.dim_s, f"Expected skeleton dim {self.dim_s}, got {Ds}"

        device = emb_v.device

        if valid_mask is None:
            valid_mask = torch.ones(B, T, dtype=torch.bool, device=device)
        else:
            valid_mask = valid_mask.to(device).bool()

        if mask_s.dtype == torch.bool:
            mask_s = mask_s.to(device)
        else:
            mask_s = (mask_s > 0.5).to(device)

        # Skeleton can only exist on valid clips
        mask_s = mask_s & valid_mask

        valid_mask_f = valid_mask.unsqueeze(-1).to(emb_v.dtype)   # [B,T,1]
        mask_s_bool = mask_s.unsqueeze(-1)                        # [B,T,1]

        # 1) Modality-specific transformed states
        h_v = torch.tanh(self.video_proj(emb_v))                  # [B,T,D]
        h_s_raw = torch.tanh(self.skel_proj(emb_s))               # [B,T,D]

        # 2) Missing skeleton handling in transformed branch
        missing_s = self.missing_skel_token.expand(B, T, -1)      # [B,T,D]
        h_s = torch.where(mask_s_bool.expand_as(h_s_raw), h_s_raw, missing_s)

        # 3) Zero-out padded clips before fusion
        h_v = h_v * valid_mask_f
        h_s = h_s * valid_mask_f

        # 4) Original GMU gate from raw inputs
        missing_s_gate = self.missing_skel_gate_token.expand(B, T, -1)   # [B,T,Ds]
        emb_s_gate = torch.where(
            mask_s_bool.expand_as(emb_s),
            emb_s,
            missing_s_gate
        )

        gate_in = torch.cat([emb_v, emb_s_gate], dim=-1)          # [B,T,Dv+Ds]
        z = torch.sigmoid(self.gate_proj(gate_in))                # [B,T,1]
        z = z * valid_mask_f

        # 5) GMU fusion
        h = z * h_v + (1.0 - z) * h_s                             # [B,T,D]
        h = h * valid_mask_f

        return h, z


class EmbCLSFusionTransformerVSMasked_gate(nn.Module):
    """
    Video-skeleton classifier with original-GMU-style fusion
    and Transformer-based temporal modeling.

    Pipeline:
        1. Apply original GMU fusion with missing skeleton handling
        2. Add temporal positional embeddings
        3. Prepend [CLS] token
        4. Encode with Transformer
        5. Classify using the final [CLS] representation
    """

    def __init__(
        self,
        dim_v,
        dim_s,
        num_labels,
        d_model=256,
        nhead=4,
        num_layers=1,
        dropout=0.2,
        max_clips=96,
    ):
        super().__init__()

        self.dim_v = dim_v
        self.dim_s = dim_s
        self.d_model = d_model
        self.max_clips = max_clips

        self.fusion = GMUVideoSkeletonMissingFusion(
            dim_v=dim_v,
            dim_s=dim_s,
            d_model=d_model,
        )

        self.clip_pos_emb = nn.Embedding(max_clips, d_model)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, d_model))

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=4 * d_model,
            dropout=dropout,
            batch_first=True,
            activation="gelu",
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        self.out_ln = nn.LayerNorm(d_model)
        self.fc1 = nn.Linear(d_model, d_model)
        self.fc2 = nn.Linear(d_model, num_labels)
        self.dropout = nn.Dropout(dropout)

        nn.init.trunc_normal_(self.cls_token, std=0.02)
        nn.init.trunc_normal_(self.clip_pos_emb.weight, std=0.02)

    def forward(
        self,
        emb_v: torch.Tensor,
        emb_s: torch.Tensor,
        mask_s: torch.Tensor,
        valid_mask: torch.Tensor,
        return_gates: bool = False,
    ):
        """
        Args:
            emb_v:      [B,T,Dv]
            emb_s:      [B,T,Ds]
            mask_s:     [B,T] bool or {0,1}, True where skeleton is available
            valid_mask: [B,T] bool, True where clip is valid (not padding)
            return_gates: if True, also return clip-wise gates

        Returns:
            logits: [B, num_labels]
            gates:  [B, T] (optional)
        """
        B, T, Dv = emb_v.shape
        _, T2, Ds = emb_s.shape

        assert T == T2, f"Video/skeleton clip mismatch: {T} vs {T2}"
        assert Dv == self.dim_v, f"Expected video dim {self.dim_v}, got {Dv}"
        assert Ds == self.dim_s, f"Expected skeleton dim {self.dim_s}, got {Ds}"

        device = emb_v.device

        if valid_mask is None:
            valid_mask = torch.ones(B, T, dtype=torch.bool, device=device)
        else:
            valid_mask = valid_mask.to(device).bool()

        # 1) Original GMU fusion with missing modality handling
        h, z = self.fusion(
            emb_v=emb_v,
            emb_s=emb_s,
            mask_s=mask_s,
            valid_mask=valid_mask,
        )                                                         # h: [B,T,D], z: [B,T,1]

        # 2) Positional embeddings
        clip_idx = torch.arange(T, device=device)
        if T > self.max_clips:
            clip_idx = clip_idx % self.max_clips

        pos = self.clip_pos_emb(clip_idx).unsqueeze(0)            # [1,T,D]
        x = h + pos

        # 3) Prepend CLS token
        cls = self.cls_token.expand(B, -1, -1)                    # [B,1,D]
        x = torch.cat([cls, x], dim=1)                            # [B,1+T,D]

        # 4) Padding mask
        pad = torch.zeros((B, T + 1), dtype=torch.bool, device=device)
        pad[:, 1:] = ~valid_mask

        # 5) Transformer encoder
        x = self.transformer(x, src_key_padding_mask=pad)

        # 6) Classification head
        h_cls = self.out_ln(x[:, 0, :])
        y = F.gelu(self.fc1(h_cls))
        y = self.dropout(y)
        logits = self.fc2(y)

        if return_gates:
            gates = z.squeeze(-1) * valid_mask.to(z.dtype)        # [B,T]
            return logits, gates

        return logits






# -----------------------------
# Positional encoding (fixed sinusoidal, MulT-style)
# -----------------------------
class SinusoidalPositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super().__init__()

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2, dtype=torch.float32)
            * (-math.log(10000.0) / d_model)
        )

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # [1, T, D]

        self.register_buffer("pe", pe, persistent=False)

    def forward(self, x):
        # x: [B, T, D]
        T = x.size(1)
        return self.pe[:, :T, :]


# -----------------------------
# Input projection (official MulT repo uses Conv1d kernel_size=1)
# -----------------------------
class Conv1dProjection(nn.Module):
    def __init__(self, in_dim, out_dim, bias=False):
        super().__init__()
        self.proj = nn.Conv1d(in_dim, out_dim, kernel_size=1, bias=bias)

    def forward(self, x):
        # x: [B, T, C]
        x = x.transpose(1, 2)   # [B, C, T]
        x = self.proj(x)        # [B, D, T]
        x = x.transpose(1, 2)   # [B, T, D]
        return x


# -----------------------------
# Transformer blocks
# -----------------------------
class FeedForward(nn.Module):
    """
    Pre-norm FFN block.
    GELU is kept for modern stability; if you want even closer to older
    transformer codebases you can replace GELU with ReLU.
    """
    def __init__(self, d_model, dropout=0.1, expansion=4):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, expansion * d_model),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(expansion * d_model, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)


class TransformerSelfBlock(nn.Module):
    """
    Standard pre-norm transformer self-attention block.
    Used for target-specific temporal memory after crossmodal fusion.
    """
    def __init__(self, d_model, nhead, dropout=0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.self_attn = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=nhead,
            dropout=dropout,
            batch_first=True,
        )
        self.norm2 = nn.LayerNorm(d_model)
        self.ffn = FeedForward(d_model, dropout=dropout)

    def forward(self, x, key_padding_mask=None):
        # x: [B, T, D]
        x_norm = self.norm1(x)
        attn_out, _ = self.self_attn(
            x_norm, x_norm, x_norm,
            key_padding_mask=key_padding_mask,
            need_weights=False,
        )
        x = x + attn_out
        x = x + self.ffn(self.norm2(x))
        return x


class CrossModalBlock(nn.Module):
    """
    One directional MulT-style crossmodal block:
        target <- source

    Important:
    - Only target is updated.
    - Source remains fixed low-level representation across stacked layers,
      matching the MulT spirit.
    - No self-attention inside the directional cross stack.
    """
    def __init__(self, d_model, nhead, dropout=0.1):
        super().__init__()
        self.norm_q = nn.LayerNorm(d_model)
        self.norm_kv = nn.LayerNorm(d_model)
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=nhead,
            dropout=dropout,
            batch_first=True,
        )
        self.norm_ffn = nn.LayerNorm(d_model)
        self.ffn = FeedForward(d_model, dropout=dropout)

    def forward(self, target, source, source_key_padding_mask=None):
        q = self.norm_q(target)
        kv = self.norm_kv(source)

        out, _ = self.cross_attn(
            query=q,
            key=kv,
            value=kv,
            key_padding_mask=source_key_padding_mask,
            need_weights=False,
        )
        target = target + out
        target = target + self.ffn(self.norm_ffn(target))
        return target


class DirectionalCrossStack(nn.Module):
    """
    MulT-faithful directional stack:
        target <- source

    Repeated crossmodal adaptation only.
    No self-attention before/after.
    """
    def __init__(self, d_model, nhead, num_layers=2, dropout=0.1):
        super().__init__()
        self.cross_layers = nn.ModuleList([
            CrossModalBlock(d_model, nhead, dropout)
            for _ in range(num_layers)
        ])

    def forward(self, target, source, source_key_padding_mask=None):
        x = target
        for blk in self.cross_layers:
            x = blk(
                target=x,
                source=source,
                source_key_padding_mask=source_key_padding_mask,
            )
        return x


# -----------------------------
# Per-target fusion memory (MulT-like)
# -----------------------------
class TargetFusionMemory(nn.Module):
    """
    For one target modality, fuse incoming directional streams by:
      1) concatenation along feature dimension
      2) self-attention temporal memory on the concatenated representation

    For the 2-modal case, each target receives one incoming stream only:
      target V receives [V_from_S]
      target S receives [S_from_V]

    We keep exactly the same class/interface as in the 3-modal case.
    With num_sources=1, this reduces to a target-specific temporal memory
    over the directional representation.
    """
    def __init__(self, d_model, num_sources, nhead, num_memory_layers=3, dropout=0.1):
        super().__init__()
        self.out_dim = num_sources * d_model

        self.memory = nn.ModuleList([
            TransformerSelfBlock(self.out_dim, nhead, dropout)
            for _ in range(num_memory_layers)
        ])

    def forward(self, xs, key_padding_mask=None):
        # xs: list of [B, T, D]
        x = torch.cat(xs, dim=-1)  # [B, T, num_sources*D]
        for blk in self.memory:
            x = blk(x, key_padding_mask=key_padding_mask)
        return x


# -----------------------------
# Output head (MulT-style residual head)
# -----------------------------
class MulTOutputHead(nn.Module):
    """
    MulT-like output head:
      - take last valid hidden state from each target sequence
      - concatenate
      - residual projection block
      - output layer
    """
    def __init__(self, per_target_dim, num_targets, num_labels, dropout=0.2):
        super().__init__()
        self.per_target_dim = per_target_dim
        self.num_targets = num_targets
        self.combined_dim = per_target_dim * num_targets
        self.out_dropout = dropout

        self.proj1 = nn.Linear(self.combined_dim, self.combined_dim)
        self.proj2 = nn.Linear(self.combined_dim, self.combined_dim)
        self.out_layer = nn.Linear(self.combined_dim, num_labels)

    @staticmethod
    def take_last_valid(x, valid_mask):
        """
        x: [B, T, D]
        valid_mask: [B, T] bool
        """
        valid_mask = valid_mask.bool()
        B, T, D = x.shape

        lengths = valid_mask.long().sum(dim=1)          # [B]
        last_idx = (lengths - 1).clamp(min=0)           # [B]
        batch_idx = torch.arange(B, device=x.device)

        return x[batch_idx, last_idx, :]                # [B, D]

    def forward(self, seqs, masks):
        assert len(seqs) == self.num_targets
        assert len(masks) == self.num_targets

        last_hs = [self.take_last_valid(s, m) for s, m in zip(seqs, masks)]
        last_hs = torch.cat(last_hs, dim=-1)            # [B, combined_dim]

        last_hs_proj = self.proj2(
            F.dropout(
                F.relu(self.proj1(last_hs)),
                p=self.out_dropout,
                training=self.training,
            )
        )
        last_hs_proj = last_hs_proj + last_hs

        output = self.out_layer(last_hs_proj)
        return output, last_hs


# -----------------------------
# Main model
# -----------------------------
class EmbCLSFusionTransformerVSMasked_cross_attention(nn.Module):
    """
    MulT-faithful 2-modal Video-Skeleton model with one explicit extension:
    temporally missing skeleton observations handled via a learned token.

    Modalities:
      V = video
      S = skeleton

    Inputs
    ------
    emb_v:      [B, T, Dv]
    emb_s:      [B, T, Ds]
    mask_s:     [B, T] bool/float, True/1 if skeleton exists at timestep
    valid_mask: [B, T] bool, True if timestep is real (not padding)
    """

    def __init__(
        self,
        dim_v,
        dim_s,
        num_labels,
        d_model=256,
        nhead=4,
        cross_num_layers=2,
        memory_num_layers=3,
        dropout=0.2,
        max_len=5000,
        use_missing_skeleton_token=True,
        skeleton_head_uses_valid_mask=True,
    ):
        super().__init__()

        self.dim_v = dim_v
        self.dim_s = dim_s
        self.d_model = d_model
        self.use_missing_skeleton_token = use_missing_skeleton_token
        self.skeleton_head_uses_valid_mask = skeleton_head_uses_valid_mask

        # input projections (MulT repo style: 1x1 conv)
        self.video_proj = Conv1dProjection(dim_v, d_model, bias=False)
        self.skel_proj = Conv1dProjection(dim_s, d_model, bias=False)

        # fixed positional encoding
        self.pos_enc = SinusoidalPositionalEncoding(d_model, max_len=max_len)

        # dropout after input projection + position
        self.input_dropout = nn.Dropout(dropout)

        # learned token for missing skeleton observations (the only intentional extension)
        if use_missing_skeleton_token:
            self.missing_skel_token = nn.Parameter(torch.zeros(1, 1, d_model))
            nn.init.trunc_normal_(self.missing_skel_token, std=0.02)
        else:
            self.register_parameter("missing_skel_token", None)

        # all pairwise directional crossmodal streams
        # target V
        self.v_from_s = DirectionalCrossStack(
            d_model=d_model, nhead=nhead, num_layers=cross_num_layers, dropout=dropout
        )

        # target S
        self.s_from_v = DirectionalCrossStack(
            d_model=d_model, nhead=nhead, num_layers=cross_num_layers, dropout=dropout
        )

        # per-target fusion memory over concatenated incoming directional streams
        # in the 2-modal case each target has exactly one incoming source
        self.fuse_v = TargetFusionMemory(
            d_model=d_model,
            num_sources=1,
            nhead=nhead,
            num_memory_layers=memory_num_layers,
            dropout=dropout,
        )
        self.fuse_s = TargetFusionMemory(
            d_model=d_model,
            num_sources=1,
            nhead=nhead,
            num_memory_layers=memory_num_layers,
            dropout=dropout,
        )

        # each target after fusion has dim = 1 * d_model
        self.head = MulTOutputHead(
            per_target_dim=1 * d_model,
            num_targets=2,
            num_labels=num_labels,
            dropout=dropout,
        )

    @staticmethod
    def unmask_first_if_all(mask):
        """
        mask: [B, T] bool, True means key is masked/unavailable.

        PyTorch MultiheadAttention can behave badly if all keys are masked.
        This makes one fallback key visible.
        """
        all_masked = mask.all(dim=1)
        if all_masked.any():
            mask = mask.clone()
            idx = all_masked.nonzero(as_tuple=True)[0]
            mask[idx, 0] = False
        return mask

    def forward(self, emb_v, emb_s, mask_s, valid_mask):
        B, T, Dv = emb_v.shape
        B2, T2, Ds = emb_s.shape

        assert B == B2, "Batch size mismatch."
        assert T == T2, "Sequence length mismatch."
        assert Dv == self.dim_v, f"Expected dim_v={self.dim_v}, got {Dv}."
        assert Ds == self.dim_s, f"Expected dim_s={self.dim_s}, got {Ds}."

        device = emb_v.device
        valid_mask = valid_mask.to(device=device, dtype=torch.bool)

        if mask_s.dtype == torch.bool:
            mask_s = mask_s.to(device)
        else:
            mask_s = (mask_s > 0.5).to(device)

        # observed skeleton only where timestep is valid and skeleton exists
        skel_present = valid_mask & mask_s                    # [B, T]
        skel_present_f = skel_present.unsqueeze(-1).to(emb_v.dtype)

        # projections
        v = self.video_proj(emb_v)                            # [B, T, D]
        s = self.skel_proj(emb_s)                             # [B, T, D]

        # missing skeleton extension
        if self.use_missing_skeleton_token:
            missing_s = self.missing_skel_token.expand(B, T, -1)
            s = skel_present_f * s + (1.0 - skel_present_f) * missing_s
        else:
            # if not using learned token, zero-out missing skeleton observations
            s = skel_present_f * s

        # positional encoding (same temporal positions for aligned streams)
        pos_v = self.pos_enc(v)
        pos_s = self.pos_enc(s)

        v = self.input_dropout(v + pos_v)
        s = self.input_dropout(s + pos_s)

        # key padding masks: True means unavailable as keys
        key_pad_v = ~valid_mask

        # source masking for skeleton:
        # only observed skeleton frames are visible as keys/values
        key_pad_s_source = ~skel_present

        # target skeleton stream lives on all valid timesteps,
        # because missing observations are represented by the learned token
        key_pad_s_target = ~valid_mask

        key_pad_v_safe = self.unmask_first_if_all(key_pad_v)
        key_pad_s_source_safe = self.unmask_first_if_all(key_pad_s_source)
        key_pad_s_target_safe = self.unmask_first_if_all(key_pad_s_target)

        # --------------------------------
        # Pairwise directional adaptations
        # --------------------------------

        # target V
        v_from_s = self.v_from_s(
            target=v,
            source=s,
            source_key_padding_mask=key_pad_s_source_safe,
        )

        # target S
        s_from_v = self.s_from_v(
            target=s,
            source=v,
            source_key_padding_mask=key_pad_v_safe,
        )

        # --------------------------------
        # Per-target fusion memory
        # --------------------------------
        v_fused = self.fuse_v(
            xs=[v_from_s],
            key_padding_mask=key_pad_v_safe,
        )   # [B, T, D]

        s_fused = self.fuse_s(
            xs=[s_from_v],
            key_padding_mask=key_pad_s_target_safe,
        )   # [B, T, D]

        # Which mask to use for final skeleton summary?
        # - valid_mask: treat missing-token steps as legitimate sequence members
        # - skel_present: only summarize over observed skeleton steps
        if self.skeleton_head_uses_valid_mask:
            s_head_mask = valid_mask
        else:
            s_head_mask = skel_present

        logits, last_hs = self.head(
            seqs=[v_fused, s_fused],
            masks=[valid_mask, s_head_mask],
        )

        return logits