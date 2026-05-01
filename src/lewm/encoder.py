"""ViT encoder + 1-layer MLP+BN projection head.

Per LeWM App. D: encoder is ViT-Tiny (dim=192, depth=12, heads=3, patch=14).
The projection step is required because the ViT's final LayerNorm would
otherwise prevent SIGReg from being optimized effectively (LayerNorm fixes
per-token scale, masking the marginal Gaussianization that SIGReg targets).
"""

from __future__ import annotations

import torch
import torch.nn as nn


# ---------------------------------------------------------------------------
# ViT building blocks (deliberately self-contained; no external ViT lib)
# ---------------------------------------------------------------------------

class PatchEmbed(nn.Module):
    def __init__(self, image_size: int = 64, patch_size: int = 8, in_chans: int = 3, dim: int = 192):
        super().__init__()
        assert image_size % patch_size == 0
        self.image_size = image_size
        self.patch_size = patch_size
        self.num_patches = (image_size // patch_size) ** 2
        self.proj = nn.Conv2d(in_chans, dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, C, H, W) -> (B, num_patches, dim)
        x = self.proj(x)
        x = x.flatten(2).transpose(1, 2)
        return x


class MLP(nn.Module):
    def __init__(self, dim: int, mlp_ratio: float = 4.0, dropout: float = 0.0):
        super().__init__()
        hidden = int(dim * mlp_ratio)
        self.fc1 = nn.Linear(dim, hidden)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden, dim)
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.drop(self.fc2(self.drop(self.act(self.fc1(x)))))


class Attention(nn.Module):
    def __init__(self, dim: int, num_heads: int, dropout: float = 0.0):
        super().__init__()
        assert dim % num_heads == 0, f"dim {dim} not divisible by num_heads {num_heads}"
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.qkv = nn.Linear(dim, dim * 3)
        self.proj = nn.Linear(dim, dim)
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, attn_mask: torch.Tensor | None = None) -> torch.Tensor:
        B, N, D = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)  # each (B, H, N, head_dim)
        # Force contiguous layout: workaround for PyTorch issue #163597 (MPS SDPA
        # regression with non-contiguous tensors at head_dim ∈ {64, 96, 128}). Cheap
        # on CPU/CUDA; necessary on MPS 2.8.x.
        q, k, v = q.contiguous(), k.contiguous(), v.contiguous()
        attn = (q @ k.transpose(-2, -1)) * self.scale
        if attn_mask is not None:
            attn = attn + attn_mask
        attn = attn.softmax(dim=-1)
        attn = self.drop(attn)
        x = (attn @ v).transpose(1, 2).reshape(B, N, D)
        x = self.drop(self.proj(x))
        return x


class TransformerBlock(nn.Module):
    """Pre-norm Transformer block with standard LayerNorm (used in encoder)."""

    def __init__(self, dim: int, num_heads: int, mlp_ratio: float = 4.0, dropout: float = 0.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = Attention(dim, num_heads, dropout)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = MLP(dim, mlp_ratio, dropout)

    def forward(self, x: torch.Tensor, attn_mask: torch.Tensor | None = None) -> torch.Tensor:
        x = x + self.attn(self.norm1(x), attn_mask=attn_mask)
        x = x + self.mlp(self.norm2(x))
        return x


# ---------------------------------------------------------------------------
# ViT encoder (ViT-Tiny: dim=192, depth=12, heads=3, patch=8 for synthetic 64x64)
# ---------------------------------------------------------------------------

class ViTEncoder(nn.Module):
    def __init__(
        self,
        image_size: int = 64,
        patch_size: int = 8,
        in_chans: int = 3,
        dim: int = 192,
        depth: int = 12,
        num_heads: int = 3,
        mlp_ratio: float = 4.0,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.patch_embed = PatchEmbed(image_size, patch_size, in_chans, dim)
        n_patches = self.patch_embed.num_patches
        self.cls_token = nn.Parameter(torch.zeros(1, 1, dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, n_patches + 1, dim))
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        self.blocks = nn.ModuleList(
            [TransformerBlock(dim, num_heads, mlp_ratio, dropout) for _ in range(depth)]
        )
        self.norm = nn.LayerNorm(dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, C, H, W) -> CLS embedding (B, dim)
        B = x.shape[0]
        x = self.patch_embed(x)
        cls = self.cls_token.expand(B, -1, -1)
        x = torch.cat([cls, x], dim=1)
        x = x + self.pos_embed
        for block in self.blocks:
            x = block(x)
        x = self.norm(x)
        return x[:, 0]  # CLS only


# ---------------------------------------------------------------------------
# 2-layer MLP projection — canonical form per official lucas-maes/le-wm.
#   Linear(in -> hidden) -> BatchNorm1d(hidden) -> GELU -> Linear(hidden -> out)
# Round 3 fix:
#   - BN sits in the MIDDLE, not at the end. Final Linear is unconstrained,
#     so SIGReg sees the actual learned distribution rather than a BN-faked
#     unit-Gaussian.
#   - Used for both the encoder's projector AND a separate pred_proj on
#     predictor outputs (asymmetric projection, BYOL/SimSiam-style anti-
#     collapse).
# ---------------------------------------------------------------------------

class ProjectionHead(nn.Module):
    """2-layer MLP with BatchNorm1d on the hidden. No BN after final Linear."""

    def __init__(self, dim_in: int, dim_out: int, hidden_dim: int = 2048):
        super().__init__()
        self.fc1 = nn.Linear(dim_in, hidden_dim)
        self.bn = nn.BatchNorm1d(hidden_dim)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden_dim, dim_out)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (..., dim_in) — flatten leading dims for BN1d
        leading = x.shape[:-1]
        x = x.reshape(-1, x.size(-1))
        x = self.fc1(x)
        x = self.bn(x)
        x = self.act(x)
        x = self.fc2(x)
        x = x.reshape(*leading, x.size(-1))
        return x


# ---------------------------------------------------------------------------
# Full LeWM encoder: ViT + projection
# ---------------------------------------------------------------------------

class LeWMEncoder(nn.Module):
    def __init__(
        self,
        image_size: int = 64,
        patch_size: int = 8,
        in_chans: int = 3,
        vit_dim: int = 192,
        depth: int = 12,
        num_heads: int = 3,
        mlp_ratio: float = 4.0,
        latent_dim: int = 192,
        proj_hidden_dim: int = 2048,
    ):
        super().__init__()
        self.vit = ViTEncoder(
            image_size=image_size,
            patch_size=patch_size,
            in_chans=in_chans,
            dim=vit_dim,
            depth=depth,
            num_heads=num_heads,
            mlp_ratio=mlp_ratio,
        )
        self.proj = ProjectionHead(vit_dim, latent_dim, hidden_dim=proj_hidden_dim)
        self.latent_dim = latent_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Accepts (B, C, H, W) or (B, T, C, H, W). Returns (B, latent_dim) or (B, T, latent_dim)."""
        squeeze_t = False
        if x.dim() == 4:
            x = x.unsqueeze(1)
            squeeze_t = True
        B, T, C, H, W = x.shape
        x = x.reshape(B * T, C, H, W)
        z = self.vit(x)
        z = self.proj(z)
        z = z.reshape(B, T, -1)
        if squeeze_t:
            z = z.squeeze(1)
        return z
