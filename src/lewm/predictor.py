"""Causal-masked transformer predictor with full DiT-style AdaLN-zero
action conditioning.

Round 3 canonical-mechanism rewrite. Matches official lucas-maes/le-wm
ARPredictor + ConditionalBlock + Attention + FeedForward.

Key mechanism (vs our pre-Round-3 version):
  - 6 modulation parameters per block: shift_msa, scale_msa, gate_msa,
    shift_mlp, scale_mlp, gate_mlp. Single Linear(dim, 6*dim) with
    Sequential(SiLU, Linear) — the final Linear is zero-init (weight & bias).
  - Multiplicative gates on the FULL residual delta:
        x = x + gate_msa * Attn(modulate(LN(x), shift_msa, scale_msa))
        x = x + gate_mlp * MLP(modulate(LN(x), shift_mlp, scale_mlp))
    At init the gates are zero → block(x, c) = x exactly. This is what
    "AdaLN-zero" means; gamma/beta-only modulation does NOT achieve it.
  - Decoupled head dim: heads × dim_head = inner_dim, NOT necessarily equal
    to model dim. Default: heads=16, dim_head=64, inner_dim=1024 — much
    higher attention capacity than dim/heads would give.
  - Predictor outputs the FULL next latent, not a residual delta. The
    rollout / loss must consume preds directly.
  - Position embedding init: torch.randn (std=1.0), not trunc_normal_(0.02).
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


def modulate(x: torch.Tensor, shift: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    """y = x * (1 + scale) + shift, per DiT convention."""
    return x * (1.0 + scale) + shift


# ---------------------------------------------------------------------------
# DiT-style action conditioner: produces (shift, scale, gate) × (msa, mlp)
# ---------------------------------------------------------------------------

class AdaLNModulation(nn.Module):
    """Sequential(SiLU, Linear(cond_dim, 6*dim)) with zero-init final Linear.

    At init: weight=0, bias=0 → all 6 modulation outputs are 0 → block becomes
    identity.
    """

    def __init__(self, cond_dim: int, dim: int):
        super().__init__()
        self.act = nn.SiLU()
        self.linear = nn.Linear(cond_dim, 6 * dim)
        nn.init.zeros_(self.linear.weight)
        nn.init.zeros_(self.linear.bias)

    def forward(self, c: torch.Tensor) -> tuple[torch.Tensor, ...]:
        # c: (..., cond_dim) → 6 tensors of shape (..., dim)
        h = self.linear(self.act(c))
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = h.chunk(6, dim=-1)
        return shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp


# ---------------------------------------------------------------------------
# Decoupled-head Attention
# ---------------------------------------------------------------------------

class DecoupledAttention(nn.Module):
    """Attention with inner_dim = heads × dim_head independent of model dim."""

    def __init__(self, dim: int, heads: int = 16, dim_head: int = 64,
                 dropout: float = 0.0):
        super().__init__()
        self.heads = heads
        self.dim_head = dim_head
        inner_dim = heads * dim_head
        self.scale = dim_head ** -0.5
        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)
        self.to_out = nn.Linear(inner_dim, dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, attn_mask: torch.Tensor | None = None) -> torch.Tensor:
        B, T, _ = x.shape
        qkv = self.to_qkv(x)  # (B, T, 3*inner)
        q, k, v = qkv.chunk(3, dim=-1)
        # split heads: (B, T, heads*dim_head) → (B, heads, T, dim_head)
        q = q.view(B, T, self.heads, self.dim_head).transpose(1, 2).contiguous()
        k = k.view(B, T, self.heads, self.dim_head).transpose(1, 2).contiguous()
        v = v.view(B, T, self.heads, self.dim_head).transpose(1, 2).contiguous()

        attn = (q @ k.transpose(-2, -1)) * self.scale
        if attn_mask is not None:
            attn = attn + attn_mask
        attn = attn.softmax(dim=-1)
        attn = self.dropout(attn)
        out = attn @ v  # (B, heads, T, dim_head)
        out = out.transpose(1, 2).contiguous().view(B, T, self.heads * self.dim_head)
        return self.dropout(self.to_out(out))


class FeedForward(nn.Module):
    def __init__(self, dim: int, mlp_dim: int = 2048, dropout: float = 0.0):
        super().__init__()
        self.fc1 = nn.Linear(dim, mlp_dim)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(mlp_dim, dim)
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.drop(self.fc2(self.drop(self.act(self.fc1(x)))))


# ---------------------------------------------------------------------------
# DiT-style ConditionalBlock (full 6-param AdaLN-zero with residual gates)
# ---------------------------------------------------------------------------

class ConditionalBlock(nn.Module):
    def __init__(self, dim: int, cond_dim: int, heads: int = 16,
                 dim_head: int = 64, mlp_dim: int = 2048, dropout: float = 0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim, elementwise_affine=False)
        self.attn = DecoupledAttention(dim, heads=heads, dim_head=dim_head,
                                       dropout=dropout)
        self.norm2 = nn.LayerNorm(dim, elementwise_affine=False)
        self.ff = FeedForward(dim, mlp_dim=mlp_dim, dropout=dropout)
        self.modulate = AdaLNModulation(cond_dim, dim)

    def forward(self, x: torch.Tensor, c: torch.Tensor,
                attn_mask: torch.Tensor | None = None) -> torch.Tensor:
        # c: (B, T, cond_dim) — per-position conditioning
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.modulate(c)
        # Attention residual with multiplicative gate on the delta
        x = x + gate_msa * self.attn(modulate(self.norm1(x), shift_msa, scale_msa),
                                     attn_mask=attn_mask)
        # MLP residual with multiplicative gate on the delta
        x = x + gate_mlp * self.ff(modulate(self.norm2(x), shift_mlp, scale_mlp))
        return x


# ---------------------------------------------------------------------------
# Action embedder: raw action → conditioning vector at model dim
# ---------------------------------------------------------------------------

class ActionEmbedder(nn.Module):
    """Embed actions to model dim via 2-layer SiLU MLP. Matches official intent
    (Conv1d-then-MLP) at the simpler concatenation grain."""

    def __init__(self, action_dim: int, embed_dim: int):
        super().__init__()
        self.fc1 = nn.Linear(action_dim, 4 * embed_dim)
        self.act = nn.SiLU()
        self.fc2 = nn.Linear(4 * embed_dim, embed_dim)

    def forward(self, a: torch.Tensor) -> torch.Tensor:
        return self.fc2(self.act(self.fc1(a)))


# ---------------------------------------------------------------------------
# Predictor (autoregressive over latent history, output is full next latent)
# ---------------------------------------------------------------------------

class LeWMPredictor(nn.Module):
    """Causal autoregressive predictor with DiT-style action conditioning.

    Input
    -----
    z : (B, T, latent_dim)  current and past latents
    a : (B, T, action_dim)  actions; action at position t is associated with z_t

    Output
    ------
    z_pred : (B, T, latent_dim)  predicted FULL next latent at each position;
             z_pred[:, t] is ẑ_{t+1} given (z_{0:t}, a_{0:t}). Causal mask
             enforces no peeking. Predictor outputs the next latent directly,
             NOT a residual delta — caller must NOT add z_t externally.
    """

    def __init__(
        self,
        latent_dim: int = 192,
        action_dim: int = 2,
        depth: int = 6,
        heads: int = 16,
        dim_head: int = 64,
        mlp_dim: int = 2048,
        dropout: float = 0.1,
        max_len: int = 16,
    ):
        super().__init__()
        self.latent_dim = latent_dim
        self.max_len = max_len
        # Position embedding init: torch.randn (std≈1.0) per official ARPredictor
        self.pos_embed = nn.Parameter(torch.randn(1, max_len, latent_dim))
        self.action_embed = ActionEmbedder(action_dim, latent_dim)
        self.blocks = nn.ModuleList([
            ConditionalBlock(latent_dim, cond_dim=latent_dim, heads=heads,
                             dim_head=dim_head, mlp_dim=mlp_dim, dropout=dropout)
            for _ in range(depth)
        ])
        # Final affine LayerNorm — matches official Transformer.norm
        self.norm = nn.LayerNorm(latent_dim)

    def forward(self, z: torch.Tensor, a: torch.Tensor) -> torch.Tensor:
        B, T, D = z.shape
        assert T <= self.max_len, f"T={T} exceeds max_len={self.max_len}"
        # Embed actions to conditioning space
        c = self.action_embed(a)  # (B, T, latent_dim)
        # Add position embedding to latents
        x = z + self.pos_embed[:, :T]
        # Causal mask: position i attends to 0..i
        mask = torch.full((T, T), float("-inf"), device=z.device, dtype=z.dtype)
        mask = torch.triu(mask, diagonal=1)
        for block in self.blocks:
            x = block(x, c, attn_mask=mask)
        x = self.norm(x)
        return x  # full next-latent prediction at each causal position
