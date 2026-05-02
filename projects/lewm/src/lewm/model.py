"""LeWM model and loss — Round 3 canonical-mechanism form.

Key topology (matches official lucas-maes/le-wm/jepa.py):

  ENCODE     : pixels → ViT → CLS token → projector(MLP+BN-in-middle) → emb
  PREDICT    : (z_history, a) → predictor → pred_proj(MLP+BN-in-middle) → pred
  LOSS       : MSE(pred, target_emb) + λ * SIGReg(emb in (T,B,D) layout)

  - SIGReg sees the post-projector embedding, NOT the post-pred_proj prediction.
  - Predictor outputs a FULL next latent (not a residual delta to z_t).
  - Asymmetric projection: predictions go through projector + pred_proj;
    targets go through projector only. This is the BYOL/SimSiam-style anti-
    collapse mechanism.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from .encoder import LeWMEncoder, ProjectionHead
from .predictor import LeWMPredictor
from .sigreg import SIGReg


class LeWM(nn.Module):
    def __init__(
        self,
        # encoder
        image_size: int = 64,
        patch_size: int = 8,
        in_chans: int = 3,
        vit_dim: int = 192,
        encoder_depth: int = 12,
        encoder_heads: int = 3,
        latent_dim: int = 192,
        proj_hidden_dim: int = 2048,
        # predictor
        predictor_depth: int = 6,
        predictor_heads: int = 16,
        predictor_dim_head: int = 64,
        predictor_mlp_dim: int = 2048,
        predictor_dropout: float = 0.1,
        action_dim: int = 2,
        max_history: int = 16,
        # SIGReg
        sigreg_num_proj: int = 1024,
        sigreg_knots: int = 17,
    ):
        super().__init__()
        self.encoder = LeWMEncoder(
            image_size=image_size, patch_size=patch_size, in_chans=in_chans,
            vit_dim=vit_dim, depth=encoder_depth, num_heads=encoder_heads,
            latent_dim=latent_dim, proj_hidden_dim=proj_hidden_dim,
        )
        # Encoder.proj is the official's "projector" MLP. We add a separate
        # pred_proj of identical shape, applied only to predictor outputs.
        self.pred_proj = ProjectionHead(
            dim_in=latent_dim, dim_out=latent_dim, hidden_dim=proj_hidden_dim,
        )
        self.predictor = LeWMPredictor(
            latent_dim=latent_dim, action_dim=action_dim,
            depth=predictor_depth, heads=predictor_heads,
            dim_head=predictor_dim_head, mlp_dim=predictor_mlp_dim,
            dropout=predictor_dropout, max_len=max_history,
        )
        self.sigreg = SIGReg(
            embed_dim=latent_dim,
            num_proj=sigreg_num_proj,
            knots=sigreg_knots,
        )
        self.latent_dim = latent_dim

    def encode(self, obs: torch.Tensor) -> torch.Tensor:
        """obs: (B, T, C, H, W) or (B, C, H, W). Returns post-projector emb."""
        return self.encoder(obs)

    def predict(self, emb: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
        """emb: (B, T, D)  actions: (B, T, A). Returns post-pred_proj predictions
        (B, T, D); position t is the predicted next emb ẑ_{t+1}."""
        h = self.predictor(emb, actions)
        return self.pred_proj(h)

    def forward(
        self, obs: torch.Tensor, actions: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Full forward: encode obs, run predictor, project predictions.

        Path C dataset asymmetry:
          obs shape:     (B, T_obs, C, H, W) where T_obs = history_size + 1
          actions shape: (B, T_a, A)         where T_a   = history_size

          Encoder pass: emb = encoder(obs)        — shape (B, T_obs, D)
          Predictor input: emb[:, :T_a]           — the "history" portion
          Predictor input action_tokens: actions  — (B, T_a, A)
          Predictor output: preds                 — (B, T_a, D)
          Pred at position t predicts emb[t+1]    — i.e. emb[:, 1:T_a+1] = emb[:, 1:]

        For backward compatibility (T_obs == T_a, single-frame-per-token), the
        predictor still works; emb[:, :T_a] = emb (full window).

        Returns (emb, preds) where:
          emb:   (B, T_obs, D)  — full post-projector encoder embeddings
          preds: (B, T_a,   D)  — post-pred_proj predictions
        """
        emb = self.encode(obs)
        T_a = actions.size(1)
        history_emb = emb[:, :T_a]
        preds = self.predict(history_emb, actions)
        return emb, preds


def lewm_loss(
    emb: torch.Tensor,
    preds: torch.Tensor,
    sigreg_module: SIGReg,
    lambda_sigreg: float = 0.09,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Path C canonical loss.

      emb:   (B, T_obs, D)        post-projector encoder embeddings
      preds: (B, T_a, D)          post-pred_proj predictions, T_a = T_obs - 1

      L_pred = MSE(preds, emb[:, 1:])    one-step-ahead, teacher-forced
      L_sigreg = SIGReg(emb in (T_obs, B, D)) per-step + B-scaled
      L = L_pred + λ * L_sigreg
    """
    T_a = preds.size(1)
    pred_loss = (preds - emb[:, 1:T_a + 1]).pow(2).mean()
    sigreg_loss = sigreg_module(emb.transpose(0, 1).contiguous())
    total = pred_loss + lambda_sigreg * sigreg_loss
    return total, pred_loss, sigreg_loss
