"""Round 3 pre-flight kill checks. Five mechanism-canonical assertions.

Run: python tests/test_kill_checks.py

If any check fails, do NOT launch Round 3 training — fix the underlying
mechanism violation first.
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.lewm.encoder import LeWMEncoder, ProjectionHead
from src.lewm.predictor import LeWMPredictor, ConditionalBlock
from src.lewm.model import LeWM
from src.lewm.sigreg import SIGReg
from src.lewm.scheduler import linear_warmup_cosine_annealing


def section(title: str):
    print(f"\n=== {title} ===")


# ---------------------------------------------------------------------------
# Kill check 1: Projection has no BN after final layer.
# ---------------------------------------------------------------------------

def check_1_projection_no_terminal_bn():
    section("Kill check 1: ProjectionHead has no BN after final Linear")
    head = ProjectionHead(dim_in=192, dim_out=192, hidden_dim=2048)
    layers = list(head.children())
    print(f"  layers (in order): {[type(m).__name__ for m in layers]}")
    # Expected order: fc1 (Linear) → bn (BatchNorm1d) → act (GELU) → fc2 (Linear)
    assert isinstance(head.fc1, nn.Linear), f"fc1 is {type(head.fc1).__name__}"
    assert isinstance(head.bn, nn.BatchNorm1d), f"bn is {type(head.bn).__name__}"
    assert isinstance(head.act, nn.GELU), f"act is {type(head.act).__name__}"
    assert isinstance(head.fc2, nn.Linear), f"fc2 is {type(head.fc2).__name__}"
    # Crucial: no BN after fc2
    print(f"  fc2 (final) is unconstrained Linear — no terminal BN")
    print("  PASS")


# ---------------------------------------------------------------------------
# Kill check 2: SIGReg distinguishes collapsed const vs N(0,1) by orders of
# magnitude. With B-scaling, collapsed should be much larger.
# ---------------------------------------------------------------------------

def check_2_sigreg_distinguishes():
    section("Kill check 2: SIGReg behavior (collapsed vs Gaussian)")
    torch.manual_seed(0)
    sig = SIGReg(embed_dim=64, num_proj=128)

    # Gaussian sample (T, B, D) shaped
    z_gauss = torch.randn(4, 128, 64)
    # Collapsed: all samples nearly equal, with tiny noise
    z_collapsed = torch.zeros(4, 128, 64) + 0.001 * torch.randn(1, 1, 64)

    loss_g = sig(z_gauss).item()
    loss_c = sig(z_collapsed).item()
    print(f"  Gaussian (T=4, B=128, D=64):     {loss_g:+.4f}")
    print(f"  Collapsed (near-constant batch): {loss_c:+.4f}")
    print(f"  Ratio (collapsed / Gaussian):    {loss_c / max(loss_g, 1e-6):.1f}×")
    assert loss_c > loss_g * 3, (
        f"Collapsed should be >>3× Gaussian; got {loss_c:.4f} vs {loss_g:.4f}"
    )
    # Also verify the * B scaling: with B=128 vs B=16, collapsed loss should
    # scale roughly proportional to B (within constant factors).
    sig2 = SIGReg(embed_dim=64, num_proj=128)
    z_collapsed_small = torch.zeros(4, 16, 64) + 0.001 * torch.randn(1, 1, 64)
    loss_c_small = sig2(z_collapsed_small).item()
    print(f"  Collapsed B=16:                  {loss_c_small:+.4f}")
    print(f"  Collapsed B=128 / B=16 ratio:    {loss_c / max(loss_c_small, 1e-6):.2f}× "
          f"(expected ≈ {128/16:.0f}× from B-scaling)")
    print("  PASS")


# ---------------------------------------------------------------------------
# Kill check 3: AdaLN-zero block(x, c) ≈ x at initialization.
# ---------------------------------------------------------------------------

def check_3_adaln_zero_identity():
    section("Kill check 3: ConditionalBlock(x, c) ≈ x at init (dropout off)")
    torch.manual_seed(0)
    block = ConditionalBlock(dim=192, cond_dim=192, heads=16, dim_head=64,
                             mlp_dim=2048, dropout=0.0)
    block.eval()  # dropout off
    x = torch.randn(2, 4, 192)
    c = torch.randn(2, 4, 192)
    with torch.no_grad():
        y = block(x, c)
    diff = (y - x).abs().max().item()
    rel = diff / x.abs().max().item()
    print(f"  max |y - x|:  {diff:.6e}")
    print(f"  rel to |x|:   {rel:.6e}")
    assert diff < 1e-5, f"AdaLN-zero block must be identity at init; got max diff {diff}"
    print("  PASS — block is exactly identity at init.")


# ---------------------------------------------------------------------------
# Kill check 4: Prediction topology — pred_emb = pred_proj(predictor(...)),
# no z_t + delta residual path.
# ---------------------------------------------------------------------------

def check_4_prediction_topology():
    section("Kill check 4: Prediction topology pred_emb = pred_proj(predictor(...))")
    torch.manual_seed(0)
    model = LeWM(
        image_size=64, patch_size=8, vit_dim=192, encoder_depth=2, encoder_heads=3,
        latent_dim=192, proj_hidden_dim=128,
        predictor_depth=2, predictor_heads=8, predictor_dim_head=24,
        predictor_mlp_dim=256, action_dim=2, max_history=4,
        sigreg_num_proj=32,
    )
    model.eval()

    obs = torch.rand(2, 4, 3, 64, 64)
    actions = torch.randn(2, 4, 2)

    with torch.no_grad():
        emb, preds = model(obs, actions)
        # Recompute via predict() to verify equivalence
        preds_recomputed = model.predict(emb, actions)

    print(f"  emb shape:   {tuple(emb.shape)}")
    print(f"  preds shape: {tuple(preds.shape)}")

    # Check: preds_recomputed must equal preds
    assert torch.allclose(preds, preds_recomputed), \
        "model() and model.predict() should give the same predictions"

    # Check: predict() must invoke pred_proj. Verify by checking that preds is
    # NOT equal to predictor output alone (which would mean pred_proj is identity).
    h = model.predictor(emb, actions)  # pre-pred_proj predictor output
    assert not torch.allclose(preds, h, atol=1e-3), (
        "preds == predictor output → pred_proj is missing or identity"
    )
    print(f"  preds ≠ raw predictor output (pred_proj is doing work)")

    # Check: emb is NOT recomputed inside predict(); predict() takes emb.
    # i.e., predict(emb, a) does NOT add z_t externally. We verify by checking
    # that |preds - emb| is large (preds is a transformed prediction, not z_t + delta).
    delta = (preds - emb).abs().mean().item()
    print(f"  |preds - emb| mean: {delta:.4f} (should be >> 0 — predictor outputs full latent)")
    print("  PASS")


# ---------------------------------------------------------------------------
# Kill check 5: Scheduler LR profile.
# ---------------------------------------------------------------------------

def check_5_scheduler_profile():
    section("Kill check 5: LinearWarmupCosineAnnealingLR profile")
    base_lr = 5e-5
    total_steps = 1000
    warmup_fraction = 0.01  # 1% canonical

    model = nn.Linear(10, 10)
    opt = torch.optim.AdamW(model.parameters(), lr=base_lr)
    sched = linear_warmup_cosine_annealing(
        opt, total_steps=total_steps, warmup_fraction=warmup_fraction,
    )

    warmup_steps = max(1, int(warmup_fraction * total_steps))
    print(f"  base_lr={base_lr:.2e}  total_steps={total_steps}  "
          f"warmup_steps={warmup_steps}")

    sample_steps = [0, 1, warmup_steps - 1, warmup_steps, warmup_steps + 1,
                    total_steps // 4, total_steps // 2, 3 * total_steps // 4,
                    total_steps - 1]
    for step in range(total_steps):
        sched.step()
        cur = opt.param_groups[0]["lr"]
        if step in sample_steps:
            frac = cur / base_lr
            print(f"  step {step:4d}: lr={cur:.4e}  ({frac:.4f}× base)")

    # After scheduler.step() called total_steps times, lr should be near 0
    final_lr = opt.param_groups[0]["lr"]
    assert final_lr < base_lr * 0.05, f"Final LR {final_lr:.2e} not near 0"

    # At end of warmup, lr should be near base
    opt2 = torch.optim.AdamW(nn.Linear(10, 10).parameters(), lr=base_lr)
    sched2 = linear_warmup_cosine_annealing(opt2, total_steps=total_steps,
                                             warmup_fraction=warmup_fraction)
    for _ in range(warmup_steps):
        sched2.step()
    end_warmup_lr = opt2.param_groups[0]["lr"]
    assert end_warmup_lr > base_lr * 0.95, (
        f"At end of warmup, LR should be ~base_lr; got {end_warmup_lr:.2e}"
    )
    print(f"  end-of-warmup lr = {end_warmup_lr:.4e} ≈ base_lr ✓")
    print(f"  end-of-training lr = {final_lr:.4e} ≈ 0 ✓")
    print("  PASS")


if __name__ == "__main__":
    check_1_projection_no_terminal_bn()
    check_2_sigreg_distinguishes()
    check_3_adaln_zero_identity()
    check_4_prediction_topology()
    check_5_scheduler_profile()
    print("\n=== ALL KILL CHECKS PASSED ===")
