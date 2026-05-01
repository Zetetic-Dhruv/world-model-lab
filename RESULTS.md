# LeWM-vanilla — training run results

## Round 1 (2026-05-01, ~18:00)

**Config:** 500 episodes × 30 steps, batch=32, λ=1.0, M=256, 3 epochs, CPU, 12 min wallclock.

**Headline:** L_pred 1.59 → 0.33; z_std 0.74 → 0.74; linear probe R² = 0.75 / 0.85 for (x, y); rollout MSE flat at ~1.0.

**Conclusion (at the time):** vanilla LeWM working; BN running-stats poisoning hypothesized as cause of flat rollout MSE.

## Round 2 (2026-05-01, ~18:50)

**Config:** identical to Round 1 + Fixes A (BN reset at step 400, gated on z_std > 0.5), C (`.contiguous()` in attention), D (NaN supervisor — threaded snapshot/watchdog with 5% NaN-rate abort threshold). CPU. Wallclock 11 min.

**Supervisor stats:** 0 NaN events, 0 recoveries, 0 escalations, 3 snapshots in ring buffer at end. As expected on CPU.

**BN reset:** verified fired at step 400 with z_std=0.515. `reset_running_stats()` returned ok.

**The result that matters:** model weights at step 1263 are **bitwise-identical** to Round 1.
```
mean abs diff in proj.fc.weight: 0.000000e+00
v1 BN running_var: 2.6352e-05
v2 BN running_var: 2.6352e-05  (identical to Round 1!)
```

The reset set `running_var = 1.0`, then BN's EMA re-tracked the same near-zero `fc_out` batch variance over the subsequent steps, and converged back to 2.6e-5.

**Round 2 reveals Round 1's diagnosis was wrong.**

## Re-diagnosis

The BN running-stats poisoning hypothesis was a *symptom*, not the cause. The actual cause:

- The encoder collapses to near-constant output (`vit_out` batch-std ≈ 3e-4, measured earlier).
- BN with `running_var ≈ 2.6e-5` divides by ≈ 0.005, amplifying ~200×.
- This amplification is what produces the apparent z_std ≈ 0.76 in training-mode forward passes (where BN uses batch stats, which are also amplified consistently with the running stats by the same near-zero divisor).
- The encoder's residual position signal survives the collapse (it's small but present). BN amplifies it along with the noise. The linear probe fits the signal+noise mixture and recovers R² ≈ 0.85.
- The predictor sees the same amplified noise/signal during training and eval — internally consistent. But rollout MSE stays high because amplified noise dominates the actual position signal beyond a few steps.

**SIGReg is not catching this.** SIGReg evaluates post-projection. After BN amplification, the marginal *looks* approximately Gaussian (because amplified small-scale variation is approximately normal). SIGReg returns ≈ 0.25, which we interpreted as "stable." It's actually "fooled."

**Why Fix A failed:** Resetting BN running_var doesn't help — the EMA reconverges to the same near-zero batch_var because the encoder hasn't changed. Fix A was treating BN as the patient when BN is the symptom.

## Status of fixes

| Fix | Implemented | Effect | Verdict |
|---|---|---|---|
| A: BN reset at step 400 | yes | ran successfully, no measurable effect | ineffective; wrong diagnosis |
| C: `.contiguous()` in attention | yes | no effect on CPU; may help on MPS (untested due to MPS NaN abort) | retain (defensive) |
| D: NaN supervisor (threaded) | yes | 0 events on CPU; verified working on MPS smoke (recovered 3×, escalated correctly at 13% NaN rate) | retain — works as designed |

## Real problem

**Encoder collapse is not being caught by SIGReg.** This is a deeper issue than the MPS bug, and it sits at the heart of the JEPA construction. The remediation requires re-thinking either:

1. Where SIGReg evaluates (currently post-projection; consider pre-projection to see the un-amplified marginal)
2. The projection head design (BN may not be the right normalization here)
3. The relative weighting (λ might need to be much higher, or scheduled)
4. The encoder capacity (12-layer ViT-Tiny may be too deep for 64×64 particle scenes — easier to collapse than to learn)

This question is now the more interesting research target than the MPS NaN.

## Status of supervisor

The supervisor (Fix D) is retained as infrastructure regardless. It worked correctly on both substrates:
- CPU: 0 events, 0 recoveries — clean overhead-free pass-through.
- MPS: detected NaN at 13% rate, rolled back successfully 3 times, escalated to abort at the 5% threshold and dumped the failing batch + state to `runs/smoke-mps/incidents/abort_0001_step27.pt`.

The supervisor is now positioned as a separate concern — a substrate-agnostic execution-level diagnostic + recovery layer. See `next_steps.md` (TBD) for the broader RTOS-utility framing discussion.

## Two things to investigate next, ordered

1. **Encoder collapse** — what's actually happening? Diagnostic: log `vit_out` batch-std, `fc_out` batch-std, BN batch_var, BN running_var continuously throughout training, not just at end. Run with smaller encoder (depth=4) and larger λ. This question precedes the rollout-quality question.
2. **MPS NaN inquiry** — the URS-structured discovery task you proposed. The supervisor handles symptoms; the inquiry is into root cause(s) and what would constitute a structurally valid fix — including potential PyTorch contribution.

These are independent investigations with potential cross-pollination (a generic execution-level RTOS utility would help with both).
