# MiniMax-M2.7 LoRA Sweep Report (2026-05-26)

Branch: `besimray/fix/minimax-lora-4gpu-run`

## Setup used for all short probes

- Hardware: 4 GPUs, `ep_size=4`
- Recipe: `examples/llm_finetune/minimax_m2/minimax_m2.7_hellaswag_lora.yaml`
- Shared overrides:
  - `local_batch_size=2`
  - `global_batch_size=512`
  - `max_steps=20` (short probe runs)
  - activation checkpointing enabled
- Fast-sweep mode was used for recent trials:
  - `TRAIN_DISABLE_CHECKPOINT=1` to avoid long end-of-run checkpoint stall

## Results so far

- `LR=3e-4`, `dim=32`, cosine scheduler, 20 steps
  - Log: `logs/minimax_lr_3e4_20260525_221853.log`
  - Loss: `13.0929 -> 9.3566` (min `9.3566` at step 19)
  - Avg TPS: `1312.87`
  - Note: strong monotonic drop over 20 steps.

- `LR=4e-4`, `dim=32`, cosine scheduler, 20 steps
  - Log: `logs/minimax_lr_4e4_fast_20260525_224502.log`
  - Loss: `13.0872 -> 8.9504` (min `8.9180` at step 16, then slight rebound)
  - Avg TPS: `1282.19`
  - Note: best observed short-run loss, but late-step rebound.

- `LR=4e-4`, `dim=32`, fixed LR, in-progress when report captured
  - Log: `logs/minimax_lr_4e4_fixed_20260525_230303.log`
  - Captured so far: `13.0949 -> 9.2568` by step 17
  - Min so far: `9.0458` at step 11, then drift upward
  - Avg TPS so far: `1231.33`
  - Note: indicates fixed `4e-4` is likely too aggressive late in run.

- `LR=3e-4`, `dim=64`, `alpha=128`, fixed LR, currently running
  - Log: `logs/minimax_lr_3e4_rank64_fixed_20260525_232013.log`
  - Early signal (first 3 steps): `13.0899 -> 12.6473`
  - Avg TPS so far: `1099.81`
  - Note: slower throughput than rank-32 runs, expected due to higher LoRA rank.

## Practical takeaways

- Best LR region for these short probes appears to be around `3e-4` to `4e-4`.
- Fixed `4e-4` shows late instability (loss rebound + rising grad norm in logs).
- Next sensible baseline:
  - use `3e-4` as base LR
  - compare `dim=32` vs `dim=64` (and possibly `dim=96`) with fixed LR in short probes
  - once best pair is chosen, run longer with scheduler for convergence run.

## Notes for resuming elsewhere

- Main launcher script: `scripts/run_minimax_m27_screen.sh`
- Useful fast-sweep flags:
  - `ENABLE_LR_SCHEDULER=0` (fixed LR for tiny probes)
  - `TRAIN_DISABLE_CHECKPOINT=1` (avoid end-of-run checkpoint stall)
  - `HF_OFFLINE=1` (use local HF cache)
