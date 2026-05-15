# terminal_log.md — Project Ouroboros
> **Rolling buffer — last relevant run only.**
> Historical record moved to [wiki/SessionLog.md](wiki/SessionLog.md).
> Trim to ≤80 lines at each session update.

---

## Last Run — Azure H100 corrected DGAC epoch-0 checkpoint (2026-05-15)

Run: `Azure H100 SCUS DGAC full budgeted` in W&B.

```text
Loaded 36906 train / 1940 val from data/coconut_v1
[GPU] NVIDIA H100 NVL  cc=sm90  VRAM=100GB  amp_dtype=bfloat16
flash-attn available: using flash_attention_2
mamba CUDA kernels: fast path ACTIVE (verified at bootstrap)
[perf] 100GB VRAM detected, but keeping gradient checkpointing for this high-depth latent workload.
DGAC HaltGate: d_model=2560  params=5121

[DGAC] Loading DiLoCo anchor from WeirdRunner/Ouroboros/diloco_state/anchor as base weights for Phase 3.4 DGAC training.
[diloco] Loaded anchor weights from diloco_state/anchor
[diloco] Loaded halt gate from diloco_state/anchor/halt_gate.pt
[DGAC] Anchor load complete. If the anchor contains halt_gate.pt, HaltGate was restored; otherwise it remains zero-init. Optimizer starts fresh unless this is eval-only.

Stage 10/10  10 latent pass(es)  + DGAC
Epochs: 3  Steps/epoch: 1154  Total: 3462
step=  1150 s=10 ep=0 ce=0.3538 gn=0.3338
[timeout] Skipping val/gen at epoch 0 - 299min remaining (< 720min val budget).
[ckpt] saved -> runs/azure_h100_dgac/stage_10/checkpoint-0001154  acc=None  ce=None
[hub] uploaded runs/azure_h100_dgac/stage_10/checkpoint-0001154 -> WeirdRunner/Ouroboros
```

Result: corrected HaltGate load path is live and the H100 run produced an epoch-0 DGAC checkpoint with `training_state.pt`, adapter weights, and `halt_gate.pt`. This is checkpoint evidence only, not a quality pass, because val/gen were skipped. Evaluate `runs/azure_h100_dgac/stage_10/checkpoint-0001154` or resume from it explicitly before making benchmark/packaging decisions.
