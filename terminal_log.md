# terminal_log.md — Project Ouroboros
> **Rolling buffer — last relevant run only.**
> Historical record moved to [wiki/SessionLog.md](wiki/SessionLog.md).
> Trim to ≤80 lines at each session update.

---

## Last Run — Azure H100 corrected DGAC epoch-0 checkpoint promoted to canonical anchor (2026-05-15)

Run: `Azure H100 SCUS DGAC full budgeted` in W&B, followed by Hub anchor promotion.

```text
Loaded 36906 train / 1940 val from data/coconut_v1
[GPU] NVIDIA H100 NVL  cc=sm90  VRAM=100GB  amp_dtype=bfloat16
flash-attn available: using flash_attention_2
mamba CUDA kernels: fast path ACTIVE (verified at bootstrap)
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

[promote] promoted_at=2026-05-15T11:24:25+00:00
[promote] source_revision=374a9a32d81242224465b786d62aaef7564639e6
[promote] source_checkpoint=runs/azure_h100_dgac/stage_10/checkpoint-0001154
[promote] anchor_prefix=diloco_state/anchor
[promote] copied adapter_model.safetensors, adapter_config.json, halt_gate.pt
[promote] terminal_stage=10 total_train_samples=36906 mark_dgac_complete=true
```

Result: corrected HaltGate load path is live, the H100 run produced an epoch-0 DGAC checkpoint, and that checkpoint is now the canonical `diloco_state/anchor`. This is still not a quality pass because val/gen were skipped during the Azure run. Next action: run `dgac-anchor-eval` against the promoted anchor and compare `val_ce`, `val_acc`, generation samples, and DGAC diagnostics against the pre-DGAC terminal-anchor baseline.
