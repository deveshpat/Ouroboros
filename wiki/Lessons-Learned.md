# Lessons Learned
> Operational hard lessons. Load when debugging a recurring failure class.

| Symptom / Mistake | Fix Applied |
|---|---|
| `kaggle kernels pull` → 403 in CI | Use local `kernels push` instead — no pull needed |
| W&B step collision between rounds | `round_step_span = shard_step_estimate + 1` |
| Fixed `min_workers` causes deadlock when B has empty shard | Dynamic `min_shard_samples` pre-computation |
| Stage never closes with geometric remainder | `remaining < min_shard_samples` → declare stage complete |
| Coordinator triggers all workers even when some have nothing | Pre-compute projected shards, trigger only active workers |
| Solo mode with outer_lr=0.7 blends stale anchor into new weights | Direct weight promotion in solo mode (skip outer update) |
| `kaggle kernels push --accelerator` → unrecognized argument | Upgrade to `kaggle>=1.8.4`; add `--accelerator NvidiaTeslaT4` to push_args |
| Worker C quota exhausted → coordinator stalls forever | `triggered_at` + 13h timeout + attendance mechanism |
| Coordinator writes `triggered_workers` but push fails silently | `triggered_at=0` manual reset → immediate re-dispatch on next run |
| `kaggle==1.6.17` + `"accelerator": "nvidiaTeslaT4"` → still P100 | Root cause 1: `--accelerator` added in v1.8.4. Root cause 2: wrong cap. Fix: `kaggle>=1.8.4` + cap fix + runtime fast-fail. All verified. |
| `wandb==0.25.0` `resume="allow"` on finished run → ephemeral run | Per-round run IDs + `group=` for stage-level grouping |
| W&B dashboard unreadable with many overlapping runs | Unique `id` per round + `group` by stage |
| `--use_halt_gate` starts from random LoRA weights in DiLoCo path | `--resume_from_diloco_anchor` loads `diloco_state/anchor/` before DGAC training |
| `attn_implementation` crash | try/except fallback in `_safe_from_pretrained` |
| `use_mamba_kernels` old TF | retry without key in `_safe_from_pretrained` |
| `last_hidden_state` None | assert in all forward paths |
| Graceful session timeout | `make_timeout_checker()` using `_SCRIPT_START` |
| `conv1d` in LoRA | Excluded from LORA_TARGET_MODULES |
| OOM at val | `empty_cache()` + `val_batch_size=2` |
| Stage 1+ samples filtered by short seq_len | `--max_seq_len 1024` |
| Exploding gradients k≥2 | `--max_grad_norm 0.3` |
| mamba-ssm 2.x API break | Pinned to 1.2.2 |
| NCCL watchdog kills DDP val | `timedelta(hours=4)` + env var |
| BF16 emulation on T4 | `_amp_dtype` checks `cc >= (8,0)` |
