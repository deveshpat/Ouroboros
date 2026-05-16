# GPU Guardrails
> Load this page when a worker gets assigned the wrong GPU, or to understand the P100 fast-fail.

---

## The P100 Problem

Kaggle assigns P100 (sm60) when:
- T4 quota is exhausted (30h/week per account)
- `--accelerator` flag is absent or uses wrong version of kaggle CLI
- `"accelerator"` JSON field has wrong capitalisation

P100 consequence: Bootstrap tries to compile mamba_ssm from source.
Compile time: ~30 min on P100. Kaggle cancels kernel around the same time. Zero training.

---

## Three-Layer Defence (all three required)

| Layer | Location | Value |
|---|---|---|
| JSON metadata field | `_build_kaggle_kernel_metadata()` | `"accelerator": "NvidiaTeslaT4"` (capital N) |
| CLI flag | `_trigger_single_worker()` push_args | `--accelerator NvidiaTeslaT4` |
| Runtime fast-fail | `main()` in `-m ouroboros.coconut` | `cc < (7,5)` → exit |

**Requires `kaggle>=1.8.4`.** The `--accelerator` CLI flag was added in v1.8.4 (PR #907).
Prior pin `kaggle==1.6.17` silently discarded the field. Verified working since Session 21.

---

## Runtime Fast-Fail Flow

```python
# In main(), DiLoCo mode only, before any training
if args.diloco_mode and device.type == "cuda":
    cc = torch.cuda.get_device_capability(device)
    if cc < (7, 5):          # sm75 = T4 minimum
        _diloco_reset_triggered_at(hf_token, diloco_state_repo)
        diloco_push_signal(worker_id, stage_k, round_n, github_token, signal_repo)
        sys.exit(0)
```

### `_diloco_reset_triggered_at()` steps
1. `hf_hub_download` → `diloco_state/round_state.json`
2. Set `triggered_at = 0.0`, `last_updated = time.time()`
3. `api.upload_file` → Hub
4. Print confirmation

`triggered_at=0` is the canonical "unconfirmed dispatch" signal. Coordinator sees it on next run (≤30 min) and re-dispatches immediately to a (hopefully) T4 session.

---

## GPU Capability Reference

| GPU | sm | VRAM | Status |
|---|---|---|---|
| T4 | sm75 | 16 GB | ✅ Target |
| V100 | sm70 | 16/32 GB | ⚠️ No cached wheels |
| P100 | sm60 | 16 GB | ❌ Rejected by guard |
| A100 | sm80 | 40/80 GB | ✅ Target; GC may auto-disable for smaller latent workloads |
| H100 | sm90 | 80/100 GB | ✅ Target; high-depth DGAC can keep GC enabled |

Gradient checkpointing is no longer a pure VRAM threshold. `_should_auto_disable_gradient_checkpointing()` may disable it at `total_vram_gb >= 40.0` for smaller latent workloads, but keeps it enabled when latent-token pressure is high or `--use_halt_gate` runs deep DGAC (`max_stage >= 6`). `OUROBOROS_FORCE_GRAD_CHECKPOINT=1` also prevents auto-disable. Latest Azure H100 evidence printed `100GB VRAM detected, but keeping gradient checkpointing for this high-depth latent workload.`

---

## Verification Status

- No P100 assignment observed since Session 21 fix deployment.
- Guard tested implicitly: `triggered_at=0` recovery path verified (Session 19).
- If P100 appears again: check Kaggle quota remaining for the affected account.


## Azure H100 DGAC Note

The Azure H100 path is not part of the Kaggle P100/T4 dispatch guardrail, but it still uses the same bootstrap contract: BF16 on sm90, `flash_attention_2`, and verified Mamba CUDA kernels before model load. The 2026-05-15 run saved `runs/azure_h100_dgac/stage_10/checkpoint-0001154` after restoring the DiLoCo anchor and `halt_gate.pt`. Treat this checkpoint as training evidence only until a separate val/gen quality gate runs.
