# Project Status — Coconut-Ouroboros
> **This page is the body. `BLUEPRINT.md` is the index. Load this page after BLUEPRINT.md.**
> Last updated: 2026-05-09

---

## Curriculum Progress

| Stage | Status | Notes |
|---|---|---|
| 0 — CoT warmup | ✅ COMPLETE | ce=0.4041, acc=0.0222 |
| 1 — 1 latent pass | ✅ COMPLETE | ce=0.4912, acc=0.0444 |
| 2 — 2 latent passes | ✅ COMPLETE | 6 rounds, A+B, ~36 906 samples |
| 3 — 3 latent passes | ✅ COMPLETE | 6 rounds; A solo r2–r3 (B only signals); C rejoined r5 |
| 4 — 4 latent passes | ✅ COMPLETE | 1 round, A+B+C all active |
| 5 — 5 latent passes | ✅ COMPLETE | 1 round, A+B+C all active |
| 6 — 6 latent passes | ✅ COMPLETE | 1 round, A+B+C all active |
| 7 — 7 latent passes | ✅ COMPLETE | 1 round, A+B+C all active |
| 8 — 8 latent passes | ✅ COMPLETE | Completed and superseded by stage 9/10 Hub state |
| 9 — 9 latent passes | ✅ COMPLETE | Stage 9 visible in W&B as complete before stage 10 dispatch |
| 10 — 10 latent passes | ✅ COMPLETE | Stage 10 round 2 aggregated A/B/C (8,665 + 8,665 + 8,664 samples), reached 36,906/36,906 samples, uploaded terminal DiLoCo anchor, and entered DGAC manual gate |

**Compute mode:** DiLoCo dynamic workers with attendance/waiting-mode fallback.
**Current gate:** Stage 10 DiLoCo is complete and the final anchor eval-only quality review passed. DGAC is cleared for explicit workflow launch; coordinator cron must not dispatch stage 11.


---

## Engineering Architecture Status

| Track | Status | Notes |
|---|---|---|
| Training monolith extraction | ✅ COMPLETE | `jamba_coconut_finetune.py` is a thin compatibility adapter delegating into `ouroboros.train` and related package modules. |
| Kaggle launch seam | ✅ COMPLETE | Notebook remains a thin adapter and preserves `!torchrun` shell magic. |
| Runtime signal tracking cleanup | ✅ COMPLETE | Generated `signals/*.json` files are ignored; only `signals/.gitkeep` belongs in source control; the signal mechanism remains active as the coordinator doorbell. |
| Coordinator zero-drift extraction | ✅ COMPLETE | `diloco_coordinator.py` is a thin adapter; aggregation, state, dispatch, and orchestration live under `ouroboros.diloco.*`. |
| Planning artifact retirement | ✅ COMPLETE | Completed PRDs/plans have been promoted into wiki documentation and removed from `prds/` and `plans/`. |
| Kaggle CPU/API workflow validation | ✅ COMPLETE | CPU-smoke validation mode, repo/runtime seams, CPU metadata dispatch, fake coordinator loop tests, manual API validation docs, and remote Hub artifact verification are in place. |
| DGAC readiness CPU-smoke gate | ✅ PASSED LIVE | GitHub Actions `coordinate #272` ran `workflow_validate=cpu-smoke`, defaulted to Worker A, pushed Kaggle kernel version 39, Kaggle exited before `torchrun`, published Hub artifacts, and coordinator verified validation run `25377312407-1`. |

Canonical architecture record: [Architecture-Extraction](Architecture-Extraction.md).
Canonical execution protocol: [Engineering-Workflow](Engineering-Workflow.md).

---

## Immediate Next Steps

1. **Launch DGAC explicitly via workflow** — use GitHub Actions → `coordinate` → **Run workflow** with `kaggle_run_mode=dgac-train`, `force_worker_ids=A`, `skip_trigger=false`, `dry_run=false`, and empty `workflow_validate`. This pushes one GPU Kaggle notebook, loads the terminal anchor, writes local checkpoints under `runs/stage3_dgac`, and pushes Hub checkpoints under `runs/stage3_dgac`.
2. **Monitor DGAC stop/rollback criteria** — halt if CE spikes materially above the Stage 10 terminal anchor baseline (`val_ce=0.4863`), generated samples collapse/repeat, GPU errors appear, or no healthy checkpoint is pushed before session timeout.
3. **Review DGAC research signal** — inspect HaltGate behavior / halt-step distribution and compare post-DGAC `val_ce`, `val_acc`, and generations against the terminal anchor baseline.
4. **Optional polish** — quiet expected Hugging Face 404 polling noise during validation artifact eventual consistency; not blocking because coordinator already verifies successfully.

---


## Latest Validation Evidence

| Gate | Result | Evidence |
|---|---|---|
| Stage 10 terminal aggregation | ✅ PASS | 2026-05-09 coordinator run read `stage=10 round=2`, found A/B/C ready, aggregated 3 workers on CPU, logged `samples_this_round=25994`, `total_samples_stage=36906`, `pct_stage_done=100`, and printed `Stage 10 COMPLETE ... Entering DGAC manual gate.` |
| Stage 11 dispatch prevention | ✅ PASS | Same run printed `Stage 10 is terminal for DiLoCo` and `Done (DGAC manual gate)` after uploading the stage 10 round 2 anchor. |
| Stage 10 terminal anchor eval-only | ✅ PASS | 2026-05-09 eval loaded `diloco_state/anchor`, reported `val_ce=0.4863`, `val_acc=0.0889`, coherent generation samples for arithmetic/code/factual/explanatory prompts, and `Mean UWR=0.733`; `k_actual=10` is expected before DGAC because HaltGate starts zero-init. |
| GitHub Actions → Kaggle API dispatch | ✅ PASS | `coordinate #272`, `workflow_validate=cpu-smoke`, Worker A default, Kaggle kernel version 39 pushed |
| Kaggle CPU validation branch | ✅ PASS | Notebook printed `[workflow-validate] CPU smoke validation complete`; exited with `SystemExit: 0` before `torchrun` |
| Remote Hub validation artifact | ✅ PASS | `diloco_state/workflow_validation/25377312407-1/worker_A_status.json` and `worker_A_report.json` published |
| Coordinator artifact verification | ✅ PASS | Coordinator printed `CPU-smoke validation verified via remote Hub artifacts: ['A']` |
| GPU/training-state safety | ✅ PASS | Report shows `gpu_requested=false`, `torchrun_requested=false`, `published=true`; status path stayed under `local_validation/...` |

## Hub State

```
WeirdRunner/Ouroboros/
  diloco_state/
    anchor/                              ← Terminal Stage 10 DiLoCo anchor: stage 10 round 2, 3 workers, 25,994 samples this round
    round_state.json                     ← stage_k=10, mode=terminal, dgac_manual_gate=true, triggered_workers=[], attendance_workers=[]
    workflow_validation/25377312407-1/   ✓ live CPU-smoke status/report verified by coordinator #272
    workers/A/round_0000_stage_10/       ✓ 10,912 samples
    workers/{A,B,C}/round_0002_stage_10/ ✓ final terminal aggregation inputs: 8,665 + 8,665 + 8,664 samples
```

---

## Open Questions

| Question | Status |
|---|---|
| Stage 2 DiLoCo: aggregated model vs sequential baseline? | 🟡 Pre-val acc rising monotonically — promising |
| Stage 3 rounds 2–3: Worker A signals absent | 🟡 Solo/attendance handled correctly by coordinator |
| TRC GPU quota conversion | 🟡 Email sent — awaiting |
| DGAC halt_step distribution at K≥2 | 🟡 Ready to measure — launch Phase 3.4 DGAC via `kaggle_run_mode=dgac-train` |
| Worker quota for DiLoCo stage 10 | ✅ No longer blocking — final A/B/C round aggregated on 2026-05-09 |
| Stage 10 terminal anchor quality gate | ✅ Passed — `val_ce=0.4863`, `val_acc=0.0889`, coherent generation, `Mean UWR=0.733` |
| CPU-smoke end-to-end workflow gate before DGAC? | 🟢 Passed live — GitHub Actions `coordinate #272`, validation run `25377312407-1`, Worker A, Hub status/report verified |

---

## Resolved Decisions (canonical reference)

| Decision | Value |
|---|---|
| Model | Jamba Reasoning 3B (`ai21labs/AI21-Jamba-Reasoning-3B`) |
| Fine-tuning | QLoRA (4-bit NF4) + LoRA r=32 |
| LoRA targets | q/k/v/o_proj, in_proj, x_proj, dt_proj, out_proj — conv1d excluded |
| Curriculum K | 10 stages |
| `--max_seq_len` | 1024 |
| `--max_grad_norm` | 0.3 (k≥2 stages) |
| `--session_timeout_hours` | 12.0 |
| `--val_batch_size` | 2 |
| val accuracy samples | 50 |
| `--val_skip_buffer_minutes` | 60 |
| NCCL timeout | `timedelta(hours=4)` |
| DiLoCo `--epochs_per_stage` | 1 |
| DGAC `--epochs_per_stage` | 3 |
| `--batch_size` | 4 (2 per GPU on Dual T4) |
| amp_dtype T4 (sm75) | FP16 |
| amp_dtype A100+ (sm80+) | BF16 |
| Gradient checkpointing | Auto-disabled at VRAM≥40GB |
| Multi-account strategy | DiLoCo dynamic parallel |
| Notebook launch cells | `!torchrun` magic commands only |
| Worker auto-detection | `DILOCO_WORKER_ID` Kaggle secret per account (`A`/`B`/`C`) |
| Kaggle trigger mechanism | Local `kernels push` — staged checked-in notebook + generated metadata. No pull needed. ✅ |
| Kaggle push success detection | Strict parser: requires `successfully pushed`; quota/error output is failed dispatch even with exit code 0. ✅ |
| Kaggle SDK version | `kaggle>=1.8.4` — first with `--accelerator` for `kernels push` (PR #907). ✅ |
| `"accelerator"` JSON field | `"NvidiaTeslaT4"` (capital N). Belt-and-suspenders. ✅ |
| `"accelerator"` CLI flag | `--accelerator NvidiaTeslaT4` in `push_args`. ✅ |
| GPU fast-fail safety net | `cc < (7,5)` → `_diloco_reset_triggered_at()` + signal + `sys.exit(0)`. ✅ |
| W&B worker run ID | `diloco-{worker_lower}-s{stage_k}-r{round_n}` — unique per round |
| W&B worker group | `diloco-{worker_lower}-s{stage_k}` |
| W&B coordinator run ID | `diloco-coordinator-s{stage_k}` |
| DiLoCo outer LR | 0.7 (diloco) / 1.0 effective (solo = direct promotion) |
| `min_shard_samples` | 32 (1 optimizer step) |
| Solo mode | 1 active worker → direct weight promotion |
| Stage close | remaining < `min_shard_samples` per active worker |
| `workflow_dispatch` inputs | `force_worker_ids`, `skip_trigger`, `dry_run`, `workflow_validate`, `kaggle_run_mode` |
| Worker timeout threshold | 13h (Kaggle 12h wall + 1h grace) |
| `triggered_at=0` semantics | Canonical "dispatch unconfirmed" signal → immediate re-dispatch. ✅ |
| Attendance round | Worker in `attendance_workers` → skips training, uploads status(samples=0), pushes signal |
| Waiting mode | All credentialed workers in `attendance_workers`. `round_n` frozen. |
| Stages 4–7 structure | 1 round each (3 workers × 12 302 samples = 36 906 = full stage) |
| DGAC base weights | Loaded from `diloco_state/anchor/` via `--resume_from_diloco_anchor` ✅ |
