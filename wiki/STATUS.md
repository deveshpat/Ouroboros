# Project Status — Coconut-Ouroboros
> **This page is the body. `BLUEPRINT.md` is the index. Load this page after BLUEPRINT.md.**
> Last updated: 2026-05-05

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
| 10 — 10 latent passes | 🔄 IN PROGRESS | Round 0: Worker A contributed 10,912 samples; round 1 is waiting/attendance after A/B/C Kaggle GPU quota push failures |

**Compute mode:** DiLoCo dynamic workers with attendance/waiting-mode fallback.
**Current blocker:** Kaggle weekly GPU quota exhausted for A/B/C; coordinator now reconciles failed quota dispatches immediately instead of waiting 13h.


---

## Engineering Architecture Status

| Track | Status | Notes |
|---|---|---|
| Training monolith extraction | ✅ COMPLETE | `jamba_coconut_finetune.py` is a thin compatibility adapter delegating into `ouroboros.train` and related package modules. |
| Kaggle launch seam | ✅ COMPLETE | Notebook remains a thin adapter and preserves `!torchrun` shell magic. |
| Runtime signal tracking cleanup | ✅ COMPLETE | Generated `signals/*.json` files are ignored; only `signals/.gitkeep` belongs in source control; the signal mechanism remains active as the coordinator doorbell. |
| Coordinator zero-drift extraction | ✅ COMPLETE | `diloco_coordinator.py` is a thin adapter; aggregation, state, dispatch, and orchestration live under `ouroboros.diloco.*`. |
| Planning artifact retirement | ✅ COMPLETE | Completed PRDs/plans have been promoted into wiki documentation and removed from `plans/`. |
| Kaggle CPU/API workflow validation | ✅ COMPLETE | CPU-smoke validation mode, repo/runtime seams, CPU metadata dispatch, fake coordinator loop tests, manual API validation docs, and remote Hub artifact verification are in place. |
| DGAC readiness CPU-smoke gate | ✅ COMPLETE | `workflow_dispatch` exposes `workflow_validate=cpu-smoke`; coordinator validates GitHub Actions → Kaggle → Hub without touching live DiLoCo round state. |

Canonical architecture record: [Architecture-Extraction](Architecture-Extraction.md).
Canonical execution protocol: [Engineering-Workflow](Engineering-Workflow.md).

---

## Immediate Next Steps

1. **Run the new workflow-dispatch CPU smoke once** — choose `workflow_validate=cpu-smoke` in GitHub Actions. Leave `force_worker_ids` empty to default to Worker A. Pass only after the coordinator verifies the remote Hub validation artifact.
2. **Wait for Kaggle GPU quota or use alternate quota** — stage 10 round 1 is waiting/attendance because A/B/C push attempts hit weekly GPU quota.
3. **Define Stage 10 → DGAC model-quality gates** — pre-val accuracy threshold, CE/accuracy trend, DGAC halt-step distribution checks, W&B metrics, stop/rollback criteria.
4. **DGAC prep** — `--resume_from_diloco_anchor` flag ready. Launch command in `BLUEPRINT.md`; launch only after CPU smoke and Stage 10 quality gates pass.

---

## Hub State

```
WeirdRunner/Ouroboros/
  diloco_state/
    anchor/                              ← Stage 10 path active; stage 10 round 0 contributed by Worker A
    round_state.json                     ← stage_k=10, round_n=1, mode=waiting, triggered_workers=[], attendance_workers=[A,B,C]
    workflow_validation/<run_id>/        ← CPU-smoke remote status/report namespace
    workers/A/round_0000_stage_10/       ✓ 10,912 samples
    workers/{A,B,C}/round_0001_stage_9/  ✓ historical stage 9 worker outputs visible in W&B
```

---

## Open Questions

| Question | Status |
|---|---|
| Stage 2 DiLoCo: aggregated model vs sequential baseline? | 🟡 Pre-val acc rising monotonically — promising |
| Stage 3 rounds 2–3: Worker A signals absent | 🟡 Solo/attendance handled correctly by coordinator |
| TRC GPU quota conversion | 🟡 Email sent — awaiting |
| DGAC halt_step distribution at K≥2 | 🔴 Open — primary research question (Phase 3.4) |
| Worker C quota stable for remainder? | 🟢 Active since stage 3 r5 — appears stable |
| Pre-val accuracy at stage 10: success threshold for DGAC? | 🔴 Open — define before Phase 3.4 |
| CPU-smoke end-to-end workflow gate before DGAC? | 🟢 Implemented — run `workflow_validate=cpu-smoke` from GitHub Actions |

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
| `--epochs_per_stage` | 1 |
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
| `workflow_dispatch` inputs | `force_worker_ids`, `skip_trigger`, `dry_run`, `workflow_validate` |
| Worker timeout threshold | 13h (Kaggle 12h wall + 1h grace) |
| `triggered_at=0` semantics | Canonical "dispatch unconfirmed" signal → immediate re-dispatch. ✅ |
| Attendance round | Worker in `attendance_workers` → skips training, uploads status(samples=0), pushes signal |
| Waiting mode | All credentialed workers in `attendance_workers`. `round_n` frozen. |
| Stages 4–7 structure | 1 round each (3 workers × 12 302 samples = 36 906 = full stage) |
| DGAC base weights | Loaded from `diloco_state/anchor/` via `--resume_from_diloco_anchor` ✅ |
