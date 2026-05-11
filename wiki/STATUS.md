# Project Status — Coconut-Ouroboros
> **This page is the body. `BLUEPRINT.md` is the index. Load this page after BLUEPRINT.md.**
> Last updated: 2026-05-10

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
| 10 — 10 latent passes | ✅ COMPLETE | Terminal DiLoCo and DGAC DiLoCo both reached 36,906/36,906 samples; latest anchor is the aggregated post-DGAC anchor |

**Compute mode:** DiLoCo dynamic workers with attendance/waiting-mode fallback.
**Current gate:** DGAC DiLoCo is complete, but the halt objective has been corrected after diagnostics showed early-halting risk. Run a short DGAC canary with correctness-aware HaltGate supervision, then rerun `dgac-anchor-eval`; coordinator cron must not dispatch stage 11.


---

## Engineering Architecture Status

| Track | Status | Notes |
|---|---|---|
| Training monolith extraction | ✅ COMPLETE | `jamba_coconut_finetune.py` is a thin compatibility adapter; `ouroboros.train` is now a compatibility façade over `ouroboros.training.checkpointing`, `evaluation`, `stage_runner`, and `session`. |
| Kaggle launch seam | ✅ COMPLETE | `ouroboros.kaggle_launch_matrix` owns launch-mode behavior; notebook remains a thin adapter and preserves literal `!torchrun` shell magic. |
| Runtime signal tracking cleanup | ✅ COMPLETE | Generated `signals/*.json` files are ignored; only `signals/.gitkeep` belongs in source control; the signal mechanism remains active as the coordinator doorbell. |
| Coordinator zero-drift extraction | ✅ COMPLETE | `diloco_coordinator.py` is a thin adapter; aggregation, state, dispatch, and orchestration live under `ouroboros.diloco.*`. |
| Planning artifact retirement | ✅ COMPLETE | Completed PRDs/plans have been promoted into wiki documentation and removed from `prds/` and `plans/`. |
| Kaggle CPU/API workflow validation | ✅ COMPLETE | CPU-smoke validation mode, repo/runtime seams, CPU metadata dispatch, fake coordinator loop tests, manual API validation docs, and remote Hub artifact verification are in place. |
| Deep-module runtime reliability | ✅ COMPLETE | Added pure seams for coordinator decisions, Kaggle launch contracts/matrix, training session plans, worker lifecycle classification, runtime env aliases, Kaggle runtime, and latent execution; latest chunked local CPU regression coverage passed `158` tests. |
| DGAC readiness CPU-smoke gate | ✅ PASSED LIVE | GitHub Actions `coordinate #272` ran `workflow_validate=cpu-smoke`, defaulted to Worker A, pushed Kaggle kernel version 39, Kaggle exited before `torchrun`, published Hub artifacts, and coordinator verified validation run `25377312407-1`. |
| JEPA-style latent prediction / multimodal Ouroboros | 🧊 DEFERRED | Direction documented in [Future-JEPA-Multimodal-Latent](Future-JEPA-Multimodal-Latent.md). No code changes until DGAC, evaluation, benchmarking, and core correctness gates are stable. |

Canonical architecture record: [Architecture-Extraction](Architecture-Extraction.md).
Runtime reliability record: [Deep-Module-Runtime-Reliability](Deep-Module-Runtime-Reliability.md).
Canonical execution protocol: [Engineering-Workflow](Engineering-Workflow.md).

---

## Immediate Next Steps

1. **Run a short DGAC canary with the corrected halt objective** — use the normal DGAC path, but inspect early metrics before spending a full epoch. Require `train/dgac_halt_loss`, `train/dgac_ponder`, `train/dgac_diversity`, and `train/dgac_halt_step_mean` to log separately.
2. **Rerun `dgac-anchor-eval` after the canary or corrected pass** — use GitHub Actions → `coordinate` with `kaggle_run_mode=dgac-anchor-eval`, `force_worker_ids=A`, `skip_trigger=false`, `dry_run=false`, and empty `workflow_validate`; trust it only if logs include `Loaded halt gate from diloco_state/anchor/halt_gate.pt`.
3. **Review DGAC research signal** — inspect HaltGate behavior / halt-step distribution and compare post-DGAC `val_ce`, `val_acc`, and generations against the terminal anchor baseline. `pct_at_1≈1.0` is acceptable only if forced-`k1` CE is near forced-full CE.
4. **Decide next branch** — if val/gen/halt distribution is good, move to benchmark + packaging PRD; if mixed, run another corrected DGAC DiLoCo pass from the aggregated anchor; if bad, rollback or inspect DGAC loss/instrumentation.
5. **Deferred research note** — JEPA-style latent prediction and multimodal input/output architecture are documented for later; do not implement before DGAC and baseline evaluations are stable.

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
| Deep-module CPU regression suite | ✅ PASS | Latest chunked local CPU validation across all 24 test files: `36 passed`, `84 passed`, `38 passed` → `158 passed` total |
| Training/runtime/launch/latent deep-module regression suite | ✅ PASS | Current post-RFC chunked coverage: all 24 test files, `158 passed` total. Single `pytest -q` can still time out locally before final summary, so chunked validation remains the reported local gate. |
| Kaggle launch + DGAC canary preflight slice | ✅ PASS | Targeted launch/notebook/deep-module/source-of-truth/DGAC halt-supervision gate passed locally: `57 passed`; `dgac-canary --dry_run` exits before Hub/Kaggle mutation. |

## Hub State

```
WeirdRunner/Ouroboros/
  diloco_state/
    anchor/                              ← Latest aggregated anchor; after DGAC this includes adapter weights + halt_gate.pt
    round_state.json                     ← stage_k=10, mode=dgac-complete, dgac_diloco_complete=true, triggered_workers=[], attendance_workers=[]
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
| DGAC halt_step distribution at K≥2 | 🟡 Ready to measure — rerun post-DGAC `dgac-anchor-eval` after HaltGate-load patch |
| Worker quota for DiLoCo stage 10 | ✅ No longer blocking — final A/B/C round aggregated on 2026-05-09 |
| Stage 10 terminal anchor quality gate | ✅ Passed — `val_ce=0.4863`, `val_acc=0.0889`, coherent generation, `Mean UWR=0.733` |
| CPU-smoke end-to-end workflow gate before DGAC? | 🟢 Passed live — GitHub Actions `coordinate #272`, validation run `25377312407-1`, Worker A, Hub status/report verified |
| JEPA-style latent prediction / multimodal output direction? | 🧊 Deferred — document only for now; future PRD should start with text-only Reasoning-JEPA v1 after DGAC evaluation |

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
| DGAC halt supervision | `--dgac_halt_supervision_weight 0.1`, `--dgac_halt_ce_tolerance 0.02`, `--dgac_halt_probe_steps 1,2,4,stage_k` |
| `--batch_size` | 4 (2 per GPU on Dual T4) |
| amp_dtype T4 (sm75) | FP16 |
| amp_dtype A100+ (sm80+) | BF16 |
| Gradient checkpointing | Auto-disabled at VRAM≥40GB |
| Multi-account strategy | DiLoCo dynamic parallel |
| Notebook launch cells | `!torchrun` magic commands only; shell tokens are tested against `ouroboros.kaggle_launch_matrix` |
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
| DGAC base weights | Loaded from `diloco_state/anchor/` via `--resume_from_diloco_anchor`; eval/training must also restore `halt_gate.pt` when present ✅ |
