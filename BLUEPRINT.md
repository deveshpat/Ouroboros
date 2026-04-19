# Project Ouroboros — Master Blueprint

> **Thread-resume header. Read Part 0 first in every new session.**
> **Source of truth:** If this doc and `.py`/`.ipynb` files ever disagree, the Python/notebook file wins.
> **DRY rule:** Session details and verbatim logs live in `terminal_log.md` only. This file holds decisions, status, and next actions.

---

## Part 0 — Quick-Resume Context

### What this project is
Coconut-Ouroboros: latent reasoning injection into Jamba Reasoning 3B (Transformer-Mamba hybrid). The Mamba SSM recurrent state acts as compressed scratch-pad across K latent thought passes, replacing token generation during reasoning. Based on Meta's Coconut (arXiv:2412.06769), extended with DGAC (Diversity-Gated Adaptive Coconut) — a novel anti-collapse halt gate.

### Current Status (2026-04-19)

| Curriculum Stage | Status | Best val |
|---|---|---|
| Stage 0 — CoT warmup | ✅ COMPLETE | ce=0.4041, acc=0.0222 |
| Stage 1 — 1 latent pass | ✅ COMPLETE | ce=0.4912, acc=0.0444 |
| Stage 2 — 2 latent passes | 🟡 59% (679/1154 steps) — anchor on Hub | — |
| Stages 3–10 | ⬜ NOT STARTED | — |
| Phase 3.4 — DGAC | ⬜ after Stage 10 | — |
| Phase 4 — GRPO | ⬜ after DGAC | — |

**Compute mode: switching from sequential relay → DiLoCo 3-way parallel (3× speedup)**

---

## Part 0.1 — Immediate Next Steps (strict order, no ambiguity)

### Step 1 — Feed agent: implement DiLoCo + DRY pass
Give the coding agent **this blueprint + `jamba_coconut_finetune.py` + `AGENT_PROMPT_diloco.md`**.
The agent implements DiLoCo additions AND the DRY refactors listed in Part 0.4.
No other files need changing yet.

### Step 2 — Run `bootstrap_diloco.py` (once, any machine with HF access)
Seeds `diloco_state/anchor/` on Hub from the existing `runs/stage3/checkpoint-0002987/`.
Also writes the initial `diloco_state/round_state.json`:
```json
{
  "stage_k": 2,
  "round_n": 0,
  "anchor_path": "diloco_state/anchor",
  "total_samples_seen": {"2": 21728},
  "completed_stages": [0, 1]
}
```
> `total_samples_seen["2"] = 679 steps × 32 effective batch = 21728` (approximate; safe to round down)

### Step 3 — GitHub repo setup (one-time)
- Add `.github/workflows/diloco_coordinator.yml` to the repo
- Add secrets: `HF_TOKEN`, `KAGGLE_USERNAME`, `KAGGLE_KEY`
- Create `signals/` directory with a placeholder `.gitkeep`

### Step 4 — Create three worker Kaggle notebooks
- `ouroboros-worker-a` (Account A), `ouroboros-worker-b` (Account B), `ouroboros-worker-c` (Account C)
- Identical except `--diloco_worker_id {A,B,C}`
- Enable "Run on API trigger" in each notebook's settings (allows GitHub Actions to trigger them via Kaggle API)

### Step 5 — Start all three workers simultaneously (Stage 2 test)
This IS the DiLoCo proof-of-concept. Each worker trains ~158 steps (~2.3h) on its shard of Stage 2's remaining data. If it works, continue. If it breaks, Account A can resume sequential from `checkpoint-0002987` — nothing is lost.

```bash
# Worker A (Account A) — all three run at the same time
torchrun --standalone --nproc_per_node=2 jamba_coconut_finetune.py \
  --data_dir data/coconut_v1 --use_4bit \
  --stage_0_epochs 1 --epochs_per_stage 1 --max_stage 10 \
  --batch_size 4 --grad_accum 8 --val_batch_size 2 \
  --val_skip_buffer_minutes 60 \
  --session_timeout_hours 12.0 --graceful_exit_buffer_minutes 20 \
  --diloco_mode --diloco_worker_id A \
  --diloco_outer_lr 0.7 \
  --diloco_state_repo WeirdRunner/Ouroboros \
  --diloco_signal_repo deveshpat/Ouroboros \
  --push_to_hub \
  --output_dir runs/diloco
```

### Step 6 — Watch coordinator
GitHub Actions fires when workers push signals. Coordinator aggregates, uploads new anchor, triggers next sessions. Monitor at `github.com/deveshpat/Ouroboros/actions`.

### Step 7 — Validate and continue
After Stage 2 completes via DiLoCo, coordinator runs val (or first worker of Stage 3 does via `--diloco_run_val`). If `val_acc` is within 10% of sequential Stage 2 benchmark (acc=0.0444), DiLoCo is confirmed. All remaining stages proceed with the same setup.

---

## Part 0.2 — Hub State: What's There, What Matters

### Current Hub layout (as of 2026-04-19)
```
WeirdRunner/Ouroboros/
  runs/stage3/
    best/                     ← misplaced (stage_1 artifact) — IGNORE
    checkpoint-0002308/       ← misplaced (stage_1 artifact) — IGNORE
    checkpoint-0002987/       ← Stage 2 anchor — USED BY bootstrap_diloco.py
    stage_0/best/             ← correct ✓
    stage_1/                  ← correct ✓
```

**No cleanup needed.** DiLoCo uses `diloco_state/` prefix entirely. The misplaced files are a legacy artifact from Session 15 (fix was applied to `save_checkpoint()` after that session ran). For sequential fallback: pass `--resume_from runs/stage3/checkpoint-0002987` explicitly.

### DiLoCo Hub layout (written by workers + coordinator)
```
WeirdRunner/Ouroboros/
  diloco_state/
    round_state.json
    anchor/
      adapter_model.safetensors
      adapter_config.json
    workers/{A,B,C}/
      status.json
      round_{N:04d}_stage_{k}/
        adapter_model.safetensors
        adapter_config.json
```

---

## Part 0.3 — Resolved Decisions

| Decision | Value |
|---|---|
| Model | Jamba Reasoning 3B (`ai21labs/AI21-Jamba-Reasoning-3B`) |
| Fine-tuning | QLoRA (4-bit NF4) + LoRA r=32 |
| LoRA targets | q/k/v/o_proj, in_proj, x_proj, dt_proj, out_proj — conv1d excluded |
| Curriculum K | 10 stages |
| `--max_seq_len` | 1024 |
| `--max_grad_norm` | 0.3 (k≥2 stages) |
| `--session_timeout_hours` | **12.0** (headless wall-clock; not quota-limited) |
| `--val_batch_size` | 2 |
| val accuracy samples | 50 |
| `--val_skip_buffer_minutes` | 60 |
| NCCL timeout | `timedelta(hours=4)` |
| `--epochs_per_stage` | 1 |
| `--batch_size` | 4 (2 per GPU on Dual T4) |
| amp_dtype T4 (sm75) | FP16 |
| amp_dtype A100+ (sm80+) | BF16 |
| Gradient checkpointing | Auto-disabled at VRAM≥40GB |
| Multi-account strategy | **DiLoCo 3-way parallel** (see `AGENT_PROMPT_diloco.md`) |
| Stage advancement (DiLoCo) | When `sum(all workers' samples_seen_this_stage) >= len(train_set)` |
| Val in DiLoCo mode | Coordinator only, once per stage (when `round_n == 0`) |
| DiLoCo outer LR | 0.7 (DiLoCo paper default) |
| DiLoCo min_workers | 2 of 3 |
| DiLoCo inner steps | Full shard per session (~384 steps at stage 3); capped at session limit for higher stages |
| TRC quota | TPU only — incompatible. Email requesting GPU conversion pending. |

---

## Part 0.4 — DRY Refactors for Coding Agent

The agent should apply these **alongside** the DiLoCo additions. No behavioral changes — pure deduplication.

### R1 — Merge the two token resolution functions
`_bootstrap_resolve_token()` and `_resolve_hf_token()` are identical in logic (Kaggle secret → Colab userdata → env var). Keep one pre-import-safe (stdlib only) implementation. Call it from both bootstrap and post-import contexts. Remove the duplicate.

### R2 — Extract shared latent pass loop
`evaluate_stage()`, `run_generation_callback()`, and `_forward_batched_latent()` all implement:
```
for step in range(n_latent):
    run backbone on prefix → get h_step
    optionally check halt gate → maybe break
    append h_step to context
```
Extract to `_run_latent_passes(model, ctx, ctx_mask, n_latent, halt_gate, args, device, amp_dtype) -> (ctx, ctx_mask, actual_k)`. Replace all three call sites.

### R3 — Cache backbone/embed/lm_head lookups
`_get_backbone()`, `_get_embed_tokens()`, `_get_lm_head()` traverse the model graph on every call and are invoked in the hot training loop. Decorate with `@functools.lru_cache(maxsize=None)` or cache results on the model object after first lookup. This is a free perf win.

### R4 — Collapse `_forward_batched_stage0` into `_forward_batched_latent`
`_forward_batched_stage0` is `_forward_batched_latent` with `n_latent=0`. The latent loop already exits immediately when `max_n_latent=0`. Remove `_forward_batched_stage0` and remove the routing branch in `coconut_forward`. Single code path, same behavior.

### R5 — `_ddp_sum()` helper
The all-reduce pattern appears 3× in `evaluate_stage()`:
```python
t = torch.tensor([a, b], device=device, dtype=torch.float64)
torch.distributed.all_reduce(t, op=torch.distributed.ReduceOp.SUM)
```
Extract to `_ddp_sum(values: list, device) -> list` that no-ops when not distributed.

---

## Part 0.5 — Pre-flight Blockers

All resolved. See `terminal_log.md` for session details.

| Blocker | Resolution |
|---|---|
| `attn_implementation` crash | try/except fallback ✅ |
| `use_mamba_kernels` old TF | `_safe_from_pretrained` retry ✅ |
| `last_hidden_state` None | assert in all forward paths ✅ |
| Graceful session timeout | `make_timeout_checker()` ✅ |
| `conv1d` in LoRA | Excluded ✅ |
| OOM at val | `empty_cache()` + `val_batch_size=2` ✅ |
| Stage 1+ samples filtered by short seq_len | `--max_seq_len 1024` ✅ |
| Exploding gradients k≥2 | `--max_grad_norm 0.3` ✅ |
| mamba-ssm 2.x API break | Pinned to 1.2.2 via git URL ✅ |
| Val at 200 samples too slow | Capped at 50 ✅ |
| NCCL watchdog kills DDP val | `timedelta(hours=4)` + env var ✅ |
| BF16 emulation on T4 | `_amp_dtype` checks `cc >= (8,0)` ✅ |
| GC wastes compute on A100 | Auto-disable at VRAM≥40GB ✅ |
| `_amp_dtype` called in hot loop | `@lru_cache` ✅ |
| Hub upload missing `stage_{k}/` subdir | Fixed in `save_checkpoint()` ✅ |
| Prior-stage `best/` accumulating | `startup_hub_sync_and_prune` prunes them ✅ |
| Sequential relay bottleneck | → DiLoCo 3-way parallel ✅ |
| Session timeout = quota display | Headless = 12h wall-clock; `--session_timeout_hours 12.0` ✅ |

---

## Part 1 — Architecture

### Jamba Reasoning 3B
```
HuggingFace : ai21labs/AI21-Jamba-Reasoning-3B   License: Apache 2.0
Layers      : 28 (26 Mamba + 2 Attention) — 13:1 ratio
Attention   : MQA (20 Q heads, 1 KV head)
Vocab / Context : 64K / 256K tokens
d_model     : 2560
Trainable   : 26,851,328 params (0.88% — LoRA adapters only)
```

### Coconut Curriculum
```
Stage 0:  [Q][S1..Sn][A]              standard CoT; labels on all steps + A
Stage k:  [Q][●*k][S_{k+1}..Sn][A]   first k steps → latent; labels shift right
Stage K:  [Q][●*K][A]                 all steps replaced; labels on A only
K = 10
```

### DGAC (Phase 3.4 only)
```
L_total = L_ce + λ₁(t)·L_ponder + λ₂·L_diversity
L_diversity = mean( Σ_k relu(cos_sim(h_k, h_{k-1}) − τ) ),  τ=0.9
λ₁: 0 for steps 0-200, ramp 0→0.01 over steps 200-500
HaltGate: Linear(2·d_model → 1), zero-init → outputs 0.5 at start
```

---

## Part 2 — Performance Model

```
t_fp16(k) ≈ 34 + 11.5·k  seconds/step  (empirical, Dual T4)
Stage 2: ~52-57s  Stage 3: ~69s  Stage 5: ~92s  Stage 10: ~149s
```

| Mode | Stages 3–10 |
|---|---|
| Sequential relay | ~278h (~3.1 weeks) |
| DiLoCo 3-way parallel | ~93h (~1 week) |
| DiLoCo + A100 (if TRC) | ~19h (~2 days) |

**Per-worker shard:** ~12,302 samples → ~384 steps/worker/stage. Stages 1-9 fit in one 12h session. Stage 10 (~149s/step × 384 = ~15.9h) needs 2 rounds; coordinator handles transparently.

---

## Part 3 — File Registry

| File | Status |
|---|---|
| `jamba_coconut_finetune.py` | 🟡 Needs agent pass: DiLoCo additions + DRY refactors |
| `diloco_coordinator.py` | ⬜ New — spec in `AGENT_PROMPT_diloco.md` |
| `bootstrap_diloco.py` | ⬜ New — spec in `AGENT_PROMPT_diloco.md` |
| `.github/workflows/diloco_coordinator.yml` | ⬜ New — spec in `AGENT_PROMPT_diloco.md` |
| `kaggle-utils.ipynb` | 🟡 Cell 5 needs worker variant per account |
| `prepare_coconut_dataset.py` | ✅ Done |
| `build_wheels_kaggle.py` | ✅ Done |

---

## Part 4 — Open Questions

| Question | Status |
|---|---|
| Stage 2 gn spikes (~1.9 pre-clip): transient? | 🟡 Monitor — CE not diverging |
| Kaggle API `/kernels/{slug}/run` — verify endpoint works | 🟡 Test before first DiLoCo run |
| TRC GPU quota conversion | 🟡 Draft email ready in `terminal_log.md` |
| DGAC halt_step distribution at K≥2 | 🔴 Open — primary research question |
| Prefix re-computation optimization | 🟡 Pre-A100, before Stage 5 if TRC succeeds |

---

## Part 5 — Hard Lessons

| Lesson | Fix |
|---|---|
| val_batch_size=16 → OOM | `--val_batch_size 2` |
| NCCL watchdog at 60min | `timedelta(hours=4)` + env var |
| `max_seq_len=512` filtered stage 1+ | `--max_seq_len 1024` |
| gn=36.9 at k=2 | `--max_grad_norm 0.3` |
| mamba-ssm 2.x broke fast path | Pinned 1.2.2 via git URL |
| mamba-ssm PyPI sdist is a stub | Must use `git+https://...@v1.2.2` |
| Per-sample loop → 113s/step | Batched forward |
| Val 200 samples → 5.5h | Cap at 50 |
| `is_bf16_supported()` true on T4 (emulation) | `cc >= (8,0)` check |
| GC wastes 20-40% on A100 | Auto-disable at VRAM≥40GB |
| `--push_to_hub` omitted → silent no-op | Always add explicitly |
| Checkpoint pruning per-stage only | `startup_hub_sync_and_prune` at session start |
| Step-time model `34 + 6k` too optimistic | Empirical: `34 + 11.5k` |
| `save_checkpoint` missing `stage_{k}/` in remote path | Fixed |
| "Parallel sessions impossible on Kaggle" | DiLoCo makes it viable |
| TRC assumed GPU access | TPU only — incompatible |
| P100 assumed better than T4 | No Tensor Cores, single GPU — worse |
| Headless sessions capped by quota display | Wall-clock = 12h; `--session_timeout_hours 12.0` |
