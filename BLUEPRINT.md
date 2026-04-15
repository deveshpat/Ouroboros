# Project Ouroboros — Master Blueprint

> **Thread-resume header. Read Part 0 first in any new session.**
> **DRY Guidelines:** Session details live in `terminal_log.md`. Blueprint holds decisions, architecture, status, and next actions only.

---

## Part 0 — Quick-Resume Context

### What this project is
Coconut-Ouroboros latent reasoning injection into a Transformer-Mamba hybrid (Jamba Reasoning 3B). The Mamba SSM recurrent state acts as compressed scratch-memory across K latent thought passes, bypassing token generation during reasoning. Core mechanism from Meta's Coconut (arXiv:2412.06769), extended with DGAC (Diversity-Gated Adaptive Coconut) — our novel anti-collapse halt gate.

### Strategic Status (Updated 2026-04-15)

| Stage | Name | Status |
|---|---|---|
| 0 | Architecture & Viability (nano) | ✅ COMPLETE |
| 1 | Pre-training (nano) | ✅ Pipeline test only; retired |
| 2 | SFT (nano) | 🔴 RETIRED |
| 3 | Coconut-Ouroboros + DGAC on Jamba Reasoning 3B | 🟡 ACTIVE — Stage 0 epoch 0 at step 629/2307 (batch_size=2). Resuming with batch_size=4 → new steps_per_epoch=1154. Resume from step 629 is valid (629 < 1154). |
| 4 | GRPO on Jamba Reasoning 3B | ⬜ NOT STARTED |
| 5 | Quantization / Edge Deploy | ⬜ NOT STARTED |

### TRC Status
✅ Accepted (email 2026-04-07). **Do not claim quota yet.** Claim after K=4 gate passes on Kaggle Dual T4.

### Dataset Confirmed (2026-04-13)
- **Train:** 36,906 samples  **Val:** 1,940 samples
- **median_steps=10  mean=10.42  max=16**
- **`--max_stage=10` for all production runs**

### GPU Arch / Hub Wheel Status
| Arch | causal_conv1d | mamba_ssm |
|---|---|---|
| sm75 (T4) | ✅ on Hub | ✅ on Hub |
| sm100 (B100) | ✅ on Hub | ❌ not yet built |

---

### Part 0.1 — Immediate Next Actions (ordered)

1. **[DONE] Phase 2.5/2.6 ordering swap** — Confirmed fixed. Generation shim runs before kernel export patch.

2. **[DONE] Batched forward for Stage 0** — Confirmed working. ~30s/step on Dual T4 profile run.

3. **[DONE] Profile Dual T4 throughput** — Profile run completed: `val_acc=0.4000` after 12 steps.

4. **[DONE] Batched latent injection for stages k>0** — `_forward_batched_latent()` + `_build_padded_prefix_batch()` fully implemented. Single backbone call per latent step across all active samples in micro-batch. `q_lens` padding is correctly applied. Blueprint was incorrectly marked PENDING — audited and confirmed complete.

5. **[IN PROGRESS → NEXT SESSION] Resume Stage 0 with A+B options combined:**
   ```bash
   torchrun --standalone --nproc_per_node=2 jamba_coconut_finetune.py \
     --data_dir data/coconut_v1 --use_4bit \
     --stage_0_epochs 1 --epochs_per_stage 1 --max_stage 10 \
     --batch_size 4 --grad_accum 8 \
     --session_timeout_hours 11.0 --graceful_exit_buffer_minutes 20 \
     --output_dir runs/stage3_curriculum
   ```
   - Auto-resumes from `runs/stage3_curriculum/stage_0/checkpoint-0000629`.
   - With `--batch_size 4`, `steps_per_epoch` = ceil(36906/32) = **1154**. Resume at step 629/1154 is valid.
   - Estimated remaining Stage 0 work: ~525 steps × ~90s/step ≈ **~13h** (vs ~28h continuing at batch_size=2).
   - After Stage 0 completes (1 epoch only), automatically advances to Stage 1–10 with batch_size=4.

6. **[DECIDED — SEE PART 2] Stage 0 is necessary but 1 epoch suffices.** Do NOT skip Stage 0. See resolved decision rationale in Part 2.

7. **After K=4 gate passes**: claim TRC, run K=10 + DGAC Phase 3.4 on A100.

8. **If a sm100 session is allocated**: run `build_wheels_kaggle.py` to cache mamba_ssm sm100 wheel.

---

### Part 0.2 — Pre-flight Blockers

| Blocker | Resolution |
|---|---|
| `attn_implementation` hardcoded crash | try/except fallback to `eager` ✅ |
| `use_mamba_kernels` kwarg on old TF | `_safe_from_pretrained` retries without kwarg ✅ |
| `use_mamba_kernels=False` hardcoded | Replaced with runtime probe ✅ |
| `last_hidden_state` silent None | assert in Stage 0 and Phase B ✅ |
| No graceful timeout → lost Kaggle work | `make_timeout_checker()` integrated ✅ |
| `conv1d` in LoRA targets | Explicitly excluded ✅ |
| NCCL watchdog (S5–S7) | `timeout=timedelta(minutes=60)` + `TORCH_NCCL_ASYNC_ERROR_HANDLING=1` ✅ |
| OOM at first val step (S8) | `torch.cuda.empty_cache()` + `val_batch_size=1` ✅ |
| `--max_seq_len 512` filtered all stage 1+ samples | Default changed to 1024 ✅ |
| Exploding gradients at k≥2 | `--max_grad_norm 0.3` ✅ |
| Wheel/ABI mismatch | Retired wheel workflow; `_bootstrap()` self-installs ✅ |
| mamba-ssm 2.x API | Pinned to `mamba-ssm==1.2.2` ✅ |
| bitsandbytes version floor missing | `bitsandbytes>=0.46.1` in `_bootstrap()` ✅ |
| causal_conv1d Hub wheel (sm100, sm75) | Built + uploaded for both arches ✅ |
| mamba-ssm 1.2.2 PyPI sdist is a 35kB stub | pip spec changed to `git+https://github.com/state-spaces/mamba.git@v1.2.2` ✅ |
| `GreedySearchDecoderOnlyOutput` + 9 other generation classes removed in `transformers>=4.44` | Comprehensive 10-alias shim in Phase 2.6 ✅ |
| Verifier called `causal_conv1d_fn` with wrong weight shape `(dim, 1, width)` | Fixed to `(dim, width)` ✅ |
| Verifier used wrong import path for `selective_state_update` | Fixed to `mamba_ssm.ops.triton.selective_state_update` ✅ |
| **Phase 2.5 ran before Phase 2.6 → noisy WARNING every session** | **✅ FIXED — phases swapped; confirmed in Session 10 screenshot** |
| **`coconut_forward` iterates per-sample at batch=1 even when `n_latent=0`** | **✅ FIXED — batched stage-0 forward path implemented; ~30s/step on Dual T4 profile run** |
| **Stages k≥1 have no batching → will be slower than stage 0** | **✅ FIXED — `_forward_batched_latent()` + `_build_padded_prefix_batch()` fully implement batched latent injection with q_len padding. Blueprint was incorrectly marked PENDING. Audited 2026-04-15.** |
| **Stage 0 alone will take ~115h at default 3 epochs_per_stage** | **✅ RESOLVED — `--stage_0_epochs 1 --batch_size 4` reduces to ~13h remaining from current checkpoint. See Part 2.** |

**One item still requires empirical verification during training run:**
- `inputs_embeds` → `last_hidden_state` path for Jamba Reasoning 3B **with fast Mamba kernels active** at stage k≥1

---

## Part 1 — Architecture

### Jamba Reasoning 3B (primary research model)
```
HuggingFace : ai21labs/AI21-Jamba-Reasoning-3B   License: Apache 2.0
Layers      : 28 total (26 Mamba, 2 Attention) → 13:1 Mamba:Attention ratio
Attention   : MQA (20 Q heads, 1 KV head)
Vocab       : 64K
Context     : 256K tokens
```

---

## Part 2 — Resolved Decisions

| Decision | Resolution |
|---|---|
| Primary research model | Jamba Reasoning 3B |
| Fine-tuning approach | QLoRA (4-bit NF4) + LoRA via `--use_4bit` |
| LoRA target modules | q_proj, k_proj, v_proj, o_proj, in_proj, x_proj, dt_proj, out_proj; conv1d excluded |
| Coconut curriculum | Progressive step replacement per Meta paper |
| Stage advancement | Epoch-based + best-accuracy checkpoint selection |
| max_stage K | **10** (confirmed from dataset median_steps) |
| DGAC halt gate | Phase 3.4 only; λ₁ annealed from 0 |
| Dataset Hub config | `coconut-v1` under `WeirdRunner/Ouroboros` |
| `attn_implementation` | Runtime detection: flash_attention_2 if available, else eager |
| `use_mamba_kernels` | Runtime probe; only False on ImportError |
| mamba-ssm version | **1.2.2 from GitHub source** |
| mamba install strategy | `_bootstrap()` downloads pre-built arch wheels from Hub; falls back to git+https |
| `--max_seq_len` | 1024 |
| `--max_grad_norm` | 0.3 for k≥2 stages |
| Session timeout | `--session_timeout_hours 11.0 --graceful_exit_buffer_minutes 20` |
| val_batch_size | 1 |
| Stage 0 forward pass | ✅ Batched — single fused backbone call per micro-batch when `n_latent=0` |
| Stage k≥1 forward pass | ✅ **DONE** — `_forward_batched_latent()` processes all active samples in one backbone call per latent step; `_build_padded_prefix_batch()` pads per `q_lens`. Audited 2026-04-15. |
| epochs_per_stage | **1 for all stages** (`--stage_0_epochs 1 --epochs_per_stage 1`). Rationale below. |
| batch_size | **4 for all stages** (Options A+B combined). Rationale below. |
| **Should Stage 0 be skipped for Jamba Reasoning 3B?** | **NO — run 1 epoch. Full rationale below.** |

### Stage 0 Skip Decision — Full Rationale (2026-04-15)

**The proposal:** Jamba Reasoning 3B already has strong CoT reasoning. Val_acc=0.40 after just 12 steps. Why spend ~38h on Stage 0?

**Why Stage 0 cannot be skipped:**

1. **The `<|lat|>` embedding and LoRA adapters have seen zero coconut-v1 samples.** Val_acc=0.40 comes from the frozen base model weights, not the LoRA adapters. The LoRA adapters must be domain-adapted to coconut-v1 before latent injection training begins. Stage k=1 assumes the model already knows what `S2…Sn+A` looks like on this specific dataset.

2. **Cold-start problem at Stage 1 without Stage 0.** At k=1, the supervised signal is on `S2…Sn+A`. If the LoRA adapters haven't learned this distribution, the CE loss is noisy and the gradient signal for learning what `●` should encode is confounded. Stage 0 decouples these two learning problems: Stage 0 solves "what does `S2…Sn+A` look like," Stage 1 then only needs to solve "what should `●` encode."

3. **Coconut paper explicitly validates this.** Table 2 in arXiv:2412.06769 shows that even strong base models suffer significant accuracy degradation when Stage 0 is removed. The curriculum anchor is the reference distribution established by Stage 0, not just the CoT format.

4. **Gradient instability risk.** You already saw gn=36.9 at k=2 even WITH proper Stage 0. Skipping Stage 0 increases the risk of exploding gradients at Stage 1+, which would require even more aggressive gradient clipping and likely produce worse final accuracy.

**Why 1 epoch is sufficient (not 3):**

- CE trending 0.517→0.343 over 629 steps (27% of epoch 0) shows rapid convergence. The LoRA adapters learn this distribution fast.
- Val_acc=0.40 after only 12 steps (smoke test) confirms the base model's prior knowledge makes domain adaptation extremely efficient.
- Jamba Reasoning 3B's pretraining means Stage 0 is format adaptation only, not capability acquisition. 1 epoch is enough to establish the coconut-v1 anchor.

**Resume compatibility with batch_size change:**

Checkpoint at step 629 was created with `batch_size=2` (steps_per_epoch=2307). Resuming with `batch_size=4` gives steps_per_epoch=1154. The resume logic uses `step_in_epoch=629`; since 629 < 1154, the epoch continues from step 629/1154 with the new data indexing. The optimizer state is parameter-level (AdamW moments) and remains valid regardless of batch_size. Data ordering shifts slightly but is inconsequential at this stage. ~525 remaining steps at ~90s/step ≈ 13h.

---

## Part 3 — Open Questions

| Question | Status |
|---|---|
| Does `inputs_embeds` → `last_hidden_state` work with Jamba + fast Mamba kernels active? | 🟡 VERIFY at stage k≥1 |
| DGAC Phase 3.4: does halt_step distribute across K≥2 after training? | 🔴 OPEN — primary research validation |
| Will `--batch_size 4` OOM on Dual T4 with QLoRA 4-bit? | 🟡 OPEN — test empirically in next session; if OOM, fall back to batch_size=2 + grad_accum=16 |
| Is 1 epoch per stage sufficient for convergence? | 🟡 MONITOR — watch val_acc after Stage 0 epoch 0 completes |
| What is actual step time at batch_size=4 on Dual T4? | 🟡 OPEN — estimate ~90s/step; profile empirically |

---

## Part 4 — File Registry

| File | Stage | Status | Notes |
|---|---|---|---|
| `baseline_trm_mamba.py` | 0 | ✅ COMPLETE | Retired; architecture reference only |
| `viability_gate.py` | 0 | ✅ COMPLETE | |
| `training_utils.py` | All nano | ✅ COMPLETE | Not used in Jamba scripts |
| `pretrain.py` | 1 | ✅ COMPLETE | Hub: ckpt-0021000 |
| `prepare_sft_dataset.py` | 2 | ✅ DONE | sft-mix-v1 cached; not reused |
| `train_sft.py` | 2 | ✅ PATCHED | Retired |
| `prepare_coconut_dataset.py` | 3 | ✅ DONE | coconut-v1 on Hub confirmed |
| `jamba_coconut_finetune.py` | 3 | 🟡 ACTIVE | All blockers resolved. Batched stage-0 ✅. Batched stage k≥1 ✅ (audited 2026-04-15). |
| `build_wheels_kaggle.py` | 3 | ✅ DONE | sm75 + sm100 causal_conv1d on Hub; sm75 mamba_ssm on Hub |
| `kaggle-utils.ipynb` | 3 | ✅ UP TO DATE | Last cell updated to batch_size=4, epochs_per_stage=1. Add `--stage_0_epochs 1`. |

---

## Part 5 — Coconut Curriculum Design

```
Stage 0:  [Q][S1][S2][S3][A]     ← standard CoT; labels on S1..Sn + A
Stage k:  [Q][●*k][S_{k+1}..Sn][A]   ← first k steps replaced; labels shift
Stage K:  [Q][●*K][A]            ← all steps replaced; labels on A only
K = 10 (confirmed from dataset)
```

---

## Part 6 — DGAC: Diversity-Gated Adaptive Coconut

```
L_total = L_ce  +  λ₁(t) · L_ponder  +  λ₂ · L_diversity

L_diversity = mean_batch( Σ_k relu(cos_sim(h_k, h_{k-1}) − τ) ),  τ = 0.9
λ₁ schedule: 0 for steps 0-200, ramp 0→0.01 over steps 200-500, flat 0.01 after
```

**HaltGate:** Linear(2*d_model → 1), zero-initialized → outputs 0.5 at Phase 3.4 start.

---

## Part 7 — Checkpoint Format

```
output_dir/
  stage_0/best/
    adapter_model/         ← PEFT LoRA weights
    training_state.pt      ← {stage_k, step, epoch, val_ce, val_acc, optimizer, scheduler}
  ...
  stage_10/best/           ← resume_from for Phase 3.4 DGAC run
    halt_gate.pt           ← HaltGate state dict (Phase 3.4 only)
```

Current live checkpoint: `runs/stage3_curriculum/stage_0/checkpoint-0000629`

---

## Part 8 — Compute Plan

| Phase | Platform | Estimate |
|---|---|---|
| Stage 0 (1 epoch, batch_size=4) | Kaggle Dual T4 | ~13h remaining from checkpoint-0000629. 1–2 sessions. |
| Stages 1–10 (batch_size=4, batched latent injection) | Kaggle Dual T4 | TBD — re-estimate empirically after Stage 0 completes. Profile step time at k=1 immediately. |
| Phase 3.4 (DGAC) | TRC A100 80GB | ~6–8h |
| Phase 4 (GRPO) | TRC A100 80GB | ~8–12h |

---

## Part 9 — Hard Lessons (Do Not Repeat)

| Lesson | Codified As |
|---|---|
| val_batch_size=16 → OOM | `--val_batch_size 1` default + `empty_cache()` |
| NCCL watchdog kills DDP | `timedelta(minutes=60)` + graceful exit |
| max_seq_len=512 filtered all stage 1+ | `--max_seq_len 1024` |
| gn=36.9 at k=2 | `--max_grad_norm 0.3` |
| `use_mamba_kernels=False` hardcoded → 100× slow | Runtime probe |
| mamba-ssm 2.x broke fast path | Pinned to 1.2.2 |
| Kaggle GPU arch unpredictable | `TORCH_CUDA_ARCH_LIST` auto-injected |
| bitsandbytes not upgraded | `bitsandbytes>=0.46.1` in bootstrap |
| mamba-ssm 1.2.2 PyPI sdist is a 35kB stub | Use `git+https://github.com/state-spaces/mamba.git@v1.2.2`. ~20h GPU quota lost. |
| Single-name generation shim → one removed class fixed per session | Comprehensive 10-alias shim. Never patch one name at a time. |
| Verifier weight shape wrong → false negative on valid wheels | Always verify kernel call signatures before writing test inputs |
| Per-sample loop at batch=1 for stage 0 → 113s/step, full run infeasible | Batched forward path for `n_latent=0` ✅ fixed; ~30s/step on Dual T4 |
| "~5s/step" estimate was for nano model | Re-estimate wall-clock empirically on target hardware before planning budgets |
| Profile run (200 samples) shows 30s/step; full run (36906) shows 60s/step | Dataset I/O overhead is real. Always profile at full scale before planning session counts. |
| Blueprint marked stage k≥1 batching as PENDING but code already had it | Audit the .py file before marking blockers. Blueprint was wrong, code was right. |
| Assumed Stage 0 could be skipped for a capable reasoning model | Stage 0 is domain adaptation + LoRA anchor, not CoT capability training. Cannot skip. 1 epoch is sufficient. |
