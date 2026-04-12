# Project Ouroboros — Master Blueprint

> **Thread-resume header. Read Part 0 first in any new session.**
> **DRY Guidelines:** Session details live in `terminal_log.md`. Blueprint holds decisions, architecture, status, and next actions only.

---

## Part 0 — Quick-Resume Context

### What this project is
Coconut-Ouroboros latent reasoning injection into a Transformer-Mamba hybrid (Jamba Reasoning 3B). The Mamba SSM recurrent state acts as compressed scratch-memory across K latent thought passes, bypassing token generation during reasoning. Core mechanism from Meta's Coconut (arXiv:2412.06769), extended with DGAC (Diversity-Gated Adaptive Coconut) — our novel anti-collapse halt gate.

### Strategic Status (Updated 2026-04-12)

| Stage | Name | Status |
|---|---|---|
| 0 | Architecture & Viability (nano) | ✅ COMPLETE |
| 1 | Pre-training (nano) | ✅ Pipeline test only; retired |
| 2 | SFT (nano) | 🔴 RETIRED |
| 3 | Coconut-Ouroboros + DGAC on Jamba Reasoning 3B | 🟡 NEXT — smoke test done; script self-sufficient; ready to run |
| 4 | GRPO on Jamba Reasoning 3B | ⬜ NOT STARTED |
| 5 | Quantization / Edge Deploy | ⬜ NOT STARTED |

### TRC Status
✅ Accepted (email 2026-04-07). **Do not claim quota yet.** Claim after K=4 gate passes on Kaggle Dual T4.

### Part 0.1 — Immediate Next Actions (ordered)

1. **Run** `prepare_coconut_dataset.py --output_dir data/coconut_v1 --push_to_hub --hf_token YOUR_TOKEN`
   - Produces `train.jsonl`, `val.jsonl`, `stats.json`
   - Pushes to Hub as `WeirdRunner/Ouroboros` config `coconut-v1` (dataset, not model)
   - Read `stats.json → train.n_steps_median` → this becomes `--max_stage` for training

2. **Smoke test** (Colab/Kaggle T4, must pass ALL checklist items):
   ```
   python jamba_coconut_finetune.py \
     --data_dir data/coconut_v1 --use_4bit \
     --epochs_per_stage 1 --max_stage 2 --max_samples 200 \
     --max_seq_len 1024 --max_grad_norm 0.3 \
     --session_timeout_hours 1.5 --wandb_mode disabled --output_dir runs/smoke
   ```
   The script self-installs all dependencies on startup (~20-30 min first run for mamba-ssm source build; seconds if already installed). Verify: `mamba CUDA kernels: OK` in output, val_acc non-zero, grad_norm < 2.0.

3. **K=0→K_max curriculum** on Kaggle Dual T4 with `torchrun --nproc_per_node=2`

4. **If K=4 gate passes**: claim TRC, run K=16 + DGAC Phase 3.4 on A100

### Part 0.2 — Pre-flight Blockers Resolved (updated 2026-04-12)

All blockers now addressed:

| Blocker | Resolution |
|---|---|
| `attn_implementation` hardcoded crash | try/except fallback to `eager` (load_model_and_tokenizer) |
| `use_mamba_kernels` kwarg on old TF | `_safe_from_pretrained` retries without kwarg if rejected |
| `use_mamba_kernels=False` hardcoded | Replaced with kernel probe; only set False on ImportError |
| `last_hidden_state` silent None | `assert out.last_hidden_state is not None` in both Stage 0 and Phase B |
| No graceful timeout → lost Kaggle work | `make_timeout_checker()` + emergency save integrated in main loop |
| `conv1d` in LoRA targets (malformed) | Explicitly excluded with comment |
| NCCL watchdog (killed S5–S7) | `timeout=timedelta(minutes=60)` + `TORCH_NCCL_ASYNC_ERROR_HANDLING=1` |
| OOM at first val step (S8) | `torch.cuda.empty_cache()` + `val_batch_size=1` default |
| `--max_seq_len 512` filtered all stage 1+ samples | Default changed to 1024; Jamba traces are long |
| Exploding gradients at k≥2 (gn=36.9) | `--max_grad_norm 0.3` in smoke test and real run commands |
| Wheel/ABI mismatch across Kaggle GPU allocations | **Retired wheel workflow.** Script self-installs via `_bootstrap()` every run; always matches live GPU+PyTorch ABI. ~20-30 min first run per fresh session. |
| mamba-ssm 2.x API broke `selective_state_update` path | Pinned to `mamba-ssm==1.2.2` in `_bootstrap()` |
| Blackwell GPU (sm_120) not in 1.2.2 arch list | Moot — source build compiles against live env automatically via `--no-build-isolation` |

One item still requires empirical verification during smoke test:
- `inputs_embeds` → `last_hidden_state` path for Jamba Reasoning 3B with fast Mamba kernels active (previously only verified on slow path)

---

## Part 1 — Architecture

### Our TRM-Mamba nano (retired)
```
92.5M params. Pretrained ~700M tokens. Architecture proven. No further training.
```

### Jamba Reasoning 3B (primary research model)
```
HuggingFace : ai21labs/AI21-Jamba-Reasoning-3B   License: Apache 2.0
Layers      : 28 total (26 Mamba, 2 Attention) → 13:1 Mamba:Attention ratio
Attention   : MQA (20 Q heads, 1 KV head)
Vocab       : 64K
Context     : 256K tokens
Post-training: mid-train 0.5T tokens (math+code) → cold-start SFT → DPO → RLVR
Reasoning   : native <think>...</think> traces; vLLM uses deepseek_r1 parser
```

**Why Reasoning 3B over Jamba2-3B:** Reasoning 3B already has explicit CoT traces that Coconut progressively replaces with latent passes. Jamba2-3B targets enterprise grounding (IFBench, RAG), has no reasoning traces, and would require re-teaching CoT from scratch.

---

## Part 2 — Resolved Decisions

| Decision | Resolution |
|---|---|
| Primary research model | Jamba Reasoning 3B (ai21labs/AI21-Jamba-Reasoning-3B, Oct 2025) |
| Fine-tuning approach | QLoRA on GPU (bitsandbytes, 4-bit NF4); LoRA only on TPU — `--use_4bit` flag |
| LoRA target modules | q_proj, k_proj, v_proj, o_proj (attention) + in_proj, x_proj, dt_proj, out_proj (Mamba); conv1d excluded |
| Coconut curriculum | Paper's correct mechanism: progressive step replacement (NOT fixed-K tokens at question end) |
| Stage advancement | Epoch-based (fixed epochs_per_stage) + best-accuracy checkpoint selection at end of stage |
| Dataset pipeline | Separate `prepare_coconut_dataset.py` → canonical JSONL; training script never touches HF datasets |
| Dataset sources | Same 5 sources as sft-mix-v1, re-processed from scratch with step segmentation |
| max_stage (K) | Set automatically from stats.json (median n_steps); override via `--max_stage` |
| DGAC halt gate | Phase 3.4 only, after K-stage curriculum complete; λ₁ annealed from 0 |
| Halt threshold | Exposed as `--halt_threshold` (default 0.5) — tunable at inference |
| TRC timing | Claim after K=4 gate passes, not before |
| Platform: dataset prep | Local or Kaggle CPU |
| Platform: smoke test | Free Colab/Kaggle T4, QLoRA, batch=1, max_samples=200 |
| Platform: K=0→K_max | Kaggle Dual T4 (2×16GB), QLoRA + DDP |
| Platform: K=16 + DGAC | TRC A100 80GB |
| Stage 1 gate | Bypassed (val_ce=5.32). Architecture proven. Hub: ckpt-0021000. |
| Stage 2 dataset | sft-mix-v1 (55k, WeirdRunner/Ouroboros) — SFT format; NOT reused for Coconut |
| Coconut dataset Hub config | `coconut-v1` under `WeirdRunner/Ouroboros` (dataset repo) |
| `attn_implementation` | Detected at runtime: flash_attention_2 if available, else eager |
| `use_mamba_kernels` | Probe at runtime: import `selective_scan_fn` + `causal_conv1d_fn`; only set False if ImportError |
| Mamba install strategy | **`_bootstrap()` in `jamba_coconut_finetune.py`** — self-installs `mamba-ssm==1.2.2` + `causal-conv1d>=1.4.0` via `pip install --no-build-isolation` on every run. Source builds against live PyTorch/CUDA env automatically. ~20-30 min first run; seconds if already installed. No wheel caching, no Hub wheel repo, no arch list management. `build_wheels_kaggle.py` retained as fallback for TRC A100 sessions only. |
| Session timeout (Kaggle) | `--session_timeout_hours 11.0 --graceful_exit_buffer_minutes 20` |
| val_batch_size | Default 1 (Stage 2 S8 OOM'd with val_batch_size=16 at seq=2048) |
| NCCL timeout | `timedelta(minutes=60)` in init_process_group |
| `--max_seq_len` for training | 1024 (512 filtered ~100% of Jamba traces at stage ≥ 1; confirmed smoke test) |
| `--max_grad_norm` for k≥2 | 0.3 (gn=36.9 at k=2 with only 1 step; confirmed smoke test) |

---

## Part 3 — Open Questions

| Question | Status |
|---|---|
| Does `inputs_embeds` → `last_hidden_state` work cleanly with Jamba Reasoning 3B **with fast Mamba kernels active**? | 🟡 VERIFY during next smoke test (previously verified slow-path only) |
| Jamba Reasoning 3B mamba-ssm TPU XLA fallback: does `use_mamba_kernels=False` compile? | 🔴 OPEN — not blocking for GPU path |
| DGAC Phase 3.4: does mean halt_step distribute across K≥2 after training? | 🔴 OPEN — primary research validation metric |
| Optimal max_stage K: does stats.json median match what Reasoning 3B's traces segment to? | 🟡 CHECK after running prepare_coconut_dataset.py |

---

## Part 4 — File Registry

| File | Stage | Status | Notes |
|---|---|---|---|
| `baseline_trm_mamba.py` | 0 | ✅ COMPLETE | Retired; architecture reference only |
| `viability_gate.py` | 0 | ✅ COMPLETE | |
| `training_utils.py` | All nano | ✅ COMPLETE | Not used in Jamba scripts |
| `pretrain.py` | 1 | ✅ COMPLETE | Hub: ckpt-0021000 |
| `prepare_sft_dataset.py` | 2 | ✅ DONE | sft-mix-v1 cached; not reused for Coconut |
| `train_sft.py` | 2 | ✅ PATCHED (DDP v2) | Retired |
| `prepare_coconut_dataset.py` | 3 | ✅ PATCHED (2026-04-11) | Added hub push, Colab token, Drive backup, `--push_to_hub` CLI |
| `jamba_coconut_finetune.py` | 3 | ✅ SELF-SUFFICIENT (2026-04-12) | `_bootstrap()` self-installs all deps; no external wheel setup required |
| `build_wheels_kaggle.py` | 3 | 🟡 FALLBACK ONLY | Retained for TRC A100 sessions where session overhead budget justifies pre-built wheels; not needed for Kaggle runs |

---

## Part 5 — Coconut Curriculum Design

### Correct mechanism (Meta paper)

```
Stage 0:  [Q][S1][S2][S3][A]     ← standard CoT; labels on S1..Sn + A
Stage 1:  [Q][●][S2][S3][A]      ← S1 replaced; labels on S2..Sn + A
Stage k:  [Q][●*k][S_{k+1}..Sn][A]   ← first k steps replaced; labels shift
Stage K:  [Q][●*K][A]            ← all steps replaced; labels on A only
```

`●` = injected latent hidden state from sequential prefix pass. K = median n_steps from dataset.

### Dataset format (output of prepare_coconut_dataset.py)
```json
{
  "id":          "bespoke_0042",
  "source":      "bespoke_stratos",
  "question":    "Solve: ...",
  "steps":       ["Step 1 text", "Step 2 text", ...],
  "answer_full": "The answer is 42.",
  "answer_norm": "42",
  "n_steps":     3
}
```

**Hub note:** When loaded from Hub backup, `steps` is a JSON-encoded string. `load_canonical_dataset()` in `jamba_coconut_finetune.py` handles deserialization transparently.

---

## Part 6 — DGAC: Diversity-Gated Adaptive Coconut

```
L_total = L_ce  +  λ₁(t) · L_ponder  +  λ₂ · L_diversity

L_ponder    = ACT ponder cost (ACT-style remainder sum)
L_diversity = mean_batch( Σ_k relu(cos_sim(h_k, h_{k-1}) − τ) )
              τ = 0.9 — penalizes passes where hidden state barely changed

λ₁ schedule (Phase 3.4):
  steps 0..200:    λ₁ = 0,  λ₂ = 0.1   (gate learns what "changed" means)
  steps 200..500:  λ₁ ramps 0→0.01     (introduce efficiency pressure)
  steps 500+:      λ₁ = 0.01
```

**HaltGate:** Linear(2*d_model → 1), zero-initialized → outputs 0.5 at Phase 3.4 start.
**Training mode:** soft ACT (weighted combination). **Inference mode:** hard halt at `halt_threshold`.

---

## Part 7 — Checkpoint Format (Stage-Scoped)

```
output_dir/
  stage_0/
    checkpoint-0001234/
      adapter_model/         ← PEFT LoRA weights (.safetensors or .bin)
      training_state.pt      ← {stage_k, step, epoch, val_ce, val_acc, optimizer, scheduler}
    best/                    ← best-accuracy checkpoint for this stage
      adapter_model/
      halt_gate.pt           ← HaltGate state dict (Phase 3.4 only)
      training_state.pt
  stage_1/
    best/
  ...
  stage_K/
    best/                    ← resume_from for Phase 3.4 DGAC run
```

---

## Part 8 — Compute Plan

| Phase | Platform | Approach | Estimate |
|---|---|---|---|
| Dataset prep | CPU (local or Kaggle) | prepare_coconut_dataset.py | ~30–60 min |
| Smoke test | Free Colab/Kaggle T4 (15GB) | QLoRA, batch=1, max_stage=2, max_samples=200, max_seq_len=1024 | ~30-60 min first run (mamba build) + ~10 min training |
| Stage 0→K | Kaggle Dual T4 (2×16GB) | QLoRA + DDP, epochs_per_stage=3 | ~25-30 min first-session install + ~4–8h training per session |
| Phase 3.4 (DGAC) | TRC A100 80GB | QLoRA or LoRA, grad_ckpt | ~6–8h |
| Phase 4 (GRPO) | TRC A100 80GB | TBD | ~8–12h |

---

## Part 9 — Hard Lessons from Prior Stages (Do Not Repeat)

| Lesson | Source | Codified As |
|---|---|---|
| val_batch_size=16 at seq=2048 → 9.25 GiB OOM | Stage 2 S8 | `--val_batch_size 1` default + `torch.cuda.empty_cache()` before val |
| NCCL watchdog kills DDP after val+gen timeout | Stage 2 S5–S7 | `timeout=timedelta(minutes=60)` + graceful exit before budget expires |
| No timeout → losing hours of training | Stage 2 S5–S7 | `make_timeout_checker()` in main loop; emergency checkpoint on trigger |
| max_seq_len=512 filtered ~100% of stage 1+ samples | Stage 3 smoke test | `--max_seq_len 1024`; Jamba traces are long |
| gn=36.9 at k=2 with few steps | Stage 3 smoke test | `--max_grad_norm 0.3` for k≥2 stages |
| Hard-coded `attn_implementation` crash on Colab | Audit 2026-04-11 | try/except flash-attn detection with eager fallback |
| `use_mamba_kernels=False` HARDCODED → ~100× slowdown always | Audit 2026-04-12 | Runtime probe: import `selective_scan_fn`; only set False on failure |
| mamba-ssm 2.x moved `selective_state_update` → Triton path; fast path silently disabled | Wheel build session 2026-04-12 | Pinned to `mamba-ssm==1.2.2` in `_bootstrap()` |
| Kaggle GPU arch unpredictable (labelled T4, allocated Blackwell sm_120) | Wheel build session 2 2026-04-12 | **Retired wheel caching entirely.** `--no-build-isolation` source build in `_bootstrap()` always compiles for the actual live GPU. Arch list management complexity eliminated. |
| Fat-wheel multi-arch build time (~60-80 min) exceeds savings for Kaggle session frequency | Decision 2026-04-12 | Direct pip install wins on total complexity budget; `build_wheels_kaggle.py` kept as TRC-only fallback |
