# Project Ouroboros — Master Blueprint

> **Thread-resume header.** Read Part 0 first in any new session.

---

## Part 0 — Quick-Resume Context

### What this project is
Novel hybrid Transformer-Mamba language model ("TRM-Mamba", 1:7 ratio) pre-trained on FineWeb-Edu, fine-tuned for chain-of-thought reasoning, with test-time compute scaling via Coconut-Ouroboros recursive inference (SERF).

### Current status

| Stage | Name | Status | Gate |
|---|---|---|---|
| 0 | Architecture & Viability | ✅ COMPLETE | All 4 gates passed |
| 1 | Pre-training | ✅ COMPLETE | val_ce=5.32 @ step 21501; Hub: ckpt-0021000 |
| 2 | SFT | 🟡 IN PROGRESS — S8 ran 250 steps, OOM at val; S9 pending after `train_sft_fixes_prompt.md` applied | answer val_ce < 1.5 |
| 3 | Recursive Inference (Coconut-Ouroboros) | ⬜ NOT STARTED | Stage 2 gate |
| 4 | GRPO | ⬜ NOT STARTED | Stage 3 gate |
| 5 | Quantization | ⬜ NOT STARTED | Stage 4 gate |

### Immediate next actions

1. **Apply `train_sft_fixes_prompt.md`** to `train_sft.py` — lowers `val_batch_size` default to 2, flushes CUDA cache before val, fixes `NCCL_ASYNC_ERROR_HANDLING` deprecation, and adds fingerprint-aware resume bucketing.
2. **Run S9** with the patched script (command in the prompt's verification checklist).

**S9 run command:**
```bash
python train_sft.py \
  --preset nano \
  --max_seq_len 2048 \
  --dataset_mix cached \
  --num_epochs 3 \
  --batch_size 4 \
  --grad_accum 8 \
  --lr 3e-4 \
  --warmup_steps 50 \
  --ema_decay 0.99 \
  --val_max_samples 500 \
  --val_batch_size 2 \
  --output_dir runs/stage2 \
  --push_to_hub \
  --hf_token $HF_TOKEN \
  --wandb_project ouroboros-stage2 \
  --save_every 500 \
  --val_every 250 \
  --gen_every 500 \
  --session_timeout_hours 11.5 \
  --graceful_exit_buffer_minutes 15
```

---

## Part 1 — Architecture

```
TokenEmbedding(vocab_size=151_680, d_model)
└─ n_groups × TRMMambaGroup:
   ├─ 1 × TRMBlock: RMSNorm + GQA(RoPE) + residual; RMSNorm + SwiGLU + residual
   └─ 7 × MambaLayer: RMSNorm + mamba_ssm.Mamba + residual   ← 1:7 ratio, fixed
FinalRMSNorm → LM Head (weight-tied)
```

| Preset | d_model | n_groups | n_heads | n_kv_heads | Params |
|---|---|---|---|---|---|
| nano | 512 | 1 | 8 | 4 | 92.5M |
| small | 1024 | 2 | 16 | 8 | ~270M |
| medium | 2048 | 2 | 16 | 8 | ~760M |

**Fixed:** vocab_size=151_680, rope_theta=1e6, mamba_d_state=16, mamba_d_conv=4, mamba_expand=2, max_seq_len=2048, dropout=0.0, rms_norm_eps=1e-5, tie_embeddings=True.

---

## Part 2 — Resolved Decisions

| Decision | Resolution |
|---|---|
| Tokenizer | Qwen2.5-0.5B; vocab_size=151,680 |
| Stage 1 | Bypassed gate (val_ce=5.32 at step 21501). SFT starts from Hub ckpt-0021000. |
| Stage 2 max_seq_len | 2048 — 1024 filtered 97% of reasoning chains |
| Stage 2 dataset | Full mix: Stratos + MetaMathQA + OpenHermes + OpenR1-Math + OpenR1-Code (~55k samples). Pre-processed and cached at `WeirdRunner/Ouroboros` (dataset repo, config `sft-mix-v1`). |
| Stage 2 target format | `User: {q}\n\nAssistant: <think>\n{reasoning}\n</think>\n{answer}{eos}` |
| Stage 2 starting weights | Hub ckpt-0003500 (old full-mix; weights loaded, optimizer reset on fingerprint change) |
| Stage 2 DDP | **Kept.** DDP worked for 250 steps in S8 before OOM. NCCL issues resolved. |
| Stage 2 val_batch_size | **2** (not 16). At seq=2048, vocab=151680, batch=16 → 9.25 GiB logit tensor → OOM. |
| Stage 3 recursion | Coconut-Ouroboros, K=1→4→16. See `stage3_agent_prompt.md`. |
| Hub repo | WeirdRunner/Ouroboros (private model repo + dataset repo) |

---

## Part 3 — File Registry

| File | Stage | Status | Notes |
|---|---|---|---|
| `baseline_trm_mamba.py` | 0 | ✅ COMPLETE | Stage 3 needs `forward_with_hidden` + `forward_from_embeddings` (see `stage3_agent_prompt.md`) |
| `viability_gate.py` | 0 | ✅ COMPLETE | |
| `training_utils.py` | All | ✅ COMPLETE | Canonical Hub/checkpoint/EMA utilities. Do not duplicate. |
| `pretrain.py` | 1 | ✅ COMPLETE | Last Hub ckpt: ckpt-0021000 |
| `prepare_sft_dataset.py` | 2 | ✅ DONE | Dataset uploaded to HF as `sft-mix-v1` (55,230 samples) |
| `train_sft.py` | 2 | 🟡 NEEDS PATCH | Apply `train_sft_fixes_prompt.md` |
| `train_sft_single_gpu.py` | 2 | 🗄️ ARCHIVED | Superseded by `train_sft.py` (DDP v2) |
| `train_sft_fixes_prompt.md` | 2 | ✅ READY | Feed to coding agent before S9 |
| `stage3_agent_prompt.md` | 3 | ✅ READY | Feed to coding agent after Stage 2 gate |
| `recursive_finetune.py` | 3 | ⬜ NOT CREATED | Generate from `stage3_agent_prompt.md` |

---

## Part 4 — Stage 2 Session History

| Session | Outcome | Root Cause |
|---|---|---|
| S1 | val_ce=4.92 plateau. Hub: ckpt-0002979. | max_seq_len=1024 filtered 97% of reasoning chains. |
| S2 | val_ce=5.71. All local ckpts deleted. | Bugs 6–10 cascade; prune bug wiped local checkpoints. |
| S3–S4 | Dry-runs only. | Patches from `stage2_patch_prompt.md` verified. |
| S5 | SIGKILL (exit 137) at step ~3700. | val(557s)+gen(90s) combined > NCCL 600s timeout. |
| S6 | SIGABRT at step 3750. | Same NCCL watchdog. `stage2_rewrite_prompt.md` not applied. |
| S7 | SIGABRT at step 3522. | Same NCCL. Rewrite still not applied. |
| S8 | 250 steps clean, OOM at first val. Hub: ckpt-0000250. | `val_batch_size=16` → 9.25 GiB logit tensor at seq=2048. |
| S9 | ⬜ NEXT — after `train_sft_fixes_prompt.md` applied | — |

**Why val OOM'd:** `logits[:, :-1, :].contiguous().view(-1, vocab_size).float()` with batch=16, seq=2048, vocab=151680 allocates `16 × 2047 × 151680 × 4 = 9.25 GiB`. Fix: default val_batch_size=2 (1.16 GiB) + `torch.cuda.empty_cache()` before val.

---

## Part 5 — Stage 2 Hyperparameters (S9 target)

```
batch_size=4, grad_accum=8  → effective batch=32 (4 per GPU × 2 GPUs × 8 accum steps)
max_seq_len=2048, lr=3e-4, warmup_steps=50, ema_decay=0.99
val_max_samples=500, val_batch_size=2
dataset_mix=cached (WeirdRunner/Ouroboros, sft-mix-v1, 55,230 samples)
num_epochs=3, total_steps≈4920 (52,469 train samples ÷ 32 eff batch)
~10s/step on dual T4 → ~14h for 3 epochs
```

---

## Part 6 — Stage 3 Coconut-Ouroboros

| Sub-stage | K | Resume from | Output dir |
|---|---|---|---|
| 3.1 | 1 | Stage 2 final | runs/stage3_k1 |
| 3.2 | 4 | Stage 3.1 final | runs/stage3_k4 |
| 3.3 | 16 | Stage 3.2 final | runs/stage3_k16 |

Full spec: `stage3_agent_prompt.md`. Requires two additions to `baseline_trm_mamba.py`: `forward_with_hidden()` and `forward_from_embeddings()`. Gate: answer val_ce ≤ stage2_val_ce × 1.05.

---

## Part 7 — Checkpoint Format

```python
# Stage 2 (train_sft.py)
{
    "stage": "sft",
    "step", "epoch", "samples_seen", "val_ce",
    "model_state_dict", "ema_backbone_state_dict",
    "optimizer", "scheduler", "scaler", "ema",
    "backbone_config", "sft_config", "data_fingerprint",
}
# Stage 3 adds: "stage": "coconut", "n_latent", "lat_token_id", "vocab_size"
```

Stage 2 loader handles Stage 1 checkpoints as cold-start (load weights + EMA, reset optimizer, return step=0). Fingerprint mismatch also triggers reset with weights loaded.

---

## Part 8 — Compute Plan

| Stage | Platform | Estimate |
|---|---|---|
| 2 | Kaggle Dual T4 | ~14h/session for 3 epochs. 1–2 sessions. |
| 3 | TRC preferred | ~4–8h per K sub-stage |
| 4 | TRC + unsloth | ~8–12h |
| 5 | Local / Jetson | ~2h |

TRC application submitted 2026-04-07. Awaiting email.
