# Project Ouroboros — Master Blueprint

> **Thread-resume header. Read Part 0 first in any new session.**
> DRY Rule: Session details live in `terminal_log.md`. Blueprint holds decisions, architecture, status, and next actions only. When cleaning up, summarise sessions to one-liners here and preserve full detail in terminal_log.md. Never delete architectural decisions, resolved bugs, or open questions.

---

## Part 0 — Quick-Resume Context

### What this project is
Novel hybrid Transformer-Mamba language model ("TRM-Mamba", 1:7 Transformer:Mamba ratio) pre-trained on FineWeb-Edu, fine-tuned for chain-of-thought reasoning, with test-time compute scaling via Coconut-Ouroboros recursive latent inference (SERF framework).

### Current status

| Stage | Name | Status | Gate |
|---|---|---|---|
| 0 | Architecture & Viability | ✅ COMPLETE | All 4 gates passed |
| 1 | Pre-training | ✅ COMPLETE | val_ce=5.32 @ step 21501; Hub: ckpt-0021000 |
| 2 | SFT | 🔴 BLOCKED — S9 completed 3 epochs, val_acc declining to ~0, no `<think>` tags, UWR=0.08. Needs diagnosis before S10. | answer val_ce < 1.5 |
| 3 | Recursive Inference (Coconut-Ouroboros) | ⬜ NOT STARTED | Stage 2 gate |
| 4 | GRPO | ⬜ NOT STARTED | Stage 3 gate |
| 5 | Quantization | ⬜ NOT STARTED | Stage 4 gate |

### TRC Status
✅ Accepted (email 2026-04-07). 30-day trial starts after submitting GCP project number. **Do not create TPUs yet — claim quota only once Stage 2 is resolved.**

### Immediate next actions (S10 plan)

**Root cause to fix first** — before launching another full run:
1. Add temperature sampling (`temp=0.8, top_p=0.9`) to generation callback — greedy decoding masks true quality on an under-trained model
2. Add `dropout=0.1` to `BaselineConfig` for regularization (val_acc declining = overfitting signal)
3. Investigate the declining val_acc: check raw val_ce values on wandb (not just accuracy). If val_ce is also rising while train_ce falls → confirmed overfitting
4. Consider reducing lr to `1e-4` and running with more epochs on TPU

**S10 run command (single GPU, diagnostic):**
```bash
python train_sft_single_gpu.py \
  --preset nano \
  --max_seq_len 2048 \
  --dataset_mix cached \
  --num_epochs 5 \
  --batch_size 2 \
  --grad_accum 8 \
  --lr 1e-4 \
  --warmup_steps 100 \
  --ema_decay 0.995 \
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

**Fixed constants:** vocab_size=151_680, rope_theta=1e6, mamba_d_state=16, mamba_d_conv=4, mamba_expand=2, max_seq_len=2048, dropout=0.0, rms_norm_eps=1e-5, tie_embeddings=True.

**Stage 3 additions needed (not yet applied):** `forward_with_hidden()` and `forward_from_embeddings()` methods on `BaselineTRMMamba`. See `stage3_agent_prompt.md`.

---

## Part 2 — Resolved Decisions

| Decision | Resolution |
|---|---|
| Tokenizer | Qwen2.5-0.5B; vocab_size=151,665 padded to 151,680 |
| Stage 1 gate | Bypassed (val_ce=5.32 > 3.0 gate, but loss was descending and architecture proven healthy). SFT starts from Hub ckpt-0021000. |
| Stage 2 max_seq_len | 2048 — 1024 filtered 97% of reasoning chains |
| Stage 2 dataset | Full mix: Stratos(30%) + MetaMathQA(20%) + OpenHermes(15%) + OpenR1-Math(20%) + OpenR1-Code(15%), ~55,230 samples. Pre-processed and cached at `WeirdRunner/Ouroboros` (dataset config `sft-mix-v1`). |
| Stage 2 target format | `User: {q}\n\nAssistant: <think>\n{reasoning}\n</think>\n{answer}{eos}` |
| Stage 2 label masking | Only answer tokens supervised (prompt_len mask). `<think>` tag IS supervised (part of answer portion). |
| Stage 2 val_batch_size | **2** (not 16). At seq=2048, vocab=151680, batch=16 → 9.25 GiB logit tensor → OOM on T4. |
| DDP NCCL timeout | Set `NCCL_TIMEOUT=1800` (30 min) before `init_process_group`. Combined val+gen must stay under this. |
| Hub checkpoint format | Stage 2 checkpoints under `runs/stage2/` subdir in `WeirdRunner/Ouroboros` model repo. |
| Stage 3 recursion | Coconut-Ouroboros, K=1→4→16 curriculum. See `stage3_agent_prompt.md`. |
| Mamba 2 | **Deferred.** Incompatible weights. Best introduction point: when scaling to `small/medium` preset on TPU, or as a standalone architecture experiment after Stage 2 gate. |

---

## Part 3 — Open Decisions / Questions

| Question | Status |
|---|---|
| S9 root cause: overfitting vs. bad label masking vs. LR too high? | 🔴 OPEN — check wandb val/ce curves, not just val/acc |
| Should we add `dropout=0.1`? | Likely YES given declining val_acc. Needs confirmation. |
| Sampling-based generation callback? | YES — add `temp=0.8, top_p=0.9`. Greedy is deceptive for under-trained models. |
| Should we pivot to fine-tuning existing models (Qwen2.5, etc.) instead of training from scratch? | 🟡 CONSIDERED — not now; from-scratch research is the point. Revisit after Stage 2 gate or if model stays degenerate after 3 more sessions. |

---

## Part 4 — File Registry

| File | Stage | Status | Notes |
|---|---|---|---|
| `baseline_trm_mamba.py` | 0 | ✅ COMPLETE | Stage 3 needs 2 method additions |
| `viability_gate.py` | 0 | ✅ COMPLETE | |
| `training_utils.py` | All | ✅ COMPLETE | Canonical utilities — never duplicate locally |
| `pretrain.py` | 1 | ✅ COMPLETE | Last Hub ckpt: ckpt-0021000 |
| `prepare_sft_dataset.py` | 2 | ✅ DONE | Dataset cached at HF (`sft-mix-v1`, 55,230 samples) |
| `train_sft.py` | 2 | ✅ PATCHED (DDP v2) | S8 script; S9 used single-GPU variant |
| `train_sft_single_gpu.py` | 2 | 🟡 IN USE | S9 used this; has `val_batch_size=16` default (safe on single GPU) |
| `train_sft_fixes_prompt.md` | 2 | ✅ APPLIED | |
| `stage2_rewrite_prompt.md` | 2 | ✅ APPLIED | |
| `stage3_agent_prompt.md` | 3 | ✅ READY | Feed to coding agent after Stage 2 gate |
| `recursive_finetune.py` | 3 | ⬜ NOT CREATED | Generate from `stage3_agent_prompt.md` |

---

## Part 5 — Stage 2 Checkpoint State

| Hub checkpoint | Step | val_ce | Notes |
|---|---|---|---|
| ckpt-0003500 | 3500 | ~5.6 | Old full-mix run (S5–S7); weights loaded in S8 with optimizer reset |
| ckpt-0000250 | 250 | — | S8 partial (OOM); used as S9 starting point |
| S9 final | ~800 | — | val_acc declining; train_ce=2.17; **not yet pushed to Hub** |

**Resume fingerprint:** S9 used `dataset_mix=cached`, `sft-mix-v1`. Any resume must match this or optimizer resets automatically.

---

## Part 6 — Stage 2 Hyperparameters

```
# Current (S9)
batch_size=2, grad_accum=8  → effective batch=16 (single GPU)
max_seq_len=2048, lr=3e-4, warmup_steps=50, ema_decay=0.99
val_max_samples=500, val_batch_size=2
dataset_mix=cached (sft-mix-v1, 55,230 samples → 52,469 train)
num_epochs=3, total_steps≈9,840 (3280 steps/epoch)
~3.5s/step on single T4 → ~9.6h for 3 epochs; S9 timed out at ~800 steps

# Proposed (S10)
lr=1e-4, warmup_steps=100, ema_decay=0.995, dropout=0.1
num_epochs=5 (give it time to grok)
```

---

## Part 7 — Stage 3 Coconut-Ouroboros (Ready to implement post-Stage 2)

| Sub-stage | K | Resume from | Output dir |
|---|---|---|---|
| 3.1 | 1 | Stage 2 final | runs/stage3_k1 |
| 3.2 | 4 | Stage 3.1 final | runs/stage3_k4 |
| 3.3 | 16 | Stage 3.2 final | runs/stage3_k16 |

Gate: answer val_ce ≤ stage2_val_ce × 1.05. Full spec: `stage3_agent_prompt.md`.

---

## Part 8 — Checkpoint Format Reference

```python
# Stage 2
{"stage": "sft", "step", "epoch", "samples_seen", "val_ce",
 "model_state_dict", "ema_backbone_state_dict",
 "optimizer", "scheduler", "scaler", "ema",
 "backbone_config", "sft_config", "data_fingerprint"}

# Stage 3 adds
{"stage": "coconut", "n_latent", "lat_token_id", "vocab_size"}
```

Stage 2 loader: Stage 1 checkpoint → load weights + EMA, reset optimizer (step=0). Fingerprint mismatch → load weights only, reset optimizer (step=0).

---

## Part 9 — Compute Plan

| Stage | Platform | Estimate |
|---|---|---|
| 2 | Kaggle Single T4 | ~10h/session for 3 epochs. 1–2 more sessions needed. |
| 3 | TRC (TPU v3-8) | ~4–8h per K sub-stage. Claim quota when Stage 2 done. |
| 4 | TRC + unsloth | ~8–12h |
| 5 | Local / Jetson | ~2h |
