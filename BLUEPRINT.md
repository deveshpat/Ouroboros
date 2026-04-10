# Project Ouroboros — Master Blueprint

> **Thread-resume header. Read Part 0 first in any new session.**
> **DRY Guidelines:** Session details live in `terminal_log.md`. Blueprint holds decisions, architecture, status, and next actions only.
> **Cleanup rules (for future DRY passes):** Summarise sessions into one-liners — never delete them. Preserve all architectural decisions and resolved bugs. Remove content only if it exists verbatim in the other file or has been superseded by a dated resolution.

---

## Part 0 — Quick-Resume Context

### What this project is
Coconut-Ouroboros latent reasoning injection into a production Transformer-Mamba hybrid. The Mamba SSM recurrent state is used as compressed scratch-memory across K latent thought passes, bypassing token generation during reasoning. Based on Meta's Coconut paper (arXiv:2412.06769) adapted for SSM architectures.

### Strategic Status (Updated 2026-04-10)

**PIVOT DECISION (2026-04-10):** Abandoned from-scratch training as the primary research vehicle. AI21's Jamba paper independently confirmed our exact architecture choices (1:7 ratio, Mamba 1 over Mamba 2 in hybrid). Stages 1–2 from scratch add no scientific value beyond what already exists. Novel contribution is exclusively Coconut-Ouroboros. **Jamba2-3B (Apache 2.0) is the new primary base model.**

| Stage | Name | Status |
|---|---|---|
| 0 | Architecture & Viability (nano) | ✅ COMPLETE |
| 1 | Pre-training (nano) | ✅ Pipeline test only; not the research artifact |
| 2 | SFT (nano) | 🔴 DEPRIORITISED — one more session for code/pipeline validation only |
| 3 | Coconut-Ouroboros on Jamba2-3B | 🟡 NEXT — claim TRC, adapt `stage3_agent_prompt.md` |
| 4 | GRPO on Jamba2-3B | ⬜ NOT STARTED |
| 5 | Quantization / Edge Deploy | ⬜ NOT STARTED |

### TRC Status
✅ Accepted (email 2026-04-07). **Claim quota now** — no longer waiting for Stage 2 gate on nano. Create GCP project → submit project number → wait for confirmation → then create TPUs.

### Immediate Next Actions

1. **One more Kaggle session:** Run nano S10 with `lr=1e-4, dropout=0.1, temp sampling`. Goal: confirm Coconut pipeline code is correct end-to-end, not to produce a capable model. Also generate `recursive_finetune.py` from `stage3_agent_prompt.md` and dry-run it.
2. **Claim TRC quota.**
3. **Write `jamba_coconut_finetune.py`:** Adapt `recursive_finetune.py` for Jamba2-3B via HuggingFace `inputs_embeds` API. See Part 7.
4. **On TPU:** Run Coconut curriculum K=1→4→16 on Jamba2-3B.

---

## Part 1 — Architecture

### Our TRM-Mamba nano (pipeline test harness only)
```
TokenEmbedding(vocab_size=151_680, d_model=512)
└─ 1 × TRMMambaGroup:
   ├─ TRMBlock: RMSNorm + GQA(RoPE, rope_theta=1e6) + residual; RMSNorm + SwiGLU + residual
   └─ 7 × MambaLayer: RMSNorm + Mamba1 + residual
FinalRMSNorm → LM Head (weight-tied)
```
92.5M params. Pretrained ~700M tokens. SFT partial.

### Jamba2-3B (primary research model)
```
HuggingFace: ai21labs/AI21-Jamba2-3B   License: Apache 2.0
Architecture: 1:7 Attn:Mamba, Mamba 1, GQA, RMSNorm, MoE SwiGLU, NO RoPE, 256K context
Post-training: mid-training 500B tokens → cold-start SFT → DPO
```

### Architecture Comparison

| Feature | Our nano | Jamba2-3B | Notes |
|---|---|---|---|
| Attn:Mamba ratio | 1:7 | 1:7 | AI21 confirmed identical by ablation |
| Mamba version | Mamba 1 | Mamba 1 | AI21: Mamba2 worse in hybrid (Appendix C.2) |
| Normalization | RMSNorm | RMSNorm | |
| Attention | GQA | GQA | |
| Positional encoding | RoPE | None | Mamba handles position implicitly; RoPE redundant but harmless |
| FFN | Dense SwiGLU | MoE SwiGLU | Irrelevant for Coconut injection |
| Scale | 92.5M / 700M tokens | 3B / 500B+ tokens | |
| Instruction following | Broken | Production-grade | |

---

## Part 2 — Resolved Decisions

| Decision | Resolution |
|---|---|
| Tokenizer | Qwen2.5-0.5B; vocab 151,665 padded to 151,680 (nano only) |
| Stage 1 gate | Bypassed (val_ce=5.32). Architecture proven healthy. Hub: ckpt-0021000. |
| Stage 2 dataset | Full mix cached at `WeirdRunner/Ouroboros` (config `sft-mix-v1`, 55,230 samples) |
| Stage 2 target format | `User: {q}\n\nAssistant: <think>\n{reasoning}\n</think>\n{answer}{eos}` |
| Stage 2 val_batch_size | 2 (batch=16 at seq=2048 → 9.25 GiB OOM on T4) |
| DDP NCCL timeout | 1800s |
| Hub ckpt subdir | `runs/stage2/` in `WeirdRunner/Ouroboros` |
| Mamba 2 | Not needed. AI21 confirmed Mamba 1 > Mamba 2 in hybrid (Jamba 1.5 paper). |
| RoPE in hybrid | Redundant (Mamba handles position implicitly). Not worth changing nano. Jamba2 correctly omits it. |
| From-scratch vs. pretrained base | **Pivoted to Jamba2-3B (2026-04-10).** Architecture already validated by AI21. Novel contribution is Coconut-Ouroboros only. |

---

## Part 3 — Open Questions

| Question | Status |
|---|---|
| Nano SFT root cause (overfitting vs. label mask vs. LR)? | 🟡 LOW PRIORITY — one S10 diagnostic run, then move on |
| `forward_from_embeddings` for Jamba2-3B HF API? | 🔴 OPEN — use `model.model(inputs_embeds=..., attention_mask=...)`. Needs `jamba_coconut_finetune.py`. |
| MoE + Coconut compatibility? | ✅ RESOLVED — MoE routing is inside blocks, after residual injection; Coconut unaffected. |
| Temperature sampling in generation callback? | ✅ ADD — `temp=0.8, top_p=0.9` for both nano S10 and Jamba runs |

---

## Part 4 — File Registry

| File | Stage | Status | Notes |
|---|---|---|---|
| `baseline_trm_mamba.py` | 0 | ✅ COMPLETE | Needs `forward_with_hidden` + `forward_from_embeddings` for Stage 3 nano dry-run |
| `viability_gate.py` | 0 | ✅ COMPLETE | |
| `training_utils.py` | All | ✅ COMPLETE | Canonical utilities |
| `pretrain.py` | 1 | ✅ COMPLETE | Hub: ckpt-0021000 |
| `prepare_sft_dataset.py` | 2 | ✅ DONE | Dataset cached (sft-mix-v1) |
| `train_sft.py` | 2 | ✅ PATCHED (DDP v2) | |
| `train_sft_single_gpu.py` | 2 | 🟡 ONE MORE SESSION | Nano S10 diagnostic |
| `stage3_agent_prompt.md` | 3 | ✅ READY | Nano pipeline test |
| `recursive_finetune.py` | 3 | ⬜ NOT CREATED | Generate from `stage3_agent_prompt.md` |
| `jamba_coconut_finetune.py` | 3 | ⬜ NOT CREATED | **Primary research script.** Jamba2-3B + Coconut on TPU. |

---

## Part 5 — Nano Checkpoint State (Reference)

| Checkpoint | Step | val_ce | Notes |
|---|---|---|---|
| ckpt-0021000 | 21000 | 5.32 | Stage 1 final; SFT starting point |
| ckpt-0003500 | 3500 | ~5.6 | Old SFT full-mix |
| S9 final | ~800 | — | val_acc declining; train_ce=2.17; not pushed |

**Resume fingerprint:** `dataset_mix=cached`, `sft-mix-v1`. Must match or optimizer resets.

---

## Part 6 — Nano SFT Hyperparameters (Reference)

```
# S9 (last run) — caused degeneration
lr=3e-4, warmup=50, ema_decay=0.99, batch_size=2, grad_accum=8
max_seq_len=2048, dataset_mix=cached, ~3.5s/step; timed out at ~800/9840 steps

# S10 (proposed — one diagnostic session)
lr=1e-4, warmup=100, dropout=0.1, ema_decay=0.995, num_epochs=5
Add temperature sampling (temp=0.8, top_p=0.9) to generation callback
```

---

## Part 7 — Coconut-Ouroboros on Jamba2-3B (Primary Research Path)

### HuggingFace inputs_embeds adaptation
```python
# Replaces our custom forward_from_embeddings method:
outputs = model.model(
    inputs_embeds=patched_embeds,   # [B, T, D] — bypasses token embedding
    attention_mask=attn_mask,
    use_cache=False,
)
hidden = outputs.last_hidden_state  # [B, T, D]
logits = model.lm_head(hidden)      # [B, T, V]
```

### Curriculum

| Sub-stage | K | Resume | Platform |
|---|---|---|---|
| 3.1 | 1 | Jamba2-3B base | TRC TPU v3-8 |
| 3.2 | 4 | 3.1 checkpoint | TRC TPU v3-8 |
| 3.3 | 16 | 3.2 checkpoint | TRC TPU v3-8 |

Gate: answer val_ce ≤ baseline val_ce × 1.05

### Research comparison (publishable experiment)
Jamba2-3B (baseline, greedy) vs. Jamba2-3B + Coconut-Ouroboros (K=16). Hypothesis: K latent passes allow Mamba SSM state to accumulate richer question context before generation, measurably improving reasoning on math/code benchmarks.

---

## Part 8 — Checkpoint Format Reference

```python
# Nano Stage 2
{"stage": "sft", "step", "epoch", "samples_seen", "val_ce",
 "model_state_dict", "ema_backbone_state_dict", "optimizer", "scheduler",
 "scaler", "ema", "backbone_config", "sft_config", "data_fingerprint"}

# Nano Stage 3 adds
{"stage": "coconut", "n_latent", "lat_token_id", "vocab_size"}
```

---

## Part 9 — Compute Plan

| Stage | Model | Platform | Estimate |
|---|---|---|---|
| Nano S10 SFT + Coconut dry-run | nano 92M | Kaggle T4 | 1 session |
| Coconut K=1 | Jamba2-3B | TRC TPU v3-8 | ~4–6h |
| Coconut K=4 | Jamba2-3B | TRC TPU v3-8 | ~6–8h |
| Coconut K=16 | Jamba2-3B | TRC TPU v3-8 | ~8–12h |
| GRPO | Jamba2-3B | TRC TPU v3-8 | ~8–12h |
| Quantization | Jamba2-3B | Local / Jetson | ~2h |
