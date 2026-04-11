# Project Ouroboros — Master Blueprint

> **Thread-resume header. Read Part 0 first in any new session.**
> **DRY Guidelines:** Session details live in `terminal_log.md`. Blueprint holds decisions, architecture, status, and next actions only.
> **Cleanup rules (for future DRY passes):** Summarise sessions into one-liners — never delete them. Preserve all architectural decisions and resolved bugs. Remove content only if it exists verbatim in the other file or has been superseded by a dated resolution.

---

## Part 0 — Quick-Resume Context

### What this project is
Coconut-Ouroboros latent reasoning injection into a Transformer-Mamba hybrid (Jamba2-3B). The Mamba SSM recurrent state acts as compressed scratch-memory across K latent thought passes, bypassing token generation during reasoning. Core mechanism from Meta's Coconut (arXiv:2412.06769), extended with DGAC (Diversity-Gated Adaptive Coconut) — our novel anti-collapse halt gate.

### Strategic Status (Updated 2026-04-10)

**PIVOT DECISION (2026-04-10):** Abandoned from-scratch nano training as the primary research vehicle. Novel contribution is exclusively Coconut-Ouroboros + DGAC on Jamba2-3B. Nano is fully retired — architecture proven, pipeline validated, no further GPU sessions.

| Stage | Name | Status |
|---|---|---|
| 0 | Architecture & Viability (nano) | ✅ COMPLETE |
| 1 | Pre-training (nano) | ✅ Pipeline test only; retired |
| 2 | SFT (nano) | 🔴 RETIRED — one diagnostic session planned for DDP v2 validation only if needed; blocked by pivot |
| 3 | Coconut-Ouroboros + DGAC on Jamba2-3B | 🟡 NEXT — generate script from agent prompt, smoke test on Colab T4, then Kaggle Dual T4 |
| 4 | GRPO on Jamba2-3B | ⬜ NOT STARTED |
| 5 | Quantization / Edge Deploy | ⬜ NOT STARTED |

### TRC Status
✅ Accepted (email 2026-04-07). **Do not claim quota yet.** Strategy: complete K=1→4 on Kaggle Dual T4 first. If K=4 gate passes, claim TRC for K=16 + GRPO (where the A100 headroom justifies the GCP setup cost).

### Immediate Next Actions

1. **Generate `jamba_coconut_finetune.py`** from `jamba_stage3_agent_prompt.md`.
2. **Smoke test on free Colab T4:** `--use_4bit --max_steps 20 --max_samples 50 --n_latent 1 --wandb_mode disabled`. Goal: no crashes, finite loss.
3. **K=1 training on Kaggle Dual T4:** QLoRA + DDP, seq_len=256, K=1.
4. **K=4 training** (from K=1 checkpoint).
5. **If K=4 gate passes:** claim TRC, run K=16 + DGAC Phase 3.4.

---

## Part 1 — Architecture

### Our TRM-Mamba nano (retired — reference only)
```
TokenEmbedding(vocab_size=151_680, d_model=512)
└─ 1 × TRMMambaGroup: TRMBlock + 7 × MambaLayer
FinalRMSNorm → LM Head (weight-tied)
92.5M params. Pretrained ~700M tokens. SFT partial. No further training planned.
```

### Jamba2-3B (primary research model)
```
HuggingFace: ai21labs/AI21-Jamba2-3B-Instruct   License: Apache 2.0
Architecture: 1:7 Attn:Mamba, Mamba 1, GQA, RMSNorm, MoE SwiGLU, no RoPE, 256K context
Post-training: mid-training 500B tokens → cold-start SFT → DPO
```
Use the **Instruct** variant as starting point. Research comparison: Jamba2-Instruct baseline (greedy) vs. Jamba2-Instruct + Coconut-Ouroboros (K=16 + DGAC). This is a cleaner scientific claim than training from the base model.

### Architecture Comparison (why Jamba2 validates our nano design)

| Feature | Our nano | Jamba2-3B | Notes |
|---|---|---|---|
| Attn:Mamba ratio | 1:7 | 1:7 | AI21 confirmed identical by ablation |
| Mamba version | Mamba 1 | Mamba 1 | AI21: Mamba2 worse in hybrid (Appendix C.2) |
| Normalization | RMSNorm | RMSNorm | |
| Attention | GQA | GQA | |
| FFN | Dense SwiGLU | MoE SwiGLU | Irrelevant for Coconut injection |

---

## Part 2 — Resolved Decisions

| Decision | Resolution |
|---|---|
| Primary research model | Jamba2-3B-Instruct (Apache 2.0, 3B, 1:7 hybrid) |
| Fine-tuning approach | QLoRA on GPU (bitsandbytes, 4-bit NF4); LoRA only on TPU (bitsandbytes not supported) — `--use_4bit` flag controls |
| LoRA target modules | q_proj, k_proj, v_proj, o_proj (attention) + in_proj, x_proj, dt_proj, out_proj (Mamba). Skip MoE FFN to limit LoRA params. |
| Nano stage 3 smoke test | Dropped. Nano API (custom `forward_from_embeddings`) does not transfer to Jamba HF API. No value in validating nano Coconut code. |
| Two stage3 prompts vs one | One Jamba-only prompt (`jamba_stage3_agent_prompt.md`). Old `stage3_agent_prompt.md` archived for reference. |
| Coconut forward efficiency | K-sequential full-prefix passes (fidelity to paper). K+1 passes per step. Gradient checkpointing required for K≥4. |
| Halt gate design | DGAC (see Part 7). Gate introduced only in Phase 3.4 after fixed-K curriculum establishes SSM state usage. |
| Halt gate inference threshold | Exposed as `--halt_threshold` (default 0.5). Allows trading quality vs. compute at inference time. |
| TRC timing | Claim after K=4 gate passes, not before. |
| Platform: smoke test | Free Colab T4, QLoRA, batch_size=1, seq_len=128 |
| Platform: K=1→4 | Kaggle Dual T4 (2×16GB), QLoRA + DDP |
| Platform: K=16 + DGAC | TRC A100 80GB (post K=4 gate) |
| mamba-ssm on TPU | Use `use_mamba_kernels=False` in model config for XLA compatibility; pure PyTorch fallback is slower but correct |
| Stage 1 gate | Bypassed (val_ce=5.32). Architecture proven. Hub: ckpt-0021000. |
| Stage 2 dataset | Full mix cached at WeirdRunner/Ouroboros (sft-mix-v1, 55,230 samples) — relevant only if nano SFT is ever resumed |

---

## Part 3 — Open Questions

| Question | Status |
|---|---|
| DGAC λ tuning: what are reasonable starting values for λ₁_max and λ₂? | 🔴 OPEN — set λ₁_max=0.01, λ₂=0.1, τ=0.9 as defaults; monitor ponder cost histogram |
| Does `inputs_embeds` bypass work cleanly with Jamba2-3B-Instruct on HF? | 🟡 VERIFY during smoke test — check `model.model(inputs_embeds=...).last_hidden_state` returns expected [B,T,D] |
| Jamba2 mamba-ssm TPU fallback: does XLA compilation succeed? | 🔴 OPEN — only relevant if Kaggle TPU is needed; not blocking for GPU path |
| DGAC Phase 3.4 gate: does mean halt step distribute across K≥2 after training? | 🔴 OPEN — primary validation metric for the research contribution |

---

## Part 4 — File Registry

| File | Stage | Status | Notes |
|---|---|---|---|
| `baseline_trm_mamba.py` | 0 | ✅ COMPLETE | Retired from active training; kept as architecture reference |
| `viability_gate.py` | 0 | ✅ COMPLETE | |
| `training_utils.py` | All | ✅ COMPLETE | Canonical utilities for nano; not used in Jamba script |
| `pretrain.py` | 1 | ✅ COMPLETE | Hub: ckpt-0021000 |
| `prepare_sft_dataset.py` | 2 | ✅ DONE | Dataset cached (sft-mix-v1) |
| `train_sft.py` | 2 | ✅ PATCHED (DDP v2) | Not being run further |
| `stage3_agent_prompt.md` | 3-nano | 🗄 ARCHIVED | Nano-specific; documents original Coconut design decisions |
| `jamba_stage3_agent_prompt.md` | 3-Jamba | ✅ READY | Primary research script spec |
| `jamba_coconut_finetune.py` | 3-Jamba | ⬜ NOT CREATED | Generate from `jamba_stage3_agent_prompt.md` |

---

## Part 5 — Jamba2-3B Coconut + DGAC Design

### HuggingFace inputs_embeds API
```python
# Core pattern for Coconut latent injection on Jamba2:
hidden_states = model.model(
    inputs_embeds=patched_embeds,   # [B, T, D] — bypasses token embedding
    attention_mask=attn_mask,
    use_cache=False,
).last_hidden_state                 # [B, T, D]
logits = model.lm_head(hidden_states)  # [B, T, V]

# To get embedding dimension D:
D = model.config.hidden_size

# For latent injection, get token embeddings:
embeds = model.model.embed_tokens(input_ids)  # [B, T, D]
```

### LoRA Configuration
```python
# GPU path (QLoRA):
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True, bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=True,
)
lora_config = LoraConfig(
    r=32, lora_alpha=64,
    target_modules=["q_proj","k_proj","v_proj","o_proj",
                    "in_proj","x_proj","dt_proj","out_proj"],
    lora_dropout=0.05, bias="none", task_type="CAUSAL_LM",
)
# TPU path (LoRA only, no quantization): same LoraConfig, no BitsAndBytesConfig
```

### DGAC: Diversity-Gated Adaptive Coconut

**The problem:** Standard halt gates collapse to K=1 early in training because the ponder cost reward dominates before the task loss has incentivized multiple-pass refinement.

**The mechanism:** Three-term loss with interacting regularizers:

```
L_total = L_ce  +  λ₁(t) · L_ponder  +  λ₂ · L_diversity

L_ce        = cross-entropy on answer tokens only
L_ponder    = ACT ponder cost: mean_batch(Σ_k remainder_k)
L_diversity = mean_batch( Σ_k relu(cos_sim(h_k, h_{k-1}) − τ) )
              τ = 0.9  (penalizes passes where hidden state barely changed)
```

**Anti-collapse invariant:**
- `L_ponder` alone → model halts at K=1
- `L_diversity` alone → model runs all K passes and makes each one change the representation
- Together: model halts early *only if* remaining passes would be no-ops (cos_sim > τ). Genuine refinement is rewarded; lazy no-op passes are penalized.

**λ₁ annealing schedule (critical for stability):**
```
Phase 3.4 warmup  (steps 0–200):   λ₁ = 0,       λ₂ = 0.1
Phase 3.4 ramp    (steps 200–500): λ₁ = 0 → 0.01 (linear), λ₂ = 0.1
Phase 3.4 main    (steps 500+):    λ₁ = 0.01,    λ₂ = 0.1
```

**HaltGate module:**
```python
class HaltGate(nn.Module):
    def __init__(self, d_model: int):
        super().__init__()
        self.gate = nn.Linear(2 * d_model, 1, bias=True)
        nn.init.zeros_(self.gate.weight)  # outputs 0.5 at init
        nn.init.zeros_(self.gate.bias)
    
    def forward(self, h_curr: Tensor, h_prev: Tensor) -> Tensor:
        # h_curr, h_prev: [B, D] — hidden state at question-end position
        return torch.sigmoid(self.gate(torch.cat([h_curr, h_prev], dim=-1)))
        # returns [B, 1] halt probability
```

**Training mode (soft, ACT-style):** Weighted combination of outputs across passes; ponder cost = ACT remainder sum.

**Inference mode (hard):** Halt when `gate(h_k, h_{k-1}) > halt_threshold` (default 0.5). Log histogram of actual halt steps — this is the primary research validation metric.

### Curriculum

| Phase | K | Gate | λ₁ | Resume from | Gate condition |
|---|---|---|---|---|---|
| 3.1 | 1 | None | 0 | Jamba2-Instruct | val_ce ≤ baseline × 1.05 |
| 3.2 | 4 | None | 0 | Phase 3.1 ckpt | val_ce ≤ baseline × 1.05 |
| 3.3 | 16 | None | 0 | Phase 3.2 ckpt | val_ce ≤ baseline × 1.05 |
| 3.4 | 16 | DGAC | annealed | Phase 3.3 ckpt | mean halt_step > 2.0 AND val_ce ≤ baseline × 1.05 |

---

## Part 6 — Checkpoint Format (Jamba + DGAC)

```python
# Saved as: output_dir/checkpoint-XXXXXXX/
#   - adapter_model/       ← LoRA adapter weights (PEFT save_pretrained)
#   - halt_gate.pt         ← HaltGate state dict
#   - training_state.pt    ← metadata below

state = {
    "stage":          "coconut_jamba",
    "phase":          "3.1" | "3.2" | "3.3" | "3.4",
    "step":           int,
    "n_latent":       int,
    "use_halt_gate":  bool,
    "val_ce":         float,
    "baseline_val_ce": float,      # Jamba2-Instruct baseline (K=0)
    "optimizer":      optimizer.state_dict(),
    "scheduler":      scheduler.state_dict(),
    "jamba_model_id": "ai21labs/AI21-Jamba2-3B-Instruct",
    "lora_config":    lora_config_dict,
}
```

---

## Part 7 — Compute Plan

| Phase | Model | Platform | Approach | Estimate |
|---|---|---|---|---|
| Smoke test | Jamba2-3B | Free Colab T4 (15GB) | QLoRA, batch=1, seq=128, K=1, steps=20 | 5 min |
| Phase 3.1 (K=1) | Jamba2-3B | Kaggle Dual T4 (2×16GB) | QLoRA + DDP, seq=256, grad_accum=8 | ~2–4h |
| Phase 3.2 (K=4) | Jamba2-3B | Kaggle Dual T4 | QLoRA + DDP + grad_ckpt | ~4–6h |
| Phase 3.3 (K=16) | Jamba2-3B | TRC A100 80GB | QLoRA or LoRA, grad_ckpt | ~8–12h |
| Phase 3.4 (DGAC) | Jamba2-3B | TRC A100 80GB | Same as 3.3 | ~6–8h |
| Phase 4 (GRPO) | Jamba2-3B | TRC A100 80GB | TBD | ~8–12h |
