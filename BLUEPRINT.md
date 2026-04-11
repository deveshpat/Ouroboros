# Project Ouroboros — Master Blueprint

> **Thread-resume header. Read Part 0 first in any new session.**
> **DRY Guidelines:** Session details live in `terminal_log.md`. Blueprint holds decisions, architecture, status, and next actions only.

---

## Part 0 — Quick-Resume Context

### What this project is
Coconut-Ouroboros latent reasoning injection into a Transformer-Mamba hybrid (Jamba Reasoning 3B). The Mamba SSM recurrent state acts as compressed scratch-memory across K latent thought passes, bypassing token generation during reasoning. Core mechanism from Meta's Coconut (arXiv:2412.06769), extended with DGAC (Diversity-Gated Adaptive Coconut) — our novel anti-collapse halt gate.

### Strategic Status (Updated 2026-04-11)

| Stage | Name | Status |
|---|---|---|
| 0 | Architecture & Viability (nano) | ✅ COMPLETE |
| 1 | Pre-training (nano) | ✅ Pipeline test only; retired |
| 2 | SFT (nano) | 🔴 RETIRED |
| 3 | Coconut-Ouroboros + DGAC on Jamba Reasoning 3B | 🟡 NEXT |
| 4 | GRPO on Jamba Reasoning 3B | ⬜ NOT STARTED |
| 5 | Quantization / Edge Deploy | ⬜ NOT STARTED |

### TRC Status
✅ Accepted (email 2026-04-07). **Do not claim quota yet.** Claim after K=4 gate passes on Kaggle Dual T4.

### Immediate Next Actions

1. Run `prepare_coconut_dataset.py --output_dir data/coconut_v1` to build canonical dataset.
2. Feed `jamba_stage3_agent_prompt.md` to coding agent → generate `jamba_coconut_finetune.py`.
3. Smoke test on free Colab T4 (see checklist in agent prompt Part 14).
4. K=0→K_max curriculum on Kaggle Dual T4.
5. If K=4 gate passes: claim TRC, run K=16 + DGAC Phase 3.4.

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

**Why Reasoning 3B over Jamba2-3B:** Reasoning 3B already has explicit CoT traces that Coconut progressively replaces with latent passes. Jamba2-3B targets enterprise grounding (IFBench, RAG), has no reasoning traces, and would require re-teaching CoT from scratch. Wrong model for Coconut.

**Architecture note:** Reasoning 3B has a 13:1 Mamba:Attention ratio (not 1:7 as Jamba 1.5). This is MORE Mamba-heavy — better for Coconut since more SSM state is available to accumulate latent reasoning.

---

## Part 2 — Resolved Decisions

| Decision | Resolution |
|---|---|
| Primary research model | Jamba Reasoning 3B (ai21labs/AI21-Jamba-Reasoning-3B, Oct 2025) |
| Fine-tuning approach | QLoRA on GPU (bitsandbytes, 4-bit NF4); LoRA only on TPU — `--use_4bit` flag |
| LoRA target modules | q_proj, k_proj, v_proj, o_proj (attention) + in_proj, x_proj, dt_proj, out_proj (Mamba) |
| Coconut curriculum | Paper's correct mechanism: progressive step replacement (NOT fixed-K tokens at question end) |
| Stage advancement | Epoch-based (fixed epochs_per_stage) + best-accuracy checkpoint selection at end of stage |
| Dataset pipeline | Separate `prepare_coconut_dataset.py` → canonical JSONL; training script never touches HF datasets |
| Dataset sources | Same 5 sources as sft-mix-v1, re-processed from scratch with step segmentation |
| max_stage (K) | Set automatically from stats.json (median n_steps); override via `--max_stage` |
| DGAC halt gate | Phase 3.4 only, after K-stage curriculum complete; λ₁ annealed from 0 |
| Halt threshold | Exposed as `--halt_threshold` (default 0.5) — tunable at inference |
| TRC timing | Claim after K=4 gate passes, not before |
| Platform: dataset prep | Local or Kaggle CPU |
| Platform: smoke test | Free Colab T4, QLoRA, batch=1, max_samples=200 |
| Platform: K=0→K_max | Kaggle Dual T4 (2×16GB), QLoRA + DDP |
| Platform: K=16 + DGAC | TRC A100 80GB |
| Stage 1 gate | Bypassed (val_ce=5.32). Architecture proven. Hub: ckpt-0021000. |
| Stage 2 dataset | sft-mix-v1 (55k, WeirdRunner/Ouroboros) — SFT format; NOT reused for Coconut |

---

## Part 3 — Open Questions

| Question | Status |
|---|---|
| Does `inputs_embeds` bypass work cleanly with Jamba Reasoning 3B on HF? | 🟡 VERIFY during smoke test |
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
| `train_sft.py` | 2 | ✅ PATCHED (DDP v2) | Not being run further |
| `prepare_coconut_dataset.py` | 3 | ✅ WRITTEN | Run once; outputs data/coconut_v1/ |
| `jamba_stage3_agent_prompt.md` | 3 | ✅ READY | Feed to coding agent → jamba_coconut_finetune.py |
| `jamba_coconut_finetune.py` | 3 | ⬜ NOT CREATED | Generate from jamba_stage3_agent_prompt.md |

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

**Critical note:** Placing K fixed latent tokens at question end (our previous design) is equivalent to the paper's "pause token" baseline, which it found **inferior** to step replacement. This rewrite implements the correct mechanism.

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
    checkpoint-0001234/   ← epoch checkpoints (pruned to keep_checkpoints_per_stage)
    best/                  ← best-accuracy checkpoint; loaded before Stage 1 begins
      adapter_model/       ← PEFT LoRA weights
      halt_gate.pt         ← HaltGate state dict (Phase 3.4 only)
      training_state.pt    ← {stage_k, step, epoch, val_ce, val_acc, optimizer, scheduler}
  stage_1/
    best/
  ...
  stage_K/
    best/                  ← final Coconut model, used as resume_from for Phase 3.4
```

---

## Part 8 — Compute Plan

| Phase | Platform | Approach | Estimate |
|---|---|---|---|
| Dataset prep | CPU (local or Kaggle) | prepare_coconut_dataset.py | ~30–60 min |
| Smoke test | Free Colab T4 (15GB) | QLoRA, batch=1, max_stage=2, max_samples=200 | ~10 min |
| Stage 0→K | Kaggle Dual T4 (2×16GB) | QLoRA + DDP, epochs_per_stage=3 | ~4–8h per session |
| Phase 3.4 (DGAC) | TRC A100 80GB | QLoRA or LoRA, grad_ckpt | ~6–8h |
| Phase 4 (GRPO) | TRC A100 80GB | TBD | ~8–12h |
