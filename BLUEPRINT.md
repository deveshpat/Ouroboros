# Project Ouroboros — Master Blueprint

> **Thread-resume header.** In any new session: read Part 0 first, then jump to
> the current stage. This file is the single source of truth for architecture,
> status, bugs, and next actions. Append new findings — never rewrite history.

---

## Part 0 — Quick-Resume Context

### What this project is
Novel hybrid Transformer-Mamba language model ("TRM-Mamba", 1:7 ratio) pre-trained
from scratch on FineWeb-Edu, then fine-tuned for chain-of-thought reasoning. Target:
a ~1B parameter model with test-time compute scaling via recursive inference (SERF).
Zero budget — Kaggle Dual T4 + Google TRC application.

### Root failure on record
`checkpoint-950` (archived, do not use): randomly-initialized model trained directly
on SFT data. Output: comma-loops and "the the the". Three root causes:
1. No pre-training — SFT cannot teach language from random noise
2. EMA initialized to zeros — bias-correction diverged at step 1
3. Novel components stacked without per-component validation

### Current status

| Stage | Name | Status | Gate |
|---|---|---|---|
| 0 | Architecture & Viability | ✅ COMPLETE | All 4 gates passed |
| 1 | Pre-training | 🟡 IN PROGRESS on Kaggle Dual T4 — step ~1700 / 61,036 | val_ce < 3.0 + UWR > 0.05 |
| 2 | SFT | 🔴 BLOCKED — 3 bugs + multi-dataset not implemented | Fix bugs first |
| 3 | Recursive Inference | ⬜ NOT STARTED | Stage 2 val_ce < 1.5 |
| 4 | GRPO | ⬜ NOT STARTED | Stage 3 gate |
| 5 | Quantization | ⬜ NOT STARTED | Stage 4 or 3 gate |

### Immediate next action

**URGENT: Fix Bug 5 in pretrain.py before the Kaggle session resets.**
The Hub 401 at step 1000 means no checkpoint has been saved yet.
Apply the `save_checkpoint` patch from Part 7 and restart, OR fix the HF token.

```bash
# Option A: fix the HF token and restart from scratch (session still running)
# Set HF_TOKEN correctly in Kaggle secrets, then re-run.

# Option B: patch save_checkpoint to always finalize locally (see Bug 5 fix)
# then restart — this is the safer long-term fix regardless.
```

After Stage 1 completes (`val_ce < 3.0` AND `mean_uwr > 0.05`):
```bash
python train_sft.py \
  --preset nano \
  --resume_from runs/stage1/checkpoint-XXXXXXX \
  --ema_decay 0.995
```
BUT: fix Bugs 1–3 in `train_sft.py` first (see Part 7).

---

## Part 1 — Root Failure Record

| Cause | Effect | Fix |
|---|---|---|
| No pre-training | Model must learn language + reasoning simultaneously from 17k examples | Pre-train on 2B tokens first |
| EMA init = zeros | Bias-correction `ema/(1-decay^t)` blows up at t=1 | `pretrain.py`: EMA init from live weights |
| Too many new components | Cannot isolate failure | One change at a time, gate each |
| No non-recursive baseline | No reference to compare against | `baseline_trm_mamba.py` now the baseline |

**Loss=54.67 explained (not a bug):** `DeepSupervisedSERF` summed CE across 8 recursive
steps with gamma=0.8 → weight sum ≈4.57. Random-init CE = ln(151665) ≈ 11.93.
So 4.57 × 11.93 ≈ 54.5. Exactly right for random init.

---

## Part 2 — Architecture Specification

### Block structure

```
TokenEmbedding(vocab_size=151_680, d_model)
└─ n_groups × TRMMambaGroup:
   ├─ 1 × TRMBlock:
   │   ├─ RMSNorm + GQA Attention (RoPE) + residual
   │   └─ RMSNorm + SwiGLU MLP + residual
   └─ 7 × MambaLayer:          ← 1:7 ratio, hard-coded, DO NOT change
       └─ RMSNorm + mamba_ssm.Mamba + residual
FinalRMSNorm
LM Head (weight tied to TokenEmbedding)
```

### Configuration presets (verified against terminal_log.md)

| Preset | d_model | n_groups | n_heads | n_kv_heads | mlp_hidden | Params |
|---|---|---|---|---|---|---|
| nano | 512 | 1 | 8 | 4 | 1408 | 92.5M |
| small | 1024 | 2 | 16 | 8 | 2816 | ~270M |
| medium | 2048 | 2 | 16 | 8 | 5632 | ~760M |

**Nano checksum (verified Stage 0):**
```
parameters          = 92,477,440      (tied-embedding dedup working)
mlp_hidden          = 1408            = ceil((8/3×512)/64)×64
total_residual_layers = 9             = 1×(2+7)
initial_loss        = 11.9904         (theoretical 11.9295, within 0.5%)
grad_norm_total     = 6.4242
```

**Fixed config values (all presets):**
```python
vocab_size            = 151_680       # ceil(151_665/128)×128 — Qwen2.5-0.5B padded
rope_theta            = 1_000_000.0   # Qwen2.5 convention
mamba_d_state         = 16
mamba_d_conv          = 4
mamba_expand          = 2
max_seq_len           = 2048
dropout               = 0.0           # pre-training; 0.1 for SFT if overfitting
rms_norm_eps          = 1e-5
tie_embeddings        = True
mlp_hidden            = ceil((8/3 × d_model) / 64) × 64
total_residual_layers = n_groups × (2 + 7)
```

### Key implementation notes (from verified baseline_trm_mamba.py)

**RoPE** — Non-interleaved. Cache rebuilt lazily on device/dtype change. Theta=1e6.

**GQA** — `repeat_interleave` applied after RoPE on compact KV tensors, then expanded.
`n_repeats = n_heads / n_kv_heads`. Verified correct.

**Padding + causal attention** — SDPA raises on `is_causal=True` + custom mask. When
`attention_mask` is provided: build combined causal+padding additive bias, call with
`is_causal=False`. Verified in Stage 0 smoke test.

**Mamba padding** — Padded positions zeroed: `h = h * mask` before AND after Mamba
forward. Mask convention: `True` = valid token.

**Residual scaling (GPT-2 style)** — Residual writers (`o_proj`, `mlp.down_proj`,
`mamba.out_proj`) scaled by `0.02 / sqrt(total_residual_layers)` after all base inits.
Two-pass to avoid double-initializing tied weights.

---

## Part 3 — Resolved Decisions

| Decision | Resolution |
|---|---|
| Tokenizer | Qwen2.5-0.5B; vocab_size=151_680 everywhere, no exceptions |
| Mamba version | `mamba_ssm.Mamba` (original). Mamba2 deferred to Stage 3+ |
| Stage 1 dataset | HuggingFaceFW/fineweb-edu sample-10BT, streamed |
| Stage 2 dataset | Bespoke-Stratos-17k (40%) + MetaMathQA (30%) + OpenHermes-2.5 (30%) |
| Hub repo | WeirdRunner/Ouroboros (private) |
| Pre-training strategy | Train from scratch. Apply to Google TRC immediately (async) |
| Recursion timing | Post-SFT only. Never train recursion from random init |
| GRPO implementation | TRL GRPOTrainer. Never reimplement from scratch |

---

## Part 4 — File Registry

### Active files

| File | Stage | Status | Notes |
|---|---|---|---|
| `baseline_trm_mamba.py` | 0 | ✅ COMPLETE | All smoke checks passed |
| `viability_gate.py` | 0 | ✅ COMPLETE | All 4 gates passed |
| `pretrain.py` | 1 | 🟡 SCRIPT VERIFIED, not run on Kaggle | See Stage 1 |
| `train_sft.py` | 2 | 🔴 2 BUGS — do not run | See Part 7 |
| `BLUEPRINT.md` | — | Living document | This file |
| `terminal_log.md` | — | Verified terminal outputs | Append only |

### Renamed files (reflect manually)

| Was | Now |
|---|---|
| `Ouroboros_Blueprint_v3.md` | `BLUEPRINT.md` |
| `ouroboros_blueprint_v2.md` | archived / delete |
| `phase1_viability_gate.py` | `viability_gate.py` |
| `train_sft_phase2.py` | `train_sft.py` |

### Deprecated (do not use)

| File | Why |
|---|---|
| `recursive_trm_mamba.py` | Produced checkpoint-950 failure |
| `train_sft_deep_supervision.py` | Deep supervision on random init; obsolete |
| `inference_trm_mamba.py` | Tied to old recursive model |
| `architecture.py` | Superseded by `baseline_trm_mamba.py` |

---

## Part 5 — Stage Definitions

---

### Stage 0 — Architecture & Viability Gate ✅ COMPLETE

**Files:** `baseline_trm_mamba.py`, `viability_gate.py`

**Completed checks:**
- [x] Model instantiates on CPU and CUDA without error
- [x] Initial loss = 11.9904 (within ±0.5 nats of theoretical 11.9295)
- [x] `loss.backward()` — no NaN gradients, no missing gradients
- [x] `param_count = 92,477,440` — tied-embedding dedup verified
- [x] Padding mask path (`is_causal=False`) exercised and correct
- [x] G1: CE converged 12.04 → 2.00 in 300 steps (threshold < 3.5) ✅
- [x] G2: Mean UWR = 0.573 at step 300 (threshold > 0.10) ✅
- [x] G3: Max grad_norm = 4.03 in final 100 steps (threshold < 10.0) ✅
- [x] G4: VRAM delta = 0.000 GB (threshold < 1.0 GB) ✅

---

### Stage 1 — Pre-training 🟡 IN PROGRESS

**Gate:** Stage 0 all gates passed ✅

**Script:** `pretrain.py`
**Hardware:** Kaggle Dual T4, DDP world_size=2
**Status:** Running — step ~1700 / 61,036 (~2.8% complete). See terminal_log.md for full output.

**⚠ CRITICAL — Bug 5 active:** Hub 401 at step 1000 caused no checkpoint to be saved.
Apply the `save_checkpoint` fix from Part 7 before the session resets.

**Dry-run checklist (completed):**
- [x] No import errors; FakeMamba smoke test passes (epoch_offset=7, loss decreasing)
- [x] Initial loss ≈ 11.93 (actual 11.98); decreases over 20 steps
- [x] Checkpoint round-trip verified (step=20 restored correctly)
- [x] `ema_backbone_state_dict` contains `lm_head.weight` alias
- [x] Val buffer builds (2M tokens, 1,887 docs, document-disjoint from training)
- [x] Epoch offset=228 (non-zero for epoch 0) ✓

**Key hyperparameters (nano, 2B tokens):**
```
batch_size=8, grad_accum=4, chunk_size=1024 → 32,768 tokens/step
total_steps = 61,036
lr=6e-4, warmup=200 steps, cosine decay to 6e-5
weight_decay=0.1, max_grad_norm=1.0, ema_decay=0.995
save_every=1000, val_every=500, gen_every=500
DDP throughput: ~5,800 tok/s. Full run ETA: ~9.4h
```

**Live observations (step 1700):**
- Loss 11.98 → 4.85 (smoothed) — healthy trajectory
- Val CE 6.38 → 5.68 (confirming generalization)
- Mean UWR 0.38–0.42 — already above 0.05 success threshold
- VRAM flat at 2.035 GB — no graph retention
- Two isolated spikes (steps 1148, 1611); spike rate 0.12%
- Code prompt loops ("1000...0") expected — FineWeb-Edu has no code

**Success criteria:**
- [ ] `val_ce < 3.0` AND `mean_uwr > 0.05` (script prints banner when met)
- [ ] No loss spikes > smoothed+0.5 for > 5% of steps
- [ ] Coherent text completions in gen callback

**Next action after Stage 1:**
```bash
python train_sft.py \
  --preset nano \
  --resume_from runs/stage1/checkpoint-XXXXXXX \
  --ema_decay 0.995
```
Fix Bugs 1–3 in `train_sft.py` first (see Part 7).

---

### Stage 2 — SFT 🔴 BLOCKED

**Gate:** Stage 1 `val_ce < 3.0` AND `mean_uwr > 0.05`

**Script:** `train_sft.py` — two bugs and multi-dataset not implemented. See Part 7.

**Agent prompt for fixing train_sft.py:** See Appendix A.

**Target format:**
```
User: {question}
Assistant: <think>
{reasoning_chain}
</think>
{final_answer}
```

**Hyperparameters:**
```
lr=3e-5 to 1e-4 (10× lower than pre-training)
warmup=100 steps, cosine decay, num_epochs=3 max
batch_size=2, grad_accum=8, max_seq_len=512
ema_decay=0.995
```

**Success criteria:**
- [ ] `val_ce < 1.5`
- [ ] Model generates `<think>` blocks before answering
- [ ] Semantically coherent answers on GEN_PROMPTS

---

### Stage 3 — Incremental Recursion ⬜ NOT STARTED

**Gate:** Stage 2 `val_ce < 1.5`

**Strategy:** Fine-tune the Stage 2 checkpoint into recursion. NEVER train recursion
from a randomly initialized model — this is what killed checkpoint-950.

```
Step 3.1: Fine-tune Stage 2 ckpt with n_loops=2
          Gate: CE returns within 5% of Stage 2 baseline CE
Step 3.2: n_loops=4. Same gate.
Step 3.3: n_loops=8 if budget permits.
```

**Critical EMA warm-start (fixes the original bug):**
```python
# CORRECT — warm-start from current hidden state
ema_state = base_hidden_state.detach().clone()

# WRONG — caused checkpoint-950 divergence
ema_state = torch.zeros_like(base_hidden_state)
```

**Recursive forward (inference-time wrapper, no architecture change):**
```python
@torch.no_grad()
def recursive_generate(model, input_ids, n_loops=4, ema_decay=0.9):
    logits, hidden = model(input_ids, return_hidden=True)
    ema = hidden.detach().clone()                     # warm start
    for step in range(1, n_loops):
        correction = 1.0 - ema_decay ** step
        loop_input = ema / correction                 # bias-corrected
        logits, hidden = model(loop_input_as_embeddings, ...)
        ema = ema_decay * ema + (1 - ema_decay) * hidden
    return logits
```

**Agent prompt:** See Appendix B.

---

### Stage 4 — GRPO ⬜ NOT STARTED

**Gate:** Stage 3 quality does not degrade vs Stage 2 (CE within 5%).

**Implementation:** TRL `GRPOTrainer` — DO NOT reimplement GRPO from scratch.

**Reward functions (implement in this order):**
1. Format: +0.1 if `<think>...</think>` tags correctly open/close
2. Correctness: +1.0 if answer verified (SymPy for math, E2B sandbox for code)
3. Length penalty (optional): mild negative for traces > 2× median length

**Compute (T4, 16GB):**
- 1.5B model in 4-bit (bitsandbytes): ~3GB
- G=4 GRPO rollouts: ~6GB additional
- Use unsloth + gradient checkpointing. Drop to G=2 if OOM.

---

### Stage 5 — Quantization ⬜ NOT STARTED

**Gate:** Stage 4 complete (or Stage 3 if skipping GRPO).

**Method:** Quamba (Hadamard + DLS). Standard 4-bit breaks Mamba models due to
activation outliers in the linear recurrence. Quamba (2024) resolves this.

**Target:** INT4/INT8 artifact for Jetson or similar edge device.

---

## Part 6 — Checkpoint Format (canonical across all stages)

Every checkpoint directory must contain exactly:
```
checkpoint-NNNNNNN/
  training_state.pt             ← full training state for resume
  resolved_backbone_config.json ← BaselineConfig as JSON for inference
```

**training_state.pt required keys:**
```python
{
    "step":                    int,
    "epoch":                   int,
    "chunks_in_epoch":         int,   # Stage 1 only — for data-position resume
    "tokens_processed":        int,   # Stage 1 only
    "model_state_dict":        dict,
    "ema_backbone_state_dict": dict,  # EMA shadow + lm_head.weight alias
    "optimizer":               dict,
    "scheduler":               dict,
    "scaler":                  dict or None,
    "ema":                     dict,
    "backbone_config":         dict,  # asdict(BaselineConfig)
    "val_ce":                  float or None,
}
```

**lm_head.weight alias rule:** When `tie_embeddings=True`, `named_parameters()` yields
one tensor for both embedding and lm_head. `ema_backbone_state_dict` MUST explicitly
contain `"lm_head.weight"` as an alias to `"token_embedding.weight"`. Verified in
pretrain.py's `save_checkpoint`.

**Hub push protocol (corrected — see Bug 5):**
1. Write fully to `checkpoint-NNNNNNN.tmp`
2. Rename `.tmp` → `checkpoint-NNNNNNN` immediately (local finalize — always happens)
3. Prune old local checkpoints
4. Attempt Hub push from `checkpoint-NNNNNNN` — fire-and-forget, warn on failure, never block

---

## Part 7 — Known Bugs and Issues

### Bug 1 — HIGH: `train_sft.py::compute_val_ce` corrupts live weights

**Symptom:** After the first `val_every` step, the model trains on EMA weights instead
of live weights. Behaviour looks normal (loss still decreases) but you are updating
EMA weights, not live weights. Silent corruption.

**Root cause:** `compute_val_ce` calls
`param.data.copy_(ema.shadow[name]...)` but never restores live weights afterward.

**Fix — drop-in replacement for compute_val_ce:**
```python
@torch.no_grad()
def compute_val_ce(model, ema, val_samples, pad_id, device, dtype, batch_size, vocab_size):
    # Save live weights before swapping in EMA
    live_backup = {}
    for name, param in model.named_parameters():
        if name in ema.shadow:
            live_backup[name] = param.data.clone()
            param.data.copy_(ema.shadow[name].to(dtype=param.data.dtype))

    model.eval()
    total_loss, total_tokens = 0.0, 0

    for start in range(0, len(val_samples), batch_size):
        batch_samples = val_samples[start : start + batch_size]
        batch     = collate(batch_samples, pad_id)
        input_ids = batch["input_ids"].to(device)
        attn_mask = batch["attention_mask"].to(device)
        labels    = batch["labels"].to(device)
        with torch.autocast(device_type="cuda", dtype=dtype):
            logits = model(input_ids, attention_mask=attn_mask)
        shift_logits = logits[:, :-1, :].contiguous().view(-1, vocab_size).float()
        shift_labels = labels[:, 1:].contiguous().view(-1)
        valid = (shift_labels != -100)
        if valid.any():
            total_loss   += F.cross_entropy(shift_logits[valid], shift_labels[valid], reduction="sum").item()
            total_tokens += int(valid.sum().item())

    val_ce = total_loss / max(total_tokens, 1)
    model.train()

    # Restore live weights
    for name, param in model.named_parameters():
        if name in live_backup:
            param.data.copy_(live_backup[name])

    return val_ce
```

**Status:** 🔴 NOT FIXED. Do not run Stage 2 without applying this fix.

---

### Bug 2 — MEDIUM: `train_sft.py::load_latest_checkpoint` fails on Stage 1 path

**Symptom:** `--resume_from runs/stage1/checkpoint-0001000` fails silently — the
function globs for `checkpoint-*` inside the checkpoint directory (finds nothing)
and returns step=0. Stage 2 starts from scratch ignoring the Stage 1 weights.

**Root cause:** `output_dir.glob("checkpoint-*")` only works if `output_dir` is the
PARENT of checkpoint directories, not the checkpoint directory itself.

**Fix — replace load_latest_checkpoint in train_sft.py:**
```python
def load_latest_checkpoint(output_dir, model, ema, optimizer, scheduler, scaler, device):
    path = Path(output_dir)

    # Case 1: path points directly at a checkpoint directory
    if (path / "training_state.pt").exists():
        state_path = path / "training_state.pt"
    else:
        # Case 2: path is parent; find the latest checkpoint-* subdirectory
        candidates = sorted(
            [p for p in path.glob("checkpoint-*") if (p / "training_state.pt").exists()],
            key=lambda p: int(p.name.split("-")[-1]),
        )
        if not candidates:
            print("  [resume] No checkpoint found — starting from scratch.")
            return 0
        state_path = candidates[-1] / "training_state.pt"

    print(f"  [resume] loading {state_path.parent.name}")
    state = torch.load(state_path, map_location=device)
    model.load_state_dict(state["model_state_dict"])
    optimizer.load_state_dict(state["optimizer"])
    scheduler.load_state_dict(state["scheduler"])
    if scaler and state.get("scaler"):
        scaler.load_state_dict(state["scaler"])
    if state.get("ema"):
        ema.load_state_dict(state["ema"])
    step = int(state.get("step", 0))
    print(f"  [resume] step={step}  val_ce={state.get('val_ce')}")
    return step
```

**Status:** 🔴 NOT FIXED. Apply before first Stage 2 run.

---

### Issue 3 — LOW: `train_sft.py` header still says "Phase 2 SFT"

**Fix:** Change line `print("  Phase 2 SFT — Project Ouroboros")` to
`print("  Stage 2 SFT — Project Ouroboros")`.

**Status:** 🟡 COSMETIC — fix alongside Bug 1 and Bug 2.

---

### Bug 3 — HIGH: `train_sft.py::collate` applies loss to all tokens including the prompt

**Symptom:** Model is trained to predict "User: {question}\n\nAssistant: <think>\n"
tokens as well as the answer. This causes the model to waste capacity predicting
its own prompt and biases it toward parroting the question format. Observed as val_ce
not falling below ~1.5 even with many epochs.

**Root cause:** `collate` copies all token IDs to `labels` unconditionally. Standard
SFT practice masks the prompt tokens with -100 so only the assistant response is
supervised.

**Fix — update `collate` to accept a prompt_length per sample, or split at
the "Assistant:" boundary at tokenize time:**
```python
# In load_and_tokenize, record where the assistant response starts:
prefix = f"User: {q}\n\nAssistant: <think>\n"
prefix_ids = tokenizer.encode(prefix, add_special_tokens=False)
prompt_len  = len(prefix_ids)
full_ids    = tokenizer.encode(text, add_special_tokens=False)
samples.append({
    "input_ids":   torch.tensor(full_ids[:max_seq_len], dtype=torch.long),
    "prompt_len":  prompt_len,   # NEW
})

# In collate, mask prompt tokens in labels:
for i, s in enumerate(samples):
    ids = s["input_ids"]
    T   = ids.size(0)
    input_ids[i, :T] = ids
    pl = min(s.get("prompt_len", 0), T)
    labels[i, pl:T]  = ids[pl:]   # only supervise the assistant response
    mask[i, :T]      = True
```

**Status:** 🔴 NOT FIXED. Add to Appendix A Change list as Change 0 (highest priority).

---

### Issue 4 — MEDIUM: Stage 2 multi-dataset mixing not implemented

**Symptom:** `train_sft.py` only loads `Bespoke-Stratos-17k`. The resolved decision
is a 40/30/30 mix with MetaMathQA and OpenHermes-2.5.

**Status:** 🟡 PENDING. See Appendix A for the full agent prompt that implements this.

---

### Bug 5 — CRITICAL: `pretrain.py::save_checkpoint` abandons local checkpoint on Hub failure

**Symptom:** When `push_to_hub=True` and the Hub upload fails (e.g. 401 Unauthorized),
`save_checkpoint` returns `None` without renaming the `.tmp` directory to its final
name. The checkpoint state is written to disk as `.tmp` but is never finalized.
If the session resets, all progress since the last successful checkpoint is lost.

**Confirmed at:** Step 1000 of the live Kaggle run (401 Unauthorized — HF token not set).
No checkpoint exists on disk. The run is at step ~1700 with no recovery point.

**Root cause:** Lines 682–686 in `pretrain.py`:
```python
if cfg.push_to_hub and hf_token:
    uploaded = sync_checkpoint_to_hub(tmp_dir, cfg.hf_repo_id, hf_token)
    if not uploaded:
        print(f"  [warn] step {step}: Hub sync failed; checkpoint not finalized.")
        return None   # ← exits before tmp_dir.replace(final_dir)
```

**Fix — always finalize locally, then attempt Hub push as fire-and-forget:**
```python
# In save_checkpoint, reorder the Hub sync AFTER local finalization:

# Step 1: always finalize locally
if final_dir.exists():
    shutil.rmtree(final_dir, ignore_errors=True)
tmp_dir.replace(final_dir)
print(f"  [ckpt] saved  -> {final_dir}")

# Step 2: prune old local checkpoints
retain = max(cfg.keep_last, 1)
existing = sorted([
    p for p in output_dir.iterdir()
    if p.is_dir() and p.name.startswith("checkpoint-") and not p.name.endswith(".tmp")
], key=lambda p: checkpoint_step_from_name(p.name))
for old in existing[:-retain]:
    shutil.rmtree(old, ignore_errors=True)
    print(f"  [ckpt] pruned -> {old.name}")

# Step 3: attempt Hub push (warn on failure, never block)
if cfg.push_to_hub and hf_token:
    uploaded = sync_checkpoint_to_hub(final_dir, cfg.hf_repo_id, hf_token)
    if not uploaded:
        print(f"  [warn] step {step}: Hub sync failed; local checkpoint retained at {final_dir}")

return final_dir
```

Also update `Part 6 — Hub push protocol` to reflect the corrected order:
1. Write fully to `checkpoint-NNNNNNN.tmp`
2. Rename `.tmp` → `checkpoint-NNNNNNN` (local finalize, always happens)
3. Attempt Hub push from `checkpoint-NNNNNNN` (fire-and-forget)

**Agent prompt:** `AGENT_PROMPT_pretrain_checkpoint_fix.md` — self-contained, feed directly to a coding agent. Makes 3 surgical changes:
  - Change 1: `save_checkpoint` — local-first, Hub fire-and-forget
  - Change 2: `load_latest_checkpoint` + 4 helper functions — explicit path / local / Hub fallback
  - Change 3: `cleanup_temporary_checkpoints` call at startup

**Status:** 🔴 NOT FIXED. Kaggle file persistence is now enabled (files survive interrupts),
so local checkpoints are safe once written. Fix this before the next `save_every` fires.

---

## Part 8 — Compute Plan

| Stage | Platform | Estimate | Notes |
|---|---|---|---|
| 0 | Colab free / local | ~15 min | ✅ Done |
| 1 | Kaggle T4 | ~18.5h (nano, 2B tokens) | Apply TRC immediately (async) |
| TRC application | — | 5 min | sites.research.google/trc |
| 2 | Kaggle T4 or TRC | ~2–4h | 3 epochs × ~50k samples |
| 3 | TRC preferred | ~4–8h | Incremental, gated per n_loops |
| 4 | TRC + unsloth | ~8–12h | GRPO G=4 rollouts |
| 5 | Local / Jetson | ~2h | Post-training quantization |

---

## Part 9 — References

- **Mamba** (Gu & Dao, 2023) — linear-time SSM design rationale
- **Jamba** (AI21 Labs, 2024) — 1:7 TRM-Mamba at 52B; our 1B is a miniaturized Jamba
- **GPT-2** (Radford et al., 2019) — residual scaling formula source
- **RoPE** (Su et al., 2021) — reference in LLaMA 2 codebase
- **SwiGLU** (Shazeer, 2020) — `ceil((8/3 × d_model) / 64) × 64` hidden_dim
- **GQA** (Ainslie et al., 2023) — grouped query attention
- **FineWeb-Edu** (HuggingFaceTB, 2024) — https://huggingface.co/datasets/HuggingFaceFW/fineweb-edu
- **DeepSeek-R1** (2025) — GRPO reward functions and implementation (Appendix)
- **Quamba** (2024) — post-training quantization for Mamba models
- **build-nanoGPT** (Karpathy) — https://github.com/karpathy/build-nanogpt
- **TRL GRPOTrainer** — https://huggingface.co/docs/trl

---

## Appendix A — Stage 2 Agent Prompt

> Self-contained. Feed this entire appendix to a coding agent to produce a corrected
> and feature-complete `train_sft.py`. The agent should edit the existing file, not
> rewrite it from scratch.

---

### Task

Fix two bugs and implement multi-dataset support in `train_sft.py`.
Do not rewrite the file. Make surgical edits only.

### Context

`train_sft.py` trains `BaselineTRMMamba` (defined in `baseline_trm_mamba.py`) on
instruction-following data using standard next-token cross-entropy loss.
It runs AFTER Stage 1 pre-training, loading a checkpoint produced by `pretrain.py`.

### Required changes

#### Change 1 — Fix compute_val_ce (BUG, HIGH)

**Problem:** The function swaps EMA weights into the model for evaluation but never
restores live weights. Every training step after the first val run uses EMA weights
instead of live weights.

**Location:** function `compute_val_ce` (~line 466)

**Action:** Add a `live_backup` save before the EMA swap and restore it after.

Replace the entire function body with this pattern:
```python
@torch.no_grad()
def compute_val_ce(model, ema, val_samples, pad_id, device, dtype, batch_size, vocab_size):
    live_backup = {}
    for name, param in model.named_parameters():
        if name in ema.shadow:
            live_backup[name] = param.data.clone()
            param.data.copy_(ema.shadow[name].to(dtype=param.data.dtype))

    model.eval()
    total_loss, total_tokens = 0.0, 0

    for start in range(0, len(val_samples), batch_size):
        batch_samples = val_samples[start : start + batch_size]
        batch     = collate(batch_samples, pad_id)
        input_ids = batch["input_ids"].to(device)
        attn_mask = batch["attention_mask"].to(device)
        labels    = batch["labels"].to(device)
        with torch.autocast(device_type="cuda", dtype=dtype):
            logits = model(input_ids, attention_mask=attn_mask)
        shift_logits = logits[:, :-1, :].contiguous().view(-1, vocab_size).float()
        shift_labels = labels[:, 1:].contiguous().view(-1)
        valid = (shift_labels != -100)
        if valid.any():
            total_loss   += F.cross_entropy(shift_logits[valid], shift_labels[valid], reduction="sum").item()
            total_tokens += int(valid.sum().item())

    val_ce = total_loss / max(total_tokens, 1)
    model.train()
    for name, param in model.named_parameters():
        if name in live_backup:
            param.data.copy_(live_backup[name])
    return val_ce
```

#### Change 2 — Fix load_latest_checkpoint (BUG, MEDIUM)

**Problem:** `output_dir.glob("checkpoint-*")` fails when the path IS the checkpoint
directory (e.g. `runs/stage1/checkpoint-0001000`). Returns step=0 silently.

**Location:** function `load_latest_checkpoint` (~line 577)

**Action:** Replace the entire function with this implementation that handles both
a direct checkpoint path and a parent directory:

```python
def load_latest_checkpoint(output_dir, model, ema, optimizer, scheduler, scaler, device):
    path = Path(output_dir)

    # Case 1: path points directly at a checkpoint directory
    direct = path / "training_state.pt"
    if direct.exists():
        state_path = direct
    else:
        # Case 2: path is parent; find latest checkpoint-* subdirectory
        candidates = sorted(
            [p for p in path.glob("checkpoint-*") if (p / "training_state.pt").exists()],
            key=lambda p: int(p.name.split("-")[-1]),
        )
        if not candidates:
            print("  [resume] No checkpoint found — starting from scratch.")
            return 0
        state_path = candidates[-1] / "training_state.pt"

    print(f"  [resume] loading {state_path.parent.name}")
    state = torch.load(state_path, map_location=device)
    model.load_state_dict(state["model_state_dict"])
    optimizer.load_state_dict(state["optimizer"])
    scheduler.load_state_dict(state["scheduler"])
    if scaler and state.get("scaler"):
        scaler.load_state_dict(state["scaler"])
    if state.get("ema"):
        ema.load_state_dict(state["ema"])
    step = int(state.get("step", 0))
    val_ce = state.get("val_ce")
    print(f"  [resume] step={step}  val_ce={val_ce}")
    return step
```

#### Change 3 — Add multi-dataset support (FEATURE)

**Goal:** Support loading Bespoke-Stratos-17k (40%), MetaMathQA (30%), and
OpenHermes-2.5 (30%) in a single interleaved training set.

**Add to CLI parser** (in `parse_args`):
```python
p.add_argument("--dataset_mix", default="stratos",
               choices=["stratos", "full"],
               help="'stratos' = Bespoke-Stratos-17k only. "
                    "'full' = 40/30/30 mix with MetaMathQA + OpenHermes-2.5.")
```

**Add dataset loaders after the existing `load_and_tokenize` function:**

```python
def _extract_metamath(example: Dict) -> Tuple[str, str, str]:
    """Extract (question, reasoning, answer) from MetaMathQA."""
    q = str(example.get("original_question") or example.get("query") or "").strip()
    a = str(example.get("response") or example.get("output") or "").strip()
    return q, "", a   # no reasoning chain in MetaMathQA


def _extract_openhermes(example: Dict) -> Tuple[str, str, str]:
    """Extract (question, reasoning, answer) from OpenHermes-2.5 conversations."""
    question = answer = ""
    for turn in (example.get("conversations") or []):
        role = str(turn.get("from", "")).lower()
        val  = str(turn.get("value", "")).strip()
        if role == "human" and not question:
            question = val
        elif role == "gpt" and not answer:
            answer = val
    return question, "", answer


def load_mixed_dataset(
    tokenizer,
    max_samples_per_source: Optional[int],
    max_seq_len: int,
) -> List[Dict]:
    """Load and interleave three datasets at 40/30/30 ratio."""
    import math as _math

    sources = [
        ("bespokelabs/Bespoke-Stratos-17k", "train",  _extract_bespoke,     0.40),
        ("meta-math/MetaMathQA",            "train",  _extract_metamath,    0.30),
        ("teknium/OpenHermes-2.5",          "train",  _extract_openhermes,  0.30),
    ]
    eos = tokenizer.eos_token or "<|endoftext|>"
    all_samples: List[Dict] = []

    for ds_name, split, extractor, _ in sources:
        print(f"  Loading {ds_name} ...")
        try:
            raw = load_dataset(ds_name, split=split)
        except Exception as e:
            print(f"  [warn] Could not load {ds_name}: {e} — skipping.")
            continue
        if max_samples_per_source is not None:
            raw = raw.select(range(min(max_samples_per_source, len(raw))))

        kept = 0
        for ex in tqdm(raw, desc=f"  {ds_name.split('/')[-1]}", leave=False):
            q, r, a = extractor(ex)
            if not q or not a:
                continue
            text = _format_training_text(q, r, a, eos)
            ids  = tokenizer.encode(text, add_special_tokens=False)
            if len(ids) < 4:
                continue
            all_samples.append({
                "input_ids": torch.tensor(ids[:max_seq_len], dtype=torch.long),
                "source": ds_name,
            })
            kept += 1
        print(f"    {kept} samples kept.")

    # Shuffle to interleave sources
    import random
    random.shuffle(all_samples)
    print(f"  Total mixed samples: {len(all_samples)}")
    return all_samples
```

**In `main()`, replace the dataset loading block:**
```python
if args.dataset_mix == "full":
    per_source = args.max_samples  # None = use all
    all_samples = load_mixed_dataset(tokenizer, per_source, args.max_seq_len)
else:
    all_samples = load_and_tokenize(
        args.dataset_name, tokenizer, args.max_samples, args.max_seq_len
    )
```

#### Change 4 — Update header string (COSMETIC)

Replace `"  Phase 2 SFT — Project Ouroboros"` with `"  Stage 2 SFT — Project Ouroboros"`.

### Verification checklist after editing

Run dry-run:
```bash
python train_sft.py \
  --preset nano \
  --max_samples 300 \
  --max_steps 100 \
  --val_every 50 \
  --gen_every 50 \
  --wandb_mode disabled
```

- [ ] No import errors
- [ ] `val_ce` column appears in training log at step 50 and 100
- [ ] After val eval, training CE continues to decrease (live weights intact)
- [ ] `--resume_from runs/stage1/checkpoint-XXXXXXX` loads step correctly
- [ ] `--dataset_mix full` loads all three sources without error

---

## Appendix B — Stage 3 Agent Prompt

> Self-contained. Feed this entire appendix to a coding agent to produce
> `recursive_finetune.py`. This must only be run after Stage 2 completes.

---

### Task

Create `recursive_finetune.py`: a fine-tuning script that takes a Stage 2 checkpoint
and incrementally adds recursive inference (SERF mechanism), gated by CE.

### Architecture context

`BaselineTRMMamba` (in `baseline_trm_mamba.py`) is a standard non-recursive model.
The recursive wrapper loops the hidden states back through the model multiple times.
**This is an inference-time computation extension, not an architectural change.**

The model must be pre-trained and SFT'd before recursion is introduced. Recursion
on a random-init or undertrained model produces degenerate output (proven by
checkpoint-950).

### Recursive forward pass specification

```python
class RecursiveSERF(nn.Module):
    """
    Wraps BaselineTRMMamba with a test-time recursive inference loop.
    During training, n_loops is fixed per curriculum step.
    During inference, n_loops can be set to any value.
    """

    def __init__(self, backbone: BaselineTRMMamba, n_loops: int = 2, ema_decay: float = 0.9):
        super().__init__()
        self.backbone = backbone
        self.n_loops = n_loops
        self.ema_decay = ema_decay

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        return_all_logits: bool = False,
    ) -> torch.Tensor:
        """
        First pass: standard forward through backbone.
        Subsequent passes: feed EMA-smoothed hidden states back as
        pseudo-embeddings, bypassing the token embedding layer.

        The backbone must expose a way to run from embeddings.
        See 'required backbone modification' below.
        """
        # Loop 0 — standard forward
        logits, hidden = self.backbone.forward_with_hidden(input_ids, attention_mask)
        all_logits = [logits] if return_all_logits else []

        ema = hidden.detach().clone()   # WARM START — not zeros

        for step in range(1, self.n_loops):
            # Bias-corrected EMA
            correction = 1.0 - self.ema_decay ** step
            loop_input = ema / correction
            logits, hidden = self.backbone.forward_from_embeddings(
                loop_input, attention_mask
            )
            ema = self.ema_decay * ema + (1.0 - self.ema_decay) * hidden.detach()
            if return_all_logits:
                all_logits.append(logits)

        if return_all_logits:
            return torch.stack(all_logits, dim=0)   # [n_loops, B, T, V]
        return logits
```

### Required backbone modification

Add two methods to `BaselineTRMMamba` in `baseline_trm_mamba.py`:

```python
def forward_with_hidden(
    self,
    input_ids: torch.Tensor,
    attention_mask: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Return (logits, final_hidden_states) for use by RecursiveSERF."""
    x = self.token_embedding(input_ids)
    if attention_mask is not None:
        attention_mask = attention_mask.to(device=input_ids.device, dtype=torch.bool)
    for group in self.groups:
        x = group(x, attention_mask)
    hidden = self.final_norm(x)
    logits = self.lm_head(hidden)
    return logits, hidden

def forward_from_embeddings(
    self,
    embeddings: torch.Tensor,
    attention_mask: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Run the backbone from pre-computed embeddings (bypass token embedding)."""
    x = embeddings
    if attention_mask is not None:
        attention_mask = attention_mask.to(device=embeddings.device, dtype=torch.bool)
    for group in self.groups:
        x = group(x, attention_mask)
    hidden = self.final_norm(x)
    logits = self.lm_head(hidden)
    return logits, hidden
```

### Training script specification

`recursive_finetune.py` must:

1. **Load a Stage 2 checkpoint** (via `--resume_from`). Use the fixed
   `load_latest_checkpoint` from Appendix A (handles direct checkpoint paths).

2. **Wrap the backbone** in `RecursiveSERF` with `n_loops` from CLI arg.

3. **Loss function** — deep supervision (weighted sum across loops):
   ```python
   gamma = 0.8
   n = len(all_logits)   # all_logits: list of [B, T, V] tensors
   weights = [gamma ** (n - 1 - i) for i in range(n)]
   total_loss = sum(w * ce_loss(logits, labels) for w, logits in zip(weights, all_logits))
   total_loss /= sum(weights)   # normalize so loss is comparable across n_loops
   ```

4. **Curriculum** — in a single run, start with `n_loops=2` for the first third of
   steps, then `n_loops=4` for the second third. This avoids shock to the model.

5. **Gate check** — after every `val_every` steps, compute val_ce with
   `n_loops=1` (baseline) AND `n_loops=current`. If
   `val_ce[n_loops] > val_ce[1] * 1.05`, print a warning and reduce `n_loops` by 1.

6. **Checkpoint format** — same as previous stages plus:
   ```python
   "n_loops": int,
   "recursive_ema_decay": float,
   ```

7. **CLI args** (minimum):
   ```
   --preset           [nano/small/medium]
   --resume_from      path to Stage 2 checkpoint
   --n_loops          int (default 4)
   --token_budget     int (default 500_000_000 for fine-tuning run)
   --lr               float (default 1e-5, lower than SFT)
   --output_dir       default runs/stage3
   --push_to_hub      flag
   --hf_token
   --wandb_mode
   ```

8. **Success criterion** — val_ce at `n_loops=4` ≤ val_ce at `n_loops=1` × 1.05.
   If met, print: `"Stage 3 gate passed. Proceed to Stage 4 (GRPO)."`

### Verification checklist

```bash
python recursive_finetune.py \
  --preset nano \
  --resume_from runs/stage2/checkpoint-XXXXXXX \
  --n_loops 2 \
  --token_budget 100_000 \
  --wandb_mode disabled
```

- [ ] Loads Stage 2 checkpoint cleanly (step > 0)
- [ ] Initial loss ≈ Stage 2 final val_ce (not 11.93 — model is already trained)
- [ ] Loss at n_loops=2 ≤ loss at n_loops=1 × 1.05 within 100 steps
- [ ] Gate check fires at val_every and prints both CE values
- [ ] Checkpoint contains `n_loops` key

---

*End of Appendix B. Appendix C (Stage 4 GRPO) will be written when Stage 3 gate is passed.*
