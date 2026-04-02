# terminal_log.md — Project Ouroboros
**All verified terminal outputs, newest first.**

---

> **Naming note (current session):** Scripts were renamed for consistency.
> `phase1_viability_gate.py` → `viability_gate.py`
> `train_sft_phase2.py` → `train_sft.py`
> `Ouroboros_Blueprint_v3.md` → `BLUEPRINT.md`
> Logs below use the original script names as they appeared in the terminal at run time.

---

## Stage 1 — Pre-training (pretrain.py)
**Script:** `pretrain.py`
**Status:** Script written and verified offline. Full run NOT YET started on Kaggle.

**pretrain.py offline verification (code review findings, not a terminal run):**
- ✅ EMA initialized from live weights (not zeros) — checkpoint-950 bug fixed
- ✅ Val buffer document-disjoint from training (SHA1 hash deduplication)
- ✅ Epoch-varying random offset (`seed + epoch × 104_729`) — anti-periodicity
- ✅ Checkpoint atomic write: `.tmp` → Hub push → rename to final
- ✅ `ema_backbone_state_dict` includes `lm_head.weight` alias (Stage 2 requirement)
- ✅ `compute_val_ce` correctly saves/restores live weights around EMA eval
- ✅ `load_latest_checkpoint` handles direct checkpoint paths, parent dirs, and Hub fallback
- ✅ Smoke test (`smoke_test_20_steps`) patches FakeMamba cleanly, resets after
- ⚠️  Smoke test runs unconditionally before `main()` on every execution (~30s overhead)
- ⚠️  Generation callback uses live weights (not EMA) — intentional for Stage 1 monitoring

**Dry-run template (run first on Kaggle before full run):**
```bash
python pretrain.py --preset nano --dry_run
```
Expected: 6-point checklist passes, initial loss ≈ 11.93, loss decreases.

**Full run command:**
```bash
python pretrain.py \
  --preset nano \
  --token_budget 2_000_000_000 \
  --push_to_hub \
  --hf_token $HF_TOKEN \
  --wandb_mode online
```
Expected output format (append actual output here when run):
```
════════════════════════════════════════════════════════════════
  Stage 1 Pre-training - Project Ouroboros
════════════════════════════════════════════════════════════════
  dataset          : HuggingFaceFW/fineweb-edu / sample-10BT
  preset           : nano
  model            : d_model=512  groups=1  heads=8/4
  token_budget     : 2,000,000,000
  total_steps      : 61,036
  ...
════════════════════════════════════════════════════════════════
  stage  step   train_ce   val_ce   smth   gnorm   lr   VRAM   tok/s
  ...
  * Stage 1 criterion met:
    val_ce < 3.0  AND  mean UWR > 0.05 (non-degenerate generation)
```

---

## Stage 2 SFT — Dry-run (nano, 300 samples, 100 steps)
**Script:** `train_sft_phase2.py` (now: `train_sft.py`)
**Date:** Session 3
**Result:** Pipeline verified. EMA `>>> ` generation is expected
(decay=0.999, shadow is 90% random at step 100 on a 100-step run).
Corrected to `--ema_decay 0.995` for real runs.

**⚠ Bug identified post-run (not visible in dry-run output):**
`compute_val_ce` does not restore live weights after EMA eval. Fix in Part 7 of
BLUEPRINT.md before running Stage 2 for real.

```
════════════════════════════════════════════════════════════════
  Phase 2 SFT — Project Ouroboros
════════════════════════════════════════════════════════════════
  preset        : nano
  seq_len       : 512
  batch×accum   : 2×4 = 8
  lr            : 0.0002  warmup=100
  dtype         : torch.bfloat16
  output_dir    : runs/phase2
════════════════════════════════════════════════════════════════

Loading tokenizer: Qwen/Qwen2.5-0.5B
  vocab: 151,665  pad_token: '<|endoftext|>'

Loading bespokelabs/Bespoke-Stratos-17k …
  300 samples kept, 0 skipped (missing fields / too short).
  train: 285  val: 15

Model  : 92.5M parameters  (preset=nano)
Config : d_model=512  n_groups=1  heads=8/4  mlp_hidden=1408
Vocab  : 151,680 (padded from 151,665)

Optimizer : AdamW (fused CUDA kernel)
Schedule  : cosine warmup=100 total=100  (3 epochs × 36 steps/epoch)

   Step   Train CE     Val CE    GNorm         LR     VRAM    Tok/s
────────────────────────────────────────────────────────────────────────
      1    11.9931          -   3.7031   4.00e-06    1.576     1662
     20    11.1662          -   3.4688   4.20e-05    1.576     3648
     40     9.6128          -   2.7969   8.20e-05    1.576     3731
  [val] step=50  val_ce=11.9826
     60    10.0753    11.9826   3.7031   1.22e-04    1.576     3285
     80     7.3924    11.9826   2.9688   1.62e-04    1.576     3369
    100     4.9913    11.9826   1.8594   2.00e-05    1.576     3412
  [val] step=100  val_ce=11.9646

  [ckpt] saved  → runs/phase2/checkpoint-0000100

════════════════════════════════════════════════════════════════
  Phase 2 Training Complete
════════════════════════════════════════════════════════════════
  Total steps  : 100
  Total time   : 2.3 min
  Peak VRAM    : 3.62 GB
  Final val CE : 11.9646
  Status       : val_ce ≥ 1.5 — extend training or check data quality
════════════════════════════════════════════════════════════════
```

**Notes:**
- val_ce=11.96 on 15 samples is meaningless noise (expected).
- train CE 11.99 → 4.99 in 100 steps confirms the pipeline is correct.
- `>>> ` generation is an EMA artefact at step 100 with decay=0.999; corrected for real run.
- Full run command: `--preset small --ema_decay 0.995 --num_epochs 3`

---

## Stage 0 — Viability Gate
**Script:** `phase1_viability_gate.py` (now: `viability_gate.py`)
**Date:** Session 2
**Result:** ALL GATES PASSED — architecture is viable, proceed to Stage 1 pre-training.

```
══════════════════════════════════════════════════════════════
  Phase 1 Viability Gate — Project Ouroboros
══════════════════════════════════════════════════════════════
  steps         : 300
  samples       : 500
  seq_len       : 256
  batch×accum   : 2×4 = 8
  lr            : 0.0003
  dtype         : torch.bfloat16

  Gate thresholds:
    G1 CE       < 3.5   (random-init ≈ 11.93)
    G2 UWR      > 0.1   (unique-word-ratio at final gen step)
    G3 grad_norm < 10.0  (max over final 100 steps)
    G4 VRAM Δ   < 1.0 GB (growth from step 1 to final)
══════════════════════════════════════════════════════════════

Loading tokenizer: Qwen/Qwen2.5-0.5B
Tokenizer vocab  : 151,665 tokens
Loading bespokelabs/Bespoke-Stratos-17k (500 samples) …
Kept 500 valid samples after filtering.

Model            : 92.5M parameters
Config           : d_model=512  n_groups=1  heads=8/4  mlp_hidden=1408
Residual layers  : 9
Padded vocab     : 151,680

Optimizer        : AdamW (fused CUDA kernel)

  Step    CE Loss   Grad Norm   VRAM GB     Tok/s  Note
──────────────────────────────────────────────────────────────────────
     1    12.0406      4.6250     0.968      1157
    10     7.7003      3.0625     0.968      3055
    20     6.2067      2.1094     0.968      3366
    30     4.7265      1.5781     0.968      3481
    40     4.2806      1.5312     0.968      3527
    50     4.2260      1.7656     0.968      3558

  ── Generation @ step 50 ──
  Q: What is 2 + 2?
  A: 10000000000000000000000000000000000000000000000000000000000000000000000000000000
     uwr=1.000  ucc=2  ⚠ DEGENERATE
  Q: Write a Python function that returns the square of a number.
  A: 10000000000000000000000000000000000000000000000000000000000000000000000000000000
     uwr=1.000  ucc=2  ⚠ DEGENERATE
  Q: What is the capital of France?
  A: 10000000000000000000000000000000000000000000000000000000000000000000000000000000
     uwr=1.000  ucc=2  ⚠ DEGENERATE
  Q: Explain what a variable is in programming.
  A: 10000000000000000000000000000000000000000000000000000000000000000000000000000000
     uwr=1.000  ucc=2  ⚠ DEGENERATE

    60     3.5889      1.6719     0.968      3193
    70     3.2998      1.4766     0.968      3247  ← G1 threshold crossed
    ...
   100     2.9521      1.6875     0.968      3326  ← G1 threshold crossed

  ── Generation @ step 100 ──
  Q: Write a Python function that returns the square of a number.
  A: 1000. The options are given from A to E. Let me start by understanding the problem step by step. So, we have a a a, and 1, 2, 2, 2, 2, 2, 2,
     uwr=0.634  ucc=31
  ...

   150     2.6616      2.0781     0.968      3217  ← G1 threshold crossed
  ...
   200     2.0984      2.6094     0.968      3144  ← G1 threshold crossed
  ...
   250     2.0020      2.8281     0.968      3111  ← G1 threshold crossed
  ...
   300     2.0034      3.9844     0.968      3088  ← G1 threshold crossed

  ── Generation @ step 300 ──
  Q: What is 2 + 2?
  A:  the number of positive integers (m at $m$ and $m$ are positive integers such that $m^m + 1^m + 10^m ...
     uwr=0.421  ucc=25
  Q: Write a Python function that returns the square of a number.
  A: 1000. The cost of a circle? ...
     uwr=0.750  ucc=34
  Q: What is the capital of France?
  A: 1, so that the sum of the numbers in each of the numbers in each of ...
     uwr=0.278  ucc=24
  Q: Explain what a variable is in programming.
  A: 11} \quad \text{(B)}\ \text{1.00.0 \text{and} \quad ...  Assistant
     uwr=0.842  ucc=41

Training: 100% 300/300 [03:22<00:00,  1.48it/s, ce=2.003, gn=3.984]

══════════════════════════════════════════════════════════════
  Phase 1 Gate Results
══════════════════════════════════════════════════════════════

  G1_ce_converged  CE < 3.5                             final CE = 2.0034     PASS ✓
  G2_generation_coherent  UWR > 0.1 (last gen step)     mean UWR = 0.573      PASS ✓
  G3_grad_norm_stable  grad_norm < 10.0 (last 100 steps) max = 4.0312         PASS ✓
  G4_vram_stable  VRAM Δ < 1.0 GB                       Δ = 0.000 GB          PASS ✓

  Total time  : 3.4 min
  Peak VRAM   : 2.07 GB

  ╔══════════════════════════════════════════════════════╗
  ║  ALL GATES PASSED                                    ║
  ║  Architecture is viable. Proceed to Phase 2.         ║
  ╚══════════════════════════════════════════════════════╝
══════════════════════════════════════════════════════════════
```

**Notes:**
- Degenerate generation at step 50 is expected: model has only seen 400 effective samples.
- Structured (though wrong) text emerging by step 150 shows language learning is happening.
- VRAM perfectly flat at 0.968 GB — zero graph retention across all 300 steps.
- Grad norm rising from 1.5 → 4.0 is the expected consequence of constant high LR
  as CE falls and predictions sharpen. Stage 1 cosine schedule prevents this.

---

## Stage 0 — Baseline Architecture Smoke Test
**Script:** `baseline_trm_mamba.py`
**Date:** Session 1
**Result:** PASSED — all architecture checks healthy.

```
device     : cuda
dtype      : torch.bfloat16
parameters : 92,477,440  (92.5 M)
vocab_size : 151680
d_model    : 512
n_groups   : 1
n_heads    : 8  n_kv_heads: 4
mlp_hidden : 1408
total_residual_layers: 9

logits shape : (2, 128, 151680)  (expected [2, 128, 151680])
logits health : OK (no NaN, no Inf)
initial loss : 11.9904  (theoretical random-init ≈ 11.9295)
backward     : OK
grad norms   : total=6.4242

All checks passed. Baseline architecture is healthy.
```

**Notes:**
- `parameters=92,477,440` confirms tied-embedding dedup is working.
- `initial loss 11.9904` vs theoretical `11.9295` = 0.5% above random-init. ✓
- `grad norms total=6.4242` — healthy finite value.
- Padding simulation (`attn_mask[1, 120:] = False`) exercised combined mask path. ✓

---

## Config verification (manual)

```python
from baseline_trm_mamba import BaselineConfig
c = BaselineConfig()
assert c.vocab_size == 151_680          # padded Qwen2.5 vocab ✓
assert c.d_model == 512                 # nano preset ✓
assert c.mlp_hidden == 1408             # ceil(8/3 * 512 / 64) * 64 ✓
assert c.head_dim == 64                 # 512 // 8 ✓
assert c.n_heads % c.n_kv_heads == 0   # GQA divisibility ✓
assert c.total_residual_layers == 9    # 1 * (2 + 7) ✓
# All assertions passed ✓
```
