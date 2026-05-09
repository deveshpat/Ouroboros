# terminal_log.md — Project Ouroboros
> **Rolling buffer — last relevant run only.**
> Historical record moved to [wiki/SessionLog.md](wiki/SessionLog.md).
> Trim to ≤80 lines at each session update.

---

## Last Run — Stage 10 Terminal Anchor Eval-Only → DGAC Cleared (2026-05-09)

Observed from Kaggle GPU eval-only output after Stage 10 terminal DiLoCo aggregation.

```text
Loaded 36906 train / 1940 val from data/coconut_v1
[GPU] Tesla T4  cc=sm75  VRAM=16GB  amp_dtype=float16
Loading tokenizer: ai21labs/AI21-Jamba-Reasoning-3B
flash-attn not installed: falling back to eager attention
mamba CUDA kernels: fast path ACTIVE (verified at bootstrap)
Loading model: ai21labs/AI21-Jamba-Reasoning-3B
trainable params: 26,851,328 || all params: 3,056,191,360 || trainable%: 0.8786
DGAC HaltGate: d_model=2560  params=5121
```

Anchor load:

```text
[DGAC] Loading DiLoCo anchor from WeirdRunner/Ouroboros/diloco_state/anchor as base weights for Phase 3.4 DGAC training.
[diloco] Loaded anchor weights from diloco_state/anchor
[DGAC] Anchor loaded. HaltGate at zero-init. gate_stage will default to curriculum_max_stage. Optimizer starts fresh.
```

Eval-only result:

```text
[eval-only] stage=10 val_ce=0.4863 val_acc=0.0889
```

Generation samples:

```text
Q: What is 15 + 27?
A: The answer is: 42  [k_actual=10 uwr=1.000]

Q: Write a Python function that returns the factorial of n.
A: Here is a simple Python function that calculates the factorial of a given number: ...  [k_actual=10 uwr=0.688]

Q: What is the capital of Japan?
A: The capital of Japan is Tokyo. ...  [k_actual=10 uwr=0.739]

Q: Explain what a neural network is in simple terms.
A: A neural network is a type of machine learning model ...  [k_actual=10 uwr=0.677]

Q: Solve for x: 3x + 6 = 21.
A: To solve for x, we need to isolate it ... 3x = 15 ...  [k_actual=10 uwr=0.560]

Mean UWR: 0.733
```

Result: terminal anchor quality gate passed. `k_actual=10` is expected before DGAC because HaltGate is zero-init. Proceed with explicit Phase 3.4 DGAC launch via `kaggle_run_mode=dgac-train`.
