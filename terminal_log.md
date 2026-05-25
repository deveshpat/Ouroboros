# Terminal Log

Rolling buffer -> last relevant run only.
Keep <=80 lines.

## Last relevant result -> HaltGate over-stop confirmed by fixed-depth ablation

### HaltGate-enabled sampled generated-answer comparison

```text
mode -> compare-coconut-val
candidate -> WeirdRunner/Ouroboros/diloco_state/anchor
baseline -> ai21labs/AI21-Jamba-Reasoning-3B
dataset -> WeirdRunner/Ouroboros coconut-v1 validation
revision -> 6a52cd0c47be1e7b85d9018225387950aefc4631
local validation -> 1940 rows, 1809 scorable, 131 missing answer_norm, 0 duplicate ids
sample -> 25 scorable rows
stage_k -> 10
halt_threshold -> 0.5
gen_max_tokens -> 128
use_chat_template -> true
mamba CUDA kernels -> fast path ACTIVE after load
baseline generated_answer_exact_match -> 0.08
candidate generated_answer_exact_match -> 0.04
candidate actual_latents_mean -> 3.16
candidate actual_latents histogram -> 1:19, 10:6
one_latent_fraction -> 0.76
status -> failed_candidate_regression
```

### Fixed-depth diagnostic ablation

```text
mode -> compare-coconut-val --disable_candidate_halt_gate --candidate_requires_halt_gate
sample -> same 25 scorable rows
baseline generated_answer_exact_match -> 0.08
candidate generated_answer_exact_match -> 0.12
candidate actual_latents_mean -> 10.0
candidate actual_latents histogram -> 10:25
status -> passed diagnostic-only gate
```

Result -> adapter/tokenizer/runtime path is unlikely to be the primary fault. HaltGate decisions are the current blocker; the gate over-stops at one latent under threshold 0.5.

Next -> train/calibrate HaltGate, then rerun sampled HaltGate-enabled `compare-coconut-val`. Do not promote, publish claims, or spend full-validation quota until sampled HaltGate-enabled artifacts pass inspection.
