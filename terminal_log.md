# Terminal Log

Rolling buffer -> last relevant run only.
Keep <=80 lines.

## Last run -> sampled generated-answer comparison failed

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
```

Result -> failed generated-answer diagnostic. Do not promote anchor or run full validation yet.

Next -> run fixed-depth ablation using `--disable_candidate_halt_gate` with `--candidate_requires_halt_gate`; if sampled fixed-depth still fails, investigate checkpoint/runtime/prompt path before spending full-validation quota.
