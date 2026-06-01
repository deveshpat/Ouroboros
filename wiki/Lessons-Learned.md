# Lessons Learned
> Operational hard lessons. Load when debugging a recurring failure class.
>
> **Guardrail rule:** this page is not allowed to be passive memory. Every table row must have a matching executable guardrail record in `ouroboros/bootstrap/guardrails.py`; local validation can use temporary inline commands instead of committed test files.

| Symptom / Mistake | Fix Applied |
|---|---|
| `kaggle kernels pull` → 403 in CI | Use local `kernels push` instead — no pull needed |
| Solo mode with outer_lr=0.7 blends stale anchor into new weights | Legacy aggregation lesson: direct weight promotion in solo mode (skip outer update) |
| `kaggle kernels push --accelerator` → unrecognized argument | Upgrade to `kaggle>=1.8.4`; add `--accelerator NvidiaTeslaT4` to push_args |
| Legacy worker quota exhausted → launch loop stalls forever | Keep current Kaggle launch manual/stateless; if automation returns, add timeout reconciliation first |
| Legacy launch state writes `triggered_workers` but push fails silently | Archived launch logic required successful push output before marking work launched |
| Kaggle CLI prints `Kernel push error`/quota text with non-fatal process behavior | Classify push output strictly: require `successfully pushed`; treat quota/error markers as failed publishing |
| `kaggle==1.6.17` + `"accelerator": "nvidiaTeslaT4"` → still P100 | Root cause 1: `--accelerator` added in v1.8.4. Root cause 2: wrong cap. Fix: `kaggle>=1.8.4` + cap fix + runtime fast-fail. All verified. |
| `--use_halt_gate` starts from random LoRA weights instead of the Hub anchor | `--resume_from_anchor` loads the configured Hub anchor before DGAC training |
| Kaggle command hidden behind launch-mode modules | `kaggle-utils.ipynb` owns the visible launch command |
| `last_hidden_state` None | assert in all forward paths |
| OOM at val | `empty_cache()` + small `val_batch_size`; validation/generation must stay inference-only |
| mamba-ssm 2.x API break | Pinned to 1.2.2 |
| NCCL watchdog kills DDP val | `timedelta(hours=4)` + env var |
| BF16 emulation on T4 | `_amp_dtype` checks `cc >= (8,0)` |
| Jamba fast path declared active but generation raises `Fast Mamba kernels are not available` | Shared post-load model runtime probe gates baseline, candidate, and inference loaders before eval/generation loops |
| HaltGate target can look good under teacher-forced CE while generated answers degrade | Treat teacher-forced CE as health-only; gate release on generated-answer artifacts and raw output inspection |
| Fixed-depth ablation can pass a small slice but fail the hardest slice | Label fixed-depth runs as diagnostic/OOM checks unless full release-valid artifacts pass |
| PEFT adapter config loaded with ignored keys | Align PEFT version with training/runtime or reproduce both paths before making public claims |
| OOM fixes can make eval complete without proving model quality | Separate memory-stability success from generated-answer quality success |
