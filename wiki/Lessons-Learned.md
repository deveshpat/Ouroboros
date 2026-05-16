# Lessons Learned
> Operational hard lessons. Load when debugging a recurring failure class.
>
> **Guardrail rule:** this page is not allowed to be passive memory. Every table row must have a matching executable guardrail record in `ouroboros/hard_lesson_guardrails.py`; tests fail when a new lesson is documented without a preflight, runtime guard, regression test, or known-error signature.

| Symptom / Mistake | Fix Applied |
|---|---|
| `kaggle kernels pull` → 403 in CI | Use local `kernels push` instead — no pull needed |
| Solo mode with outer_lr=0.7 blends stale anchor into new weights | Direct weight promotion in solo mode (skip outer update) |
| `kaggle kernels push --accelerator` → unrecognized argument | Upgrade to `kaggle>=1.8.4`; add `--accelerator NvidiaTeslaT4` to push_args |
| Worker C quota exhausted → coordinator stalls forever | `triggered_at` + 13h timeout + attendance mechanism |
| Coordinator writes `triggered_workers` but push fails silently | `triggered_at=0` manual reset → immediate re-dispatch on next run |
| Kaggle CLI prints `Kernel push error`/quota text with non-fatal process behavior | Classify push output strictly: require `successfully pushed`; treat quota/error markers as failed dispatch for immediate reconciliation |
| `kaggle==1.6.17` + `"accelerator": "nvidiaTeslaT4"` → still P100 | Root cause 1: `--accelerator` added in v1.8.4. Root cause 2: wrong cap. Fix: `kaggle>=1.8.4` + cap fix + runtime fast-fail. All verified. |
| `--use_halt_gate` starts from random LoRA weights in DiLoCo path | `--resume_from_diloco_anchor` loads `diloco_state/anchor/` before DGAC training |
| `last_hidden_state` None | assert in all forward paths |
| OOM at val | `empty_cache()` + `val_batch_size=2`; validation/generation/DGAC diagnostics must stay inference-only |
| DGAC diagnostics eval-only OOM in `selective_scan_fn`/bitsandbytes dequantization | `run_dgac_diagnostics()` and DGAC CE helpers run under `torch.inference_mode()`; known log signature triages directly to this guardrail |
| mamba-ssm 2.x API break | Pinned to 1.2.2 |
| NCCL watchdog kills DDP val | `timedelta(hours=4)` + env var |
| BF16 emulation on T4 | `_amp_dtype` checks `cc >= (8,0)` |
