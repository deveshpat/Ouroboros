---
title: Kaggle GPU P100 Silent Fallback
type: debug
sources:
  - terminal_log.md
  - BLUEPRINT.md
  - ouroboros/diloco/kaggle_dispatch.py
  - .github/workflows/diloco_coordinator.yml
updated: 2026-04-30
---

# Kaggle GPU P100 Silent Fallback

## Symptom

Worker B kernel (version 69, then 70) ran on GPU P100 instead of T4. The coordinator
dispatched with `"accelerator": "nvidiaTeslaT4"` in `kernel-metadata.json` and the
intent was clear, but Kaggle silently assigned P100 (the documented default GPU).
P100 is `sm60` — it cannot run BF16 and is ~2× slower than T4 for this workload.
The worker ran for 28 minutes before being manually cancelled.

## Root Cause — Two Independent Failures

**Root cause 1 (primary): Feature didn't exist in the pinned SDK version.**

`kaggle==1.6.17` was pinned for an unrelated reason (avoiding a 403 on
`KernelsApiService/GetKernel` — the *pull* endpoint). It predates the
`--accelerator` flag by at least two major versions. The `KernelPushRequest`
object in v1.6.17's REST/Swagger API has no GPU-type field. The
`"accelerator"` key in `kernel-metadata.json` was silently discarded by the
client before the API call. Kaggle received a request with only `"enable_gpu": true`
and assigned P100 as the documented default.

Evidence: `github.com/Kaggle/kaggle-cli/blob/main/CHANGELOG.md`
> v1.8.4: "Add `--acc` to set accelerator for: `kaggle kernels push` ... (#907)"

**Root cause 2 (secondary): Wrong capitalisation.**

Even if the JSON field had been read, `"nvidiaTeslaT4"` (lowercase n) does not
match the official valid value `"NvidiaTeslaT4"` (capital N).

Evidence: `github.com/Kaggle/kaggle-cli/blob/main/docs/kernels.md`
> `--accelerator <ACCELERATOR_ID>`: `"NvidiaTeslaP100"` (default), `"NvidiaTeslaT4"`, `"TpuV6E8"`

**Why the original pin was safe to lift:**
The Session 15 403 was on `KernelsApiService/GetKernel` (pull endpoint). The new
flow is push-only. `kernels push` uses a different method in `kaggle>=1.8.3` that
is not blocked by the Kaggle API ACL that caused the original 403.

## Fix (three-part, all verified)

**1. Upgrade SDK — `.github/workflows/diloco_coordinator.yml`:**
```
- "kaggle==1.6.17"
+ "kaggle>=1.8.4"
```

**2. Correct capitalisation — `kernel-metadata.json` (via `build_kaggle_kernel_metadata()`):**
```python
- "accelerator": "nvidiaTeslaT4"
+ "accelerator": "NvidiaTeslaT4"    # KAGGLE_ACCELERATOR_T4 constant
```

**3. CLI flag — `push_args` in `KaggleCliDispatcher.dispatch()`:**
```python
["kaggle", "kernels", "push", "-p", str(tmp_path), "--accelerator", request.accelerator]
```
Belt-and-suspenders: both JSON metadata and CLI flag are set.

**4. Runtime fast-fail safety net — `main()` in the worker script:**
If `diloco_mode` and GPU compute capability `cc < (7,5)` (i.e. P100 at sm60):
- Reset `triggered_at=0` in round state
- Push signal file
- `sys.exit(0)`

This causes the coordinator to re-dispatch on its next run rather than wasting
a 12h session on the wrong GPU.

## Verification

No P100 assignments observed since deploying the fix. The runtime fast-fail has
not triggered (meaning the SDK fix is doing its job). Verified across stages 4, 5,
6, and 7 round 0.

## Constants

`KAGGLE_ACCELERATOR_T4 = "NvidiaTeslaT4"` is the single source of truth in
`ouroboros/diloco/kaggle_dispatch.py`. All metadata generation and dispatch calls
reference this constant. Never hardcode the accelerator string elsewhere.
