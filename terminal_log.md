# terminal_log.md — Project Ouroboros
> **Rolling buffer — last relevant run only.**
> Historical record moved to [wiki/SessionLog.md](wiki/SessionLog.md).
> Trim to ≤80 lines at each session update.

---

## Last Run — DGAC DiLoCo Complete; post-DGAC eval loader bug found (2026-05-10)

Coordinator evidence:

```text
[coordinator] New anchor uploaded:
DiLoCo anchor: stage 10 round 1 (3 workers, 1386 samples, mode=diloco)
[coordinator] Stage 10 progress:
36906/36906 samples seen
[coordinator] DGAC DiLoCo COMPLETE (36906/36906 samples).
[coordinator] Done (DGAC DiLoCo complete).
```

Eval-loader bug evidence from the attempted `dgac-anchor-eval` path:

```text
[diloco] Loaded anchor weights from diloco_state/anchor
[DGAC] Anchor loaded. HaltGate at zero-init. gate_stage will default to curriculum_max_stage. Optimizer starts fresh.
```

Result: do not use that eval for post-DGAC HaltGate quality. Patch `--resume_from_diloco_anchor` so it passes the live `HaltGate` into `diloco_download_anchor`; rerun `kaggle_run_mode=dgac-anchor-eval` and require:

```text
[diloco] Loaded halt gate from diloco_state/anchor/halt_gate.pt
```
