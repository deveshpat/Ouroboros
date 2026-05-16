# Terminal Log

Rolling buffer -> last relevant run only.
Keep <=80 lines.

## Last run -> H100 DGAC checkpoint promoted

```text
Loaded 36906 train / 1940 val
GPU -> NVIDIA H100 NVL sm90 100GB -> BF16
mamba fast path -> active
DGAC HaltGate -> d_model=2560 params=5121
anchor load -> diloco_state/anchor adapter + halt_gate.pt
stage -> 10/10
step 1150 -> ce=0.3538 gn=0.3338
val/gen -> skipped by timeout buffer
checkpoint -> runs/azure_h100_dgac/stage_10/checkpoint-0001154
Hub upload -> WeirdRunner/Ouroboros
promote -> diloco_state/anchor adapter/config/halt_gate.pt
```

Result -> canonical anchor updated.
Warning -> no quality proof yet.
Next -> run `dgac-anchor-eval`.
