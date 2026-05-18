# Terminal Log

Rolling buffer -> last relevant run only.
Keep <=80 lines.

## Last run -> DGAC anchor eval-only health pass

```text
Loaded 36906 train / 1940 val from data/coconut_v1
Step stats: median=10 mean=10.42 max=16
GPU -> Tesla T4 sm75 16GB -> fp16
model -> ai21labs/AI21-Jamba-Reasoning-3B
<|lat|> token id -> 65536
mamba CUDA kernels -> fast path ACTIVE
flash-attn -> not installed, eager fallback
trainable params -> 26,851,328 / 3,056,191,360 = 0.8786%
DGAC HaltGate -> d_model=2560 params=5121
anchor load -> WeirdRunner/Ouroboros/diloco_state/anchor
adapter -> restored
halt_gate.pt -> restored
mode -> eval-only
stage -> 10
validation samples -> 1940
rank0 shard -> 970/970 complete
val_ce -> 0.4114
val_token_acc -> 0.8693
```

Result -> canonical anchor is healthy enough for next release gate.

Warning -> this is not a comparison benchmark and should not be used to claim Ouroboros beats base Jamba.

Next -> run unbiased Jamba-vs-Ouroboros comparison eval with clean prompt/template/decoding controls and explicit contamination notes.
