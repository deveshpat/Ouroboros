# GPU Guardrails

Load when runtime crashes, slow path, or wrong GPU appears.

## Device

T4/V100 -> FP16.
A100/H100 -> BF16.
MPS -> FP16 when available.
CPU -> FP32.

## Kaggle

metadata accelerator -> `NvidiaTeslaT4`.
CLI accelerator -> `--accelerator NvidiaTeslaT4`.
GPU cc < 7.5 -> fast-fail -> reset dispatch -> signal.

## Eval memory

eval/gen -> no grad / inference mode.
OOM at val -> check inference guard before burning GPU time.

## NCCL

set watchdog/heartbeat env before torch import.
DDP val timeout -> Bootstrap-owned guard.
