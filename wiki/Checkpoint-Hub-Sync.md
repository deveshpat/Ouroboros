# Checkpoint + Hub Sync

Load when checkpoint/resume/Hub/anchor path confusing.

## Hub map

```text
WeirdRunner/Ouroboros
-> diloco_state/anchor/adapter_model.safetensors
-> diloco_state/anchor/adapter_config.json
-> diloco_state/anchor/halt_gate.pt
-> diloco_state/round_state.json
-> diloco_state/workers/{A,B,C}/status.json
-> diloco_state/workers/{A,B,C}/round_{NNNN}_stage_{K}/adapter_model.safetensors
-> runs/*/stage_{K}/checkpoint-{NNNNNNN}/
```

## Startup sync

rank0 sequential path -> upload local checkpoints -> prune numbered locals -> keep resume target + `best/`.

Purpose -> avoid Kaggle disk overflow.

## Anchor load

`diloco_download_anchor` -> download adapter -> load PEFT weights -> load `halt_gate.pt` when present.

First round -> no anchor -> random LoRA ok.
`required=True` -> missing anchor -> fail fast.

## DGAC anchor start

```text
--use_halt_gate --resume_from_diloco_anchor
-> load diloco_state/anchor
-> restore adapter + halt_gate.pt
-> fresh optimizer unless eval-only
-> run terminal-stage DGAC
```

Use this for anchor-start DGAC only.
Numbered checkpoint resume -> use `--resume_from` or Hub checkpoint discovery.
Do not pair numbered checkpoint eval with `--resume_from_diloco_anchor`.

## Worker upload

worker train -> save adapter -> upload worker round dir -> upload `status.json` -> push signal.
attendance worker -> upload `samples_seen=0` status -> coordinator sees presence.

## H100 DGAC checkpoint

Azure H100 run -> `runs/azure_h100_dgac/stage_10/checkpoint-0001154`.
Promoted -> `diloco_state/anchor`.
Val/gen skipped -> training evidence only until eval gate runs.

## Resume priority

`--resume_from` -> latest local checkpoint -> latest Hub checkpoint -> stage `best/`.
DDP -> rank0 resolves -> writes `.resolved_resume_path.txt` -> ranks read after barrier.
