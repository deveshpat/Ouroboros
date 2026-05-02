# Checkpoint & Hub Sync
> Load this page when debugging checkpoint resume, Hub uploads, or DiLoCo anchor loading.

---

## Path Map

```
WeirdRunner/Ouroboros (HF Hub)
├── diloco_state/
│   ├── anchor/
│   │   ├── adapter_model.safetensors   ← aggregated anchor after each coordinator run
│   │   └── adapter_config.json
│   ├── round_state.json                ← coordinator state machine
│   └── workers/{A,B,C}/
│       ├── status.json                 ← {worker_id, stage_k, round_n, samples_seen, status, weights_path}
│       └── round_{NNNN}_stage_{K}/
│           ├── adapter_model.safetensors
│           └── adapter_config.json
└── runs/stage3/                        ← sequential curriculum checkpoints (if --push_to_hub)
    └── stage_{K}/
        ├── checkpoint-{NNNNNNN}/
        └── best/
```

---

## Startup Sync + Prune (`startup_hub_sync_and_prune`)

Called once at session start (rank 0 only, sequential path only — not DiLoCo).

1. Find all local checkpoints under `output_dir/`
2. Upload each to Hub (under `hf_stage_subdir/stage_{K}/`)
3. Delete all local numbered checkpoints EXCEPT the resume target
4. Always preserve `best/` dirs

Purpose: prevent Kaggle disk overflow across sessions. Upload failures are logged but pruning still proceeds.

---

## DiLoCo Anchor Load (`diloco_download_anchor`)

Called by coordinator (aggregation) and worker (round start).

```python
hf_hub_download(repo_id, filename=f"{anchor_path}/adapter_model.safetensors", token)
set_peft_model_state_dict(model, load_file(local_path, device=str(device)))
```

Falls back silently if no anchor exists (first round uses random LoRA init).

---

## DGAC Resume from DiLoCo (`--resume_from_diloco_anchor`)

For Phase 3.4 (DGAC) only. Loads DiLoCo stage-K aggregate instead of a sequential checkpoint.

```
--use_halt_gate --resume_from_diloco_anchor
  → diloco_download_anchor(model, "diloco_state/anchor/")
  → HaltGate at zero-init
  → Optimizer starts fresh
  → run_training_stages([curriculum_max_stage], ...)
```

Bypasses `find_latest_resume_checkpoint()` entirely.
Requires `--hf_token` and `--diloco_state_repo WeirdRunner/Ouroboros`.

---

## Worker State Upload (`diloco_upload_worker_state`)

After each training round:
1. `model.save_pretrained(upload_dir)` — saves adapter weights locally
2. Upload `adapter_model.safetensors` + `adapter_config.json` to `diloco_state/workers/{id}/round_{N}_stage_{K}/`
3. Upload `status.json` to `diloco_state/workers/{id}/status.json`

Attendance workers (samples_seen=0) still upload status.json — coordinator uses it to confirm presence.

---

## Checkpoint Resume Logic (sequential path)

Priority:
1. `--resume_from <path>` (explicit)
2. Latest local checkpoint (highest `(stage_k, epoch, step_in_epoch, step)`)
3. Latest Hub checkpoint (scanned via `_list_hub_stage_checkpoints`, downloaded to `.hub_resume/`)

`best/` checkpoints are used to load weights between stages but are NOT resumed mid-training (`.name != "best"` check).

In DDP: rank 0 resolves path, writes to `output_dir/.resolved_resume_path.txt`, all ranks read it after barrier.
