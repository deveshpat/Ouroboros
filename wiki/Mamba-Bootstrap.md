# Mamba Bootstrap
> Load this page when debugging wheel installation, CUDA kernel failures, or fast-path issues.

---

## Three-Phase Bootstrap

`_bootstrap()` runs unconditionally at module top-level before any third-party imports.
Under `torchrun`, rank 0 performs shared phases while other ranks wait via filesystem sync.

### Phase 1 — Pure-Python deps (rank 0 only, once per launch)
```
pip install transformers>=4.54.0 peft datasets tqdm wandb
           bitsandbytes>=0.46.1 accelerate huggingface_hub einops safetensors
```

### Phase 2 — Arch-aware wheel install (rank 0 only, once per launch)
For each of `causal_conv1d-1.6.1` and `mamba_ssm-1.2.2`:

```
1. Detect GPU: cc = torch.cuda.get_device_capability() → arch_suffix = f"sm{cc[0]}{cc[1]}"
2. Try Kaggle dataset cache (/kaggle/input/ouroboros-cache/wheels/) — sm75 only
3. Try Hub: hf_hub_download(repo="WeirdRunner/Ouroboros", filename=f"{base}-{arch_suffix}.whl")
4. Miss → source compile (MAX_JOBS=4, TORCH_CUDA_ARCH_LIST=f"{cc[0]}.{cc[1]}+PTX")
5. Upload built wheel to Hub as {base}-{arch_suffix}.whl for future sessions
6. pip install --force-reinstall --no-deps <wheel>
```

Hub accumulates one wheel per GPU arch. Compilation is rare after first session on each arch.

### Phase 3 — Fast-path verification (every rank, after shared install)
```
1. Import 5 symbols: selective_scan_fn, selective_state_update, mamba_inner_fn,
                     causal_conv1d_fn, causal_conv1d_update
2. Assert none are None
3. Run real CUDA ops: causal_conv1d_fn, selective_scan_fn, selective_state_update
4. torch.cuda.synchronize()
```

**Failure → `sys.exit(1)`. No slow-path fallback.** Slow path = ~500s/step = unusable.

---

## ABI Fingerprint (logged at Phase 3)

```
GPU={name} sm{cc} | CUDA={torch.version.cuda} | PyTorch={torch.__version__} | Python=cp{major}{minor}
```

Wheels are keyed by this fingerprint. Mismatch = recompile.

---

## Known Arch Suffixes

```python
_KNOWN_ARCH_SUFFIXES = [
    "sm60", "sm70", "sm72", "sm75", "sm80", "sm86", "sm87", "sm89",
    "sm90", "sm100", "sm120", "smunknown",
]
```

sm75 = T4 (target). sm60 = P100 (rejected by GPU guard before reaching bootstrap).

---

## Top-Level Export Shim

Jamba's transformers implementation resolves `selective_state_update` from the top-level
`mamba_ssm` package. mamba_ssm==1.2.2 only exposes it from a submodule.

Fix (`_patch_kernel_top_level_exports`): sets attributes on the top-level packages directly.
Called at bootstrap + after model load (`_patch_transformers_jamba_fast_path_globals`).

---

## DDP Synchronisation

```
rank 0: run Phase 1 + 2 → write /tmp/ouroboros_bootstrap_sync/{launch_key}/install.ok.json
rank N: poll for install.ok.json (2s interval, 2h deadline)
all ranks: run Phase 3 independently → write rank_{N}.ok
all ranks: poll for all rank_*.ok files (1s interval, 30min deadline)
```

Launch key = SHA1 of TORCHELASTIC_RUN_ID + MASTER_ADDR + MASTER_PORT + WORLD_SIZE + script path.
Prevents cross-session collisions on the same machine.

---

## Kaggle Dataset Cache (sm75 fast path)

`/kaggle/input/ouroboros-cache/wheels/` — attached via `"dataset_sources": ["weirdrunner007/ouroboros-cache"]`
in kernel-metadata.json. Skips Hub download for T4 sessions. Only active when `arch_suffix == "sm75"`.
