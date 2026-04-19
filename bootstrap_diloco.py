#!/usr/bin/env python3
"""
Bootstrap the initial DiLoCo anchor and round_state.json on HuggingFace Hub.

This copies an existing adapter checkpoint into:
    diloco_state/anchor/

Default source matches the blueprint's current Hub layout:
    runs/stage3/checkpoint-0002987/adapter_model/
"""

import argparse
import json
import os
import tempfile
from pathlib import Path
from typing import List

from huggingface_hub import HfApi, hf_hub_download


DEFAULT_REPO_ID = "WeirdRunner/Ouroboros"
DEFAULT_SOURCE_CHECKPOINT = "runs/stage3/checkpoint-0002987"
DEFAULT_STAGE_K = 2
DEFAULT_ROUND_N = 0
DEFAULT_SAMPLES_SEEN = 21728
DEFAULT_COMPLETED_STAGES = [0, 1]
ANCHOR_PREFIX = "diloco_state/anchor"
ROUND_STATE_PATH = "diloco_state/round_state.json"



def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Initialize DiLoCo anchor + round state on HF Hub")
    parser.add_argument("--hf_token", default=os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_HUB_TOKEN"))
    parser.add_argument("--repo_id", default=DEFAULT_REPO_ID)
    parser.add_argument("--source_checkpoint", default=DEFAULT_SOURCE_CHECKPOINT)
    parser.add_argument("--stage_k", type=int, default=DEFAULT_STAGE_K)
    parser.add_argument("--round_n", type=int, default=DEFAULT_ROUND_N)
    parser.add_argument("--samples_seen", type=int, default=DEFAULT_SAMPLES_SEEN)
    parser.add_argument("--completed_stages", nargs="*", type=int, default=DEFAULT_COMPLETED_STAGES)
    return parser.parse_args()



def normalize_source_adapter_prefix(source_checkpoint: str) -> str:
    source_checkpoint = source_checkpoint.strip("/")
    if source_checkpoint.endswith("/adapter_model"):
        return source_checkpoint
    return f"{source_checkpoint}/adapter_model"



def upload_json(repo_id: str, path_in_repo: str, data: dict, token: str, message: str) -> None:
    api = HfApi(token=token)
    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False, encoding="utf-8") as tf:
        json.dump(data, tf, indent=2)
        tmp_path = tf.name
    try:
        api.upload_file(
            path_or_fileobj=tmp_path,
            path_in_repo=path_in_repo,
            repo_id=repo_id,
            token=token,
            commit_message=message,
        )
    finally:
        Path(tmp_path).unlink(missing_ok=True)



def main() -> None:
    args = parse_args()
    if not args.hf_token:
        raise SystemExit("HF token required. Pass --hf_token or set HF_TOKEN / HUGGINGFACE_HUB_TOKEN.")

    api = HfApi(token=args.hf_token)
    api.create_repo(repo_id=args.repo_id, repo_type="model", private=True, exist_ok=True, token=args.hf_token)

    source_prefix = normalize_source_adapter_prefix(args.source_checkpoint)
    print(f"[bootstrap] Source adapter prefix: {source_prefix}")

    for fname in ["adapter_model.safetensors", "adapter_config.json"]:
        source_file = f"{source_prefix}/{fname}"
        print(f"[bootstrap] Downloading {source_file} ...")
        local = hf_hub_download(repo_id=args.repo_id, filename=source_file, token=args.hf_token)
        print(f"[bootstrap] Uploading {ANCHOR_PREFIX}/{fname} ...")
        api.upload_file(
            path_or_fileobj=local,
            path_in_repo=f"{ANCHOR_PREFIX}/{fname}",
            repo_id=args.repo_id,
            token=args.hf_token,
            commit_message=(
                f"Initialize DiLoCo anchor from {args.source_checkpoint} ({fname})"
            ),
        )

    completed_stages: List[int] = sorted(set(int(x) for x in args.completed_stages))
    initial_state = {
        "stage_k": int(args.stage_k),
        "round_n": int(args.round_n),
        "anchor_path": ANCHOR_PREFIX,
        "total_samples_seen": {str(int(args.stage_k)): int(args.samples_seen)},
        "completed_stages": completed_stages,
        "insufficient_worker_rounds": 0,
        "last_updated": 0,
    }
    upload_json(
        args.repo_id,
        ROUND_STATE_PATH,
        initial_state,
        args.hf_token,
        message=(
            f"Initialize DiLoCo round state: stage {args.stage_k} "
            f"round {args.round_n}"
        ),
    )

    print("[bootstrap] DiLoCo anchor and round_state.json initialized successfully.")


if __name__ == "__main__":
    main()
