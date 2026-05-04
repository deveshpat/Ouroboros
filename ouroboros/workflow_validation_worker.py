"""Tiny CPU-safe fake worker used by workflow validation command contracts."""

from __future__ import annotations

import argparse
import json

from ouroboros.workflow_validation import write_cpu_smoke_worker_status


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Write a local CPU-smoke worker status JSON")
    parser.add_argument("--worker_id", required=True, choices=["A", "B", "C"])
    parser.add_argument("--stage_k", type=int, required=True)
    parser.add_argument("--round_n", type=int, required=True)
    parser.add_argument("--status_path", required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    status = write_cpu_smoke_worker_status(
        worker_id=args.worker_id,
        stage_k=args.stage_k,
        round_n=args.round_n,
        status_path=args.status_path,
    )
    print(json.dumps(status, sort_keys=True))


if __name__ == "__main__":
    main()
