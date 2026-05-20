"""Bootstrap-safe CLI for Ouroboros evaluation artifacts."""

from __future__ import annotations

import argparse
from collections.abc import Iterable


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m ouroboros.eval",
        description="Create Ouroboros Coconut validation artifacts and comparisons.",
    )
    subparsers = parser.add_subparsers(dest="command")

    inspect_cmd = subparsers.add_parser(
        "inspect-coconut-val",
        help="Inspect a local Coconut validation JSONL without model loading.",
    )
    inspect_cmd.add_argument("--data_dir", required=True)
    inspect_cmd.set_defaults(_handler="inspect_coconut_val")

    dry_run = subparsers.add_parser(
        "dry-run-coconut-val",
        help="Write Coconut validation run_config/summary artifacts without model loading.",
    )
    dry_run.add_argument("--data_dir", required=True)
    dry_run.add_argument("--dataset_repo", required=True)
    dry_run.add_argument("--dataset_config", required=True)
    dry_run.add_argument("--dataset_split", required=True)
    dry_run.add_argument("--dataset_revision", required=True)
    dry_run.add_argument("--output_dir", required=True)
    dry_run.set_defaults(_handler="dry_run_coconut_val")

    compare = subparsers.add_parser(
        "compare-coconut-val",
        help="Run generated-answer base-vs-Ouroboros Coconut validation comparison.",
    )
    compare.add_argument("--data_dir", required=True)
    compare.add_argument("--dataset_repo", required=True)
    compare.add_argument("--dataset_config", required=True)
    compare.add_argument("--dataset_split", required=True)
    compare.add_argument("--dataset_revision", required=True)
    compare.add_argument("--baseline_model_id", required=True)
    compare.add_argument("--candidate_repo_id", required=True)
    compare.add_argument("--candidate_subdir", default="")
    compare.add_argument("--candidate_adapter_dir")
    compare.add_argument("--candidate_requires_halt_gate", action="store_true")
    compare.add_argument("--gen_max_tokens", type=int, default=128)
    compare.add_argument("--stage_k", type=int, default=10)
    compare.add_argument("--max_seq_len", type=int, default=512)
    compare.add_argument("--halt_threshold", type=float, default=0.5)
    compare.add_argument("--device", default="auto")
    compare.add_argument("--dtype", default="auto")
    compare.add_argument("--use_chat_template", action="store_true", default=True)
    compare.add_argument("--no_chat_template", dest="use_chat_template", action="store_false")
    compare.add_argument("--disable_mamba_kernels", action="store_true")
    compare.add_argument("--limit_samples", type=int)
    compare.add_argument("--output_dir", required=True)
    compare.set_defaults(_handler="compare_coconut_val")

    return parser


def main(argv: Iterable[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(list(argv) if argv is not None else None)
    if not getattr(args, "command", None):
        parser.print_help()
        return

    # Heavy imports stay behind subcommands so --help remains weight/model safe.
    from ouroboros.eval import coconut_val

    handler = getattr(coconut_val, args._handler)
    handler(args)
