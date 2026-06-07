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

    budget = subparsers.add_parser(
        "audit-coconut-val-budget",
        help="CPU/tokenizer-only Coconut validation truncation audit without model loading.",
    )
    budget.add_argument("--data_dir", required=True)
    budget.add_argument("--tokenizer_model_id", required=True)
    budget.add_argument("--max_seq_len", type=int, default=1024)
    budget.add_argument("--stage_k", type=int, default=10)
    budget.add_argument("--limit_samples", type=int)
    budget.add_argument(
        "--sample_strategy",
        choices=("first", "longest"),
        default="first",
        help="When --limit_samples is set, choose the first rows or the rows with the largest token-budget requirement.",
    )
    budget.add_argument("--use_chat_template", action="store_true", default=True)
    budget.add_argument("--no_chat_template", dest="use_chat_template", action="store_false")
    budget.add_argument(
        "--allow_truncated_eval",
        action="store_true",
        help="Keep exit code 0 when the tokenizer-only audit detects prompt truncation.",
    )
    budget.add_argument("--output_dir")
    budget.set_defaults(_handler="audit_coconut_val_budget")

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
    compare.add_argument(
        "--disable_candidate_halt_gate",
        action="store_true",
        help=(
            "Verify/load the candidate adapter but run fixed-depth latent inference without "
            "consulting HaltGate. Use this for HaltGate-vs-fixed-depth ablations."
        ),
    )
    compare.add_argument("--gen_max_tokens", type=int, default=128)
    compare.add_argument("--stage_k", type=int, default=10)
    compare.add_argument(
        "--max_seq_len",
        type=int,
        default=8192,
        help=(
            "Bounded eval context length. Default 8192 for full release scoring after token-budget audit; "
            "use 512 for T4 smoke runs if needed."
        ),
    )
    compare.add_argument("--halt_threshold", type=float, default=0.5)
    compare.add_argument("--device", default="auto")
    compare.add_argument("--dtype", default="auto")
    compare.add_argument(
        "--model_device_map",
        choices=("single", "auto", "balanced", "balanced_low_0", "sequential"),
        default="balanced_low_0",
        help=(
            "Eval-only model placement. single preserves the old cuda:0 pinning; "
            "balanced_low_0/auto/balanced let Transformers shard model layers across visible CUDA GPUs."
        ),
    )
    compare.add_argument("--use_chat_template", action="store_true", default=True)
    compare.add_argument("--no_chat_template", dest="use_chat_template", action="store_false")
    compare.add_argument("--disable_mamba_kernels", action="store_true")
    compare.add_argument("--limit_samples", type=int)
    compare.add_argument(
        "--sample_strategy",
        choices=("first", "longest"),
        default="first",
        help="When --limit_samples is set, choose the first rows or the rows with the largest token-budget requirement.",
    )
    compare.add_argument(
        "--min_candidate_margin",
        type=float,
        default=0.0,
        help=(
            "Required candidate exact-match margin over baseline. The comparison writes "
            "artifacts, then exits non-zero if candidate < baseline + margin."
        ),
    )
    compare.add_argument(
        "--allow_candidate_regression",
        action="store_true",
        help="Keep exit code 0 even when the candidate underperforms the baseline.",
    )
    compare.add_argument(
        "--allow_truncated_eval",
        action="store_true",
        help=(
            "Keep exit code 0 when prompts are truncated. By default, artifacts are written "
            "and the command exits non-zero because the score is not clean."
        ),
    )
    compare.add_argument(
        "--cleanup_every_n_samples",
        type=int,
        default=25,
        help="Run Python/CUDA memory cleanup between eval rows every N samples. Use 0 to disable.",
    )
    compare.add_argument("--output_dir", required=True)
    compare.set_defaults(_handler="compare_coconut_val")

    lm_eval_hf = subparsers.add_parser(
        "lm-eval-hf",
        help="Run lm-evaluation-harness using the faithful OuroborosLM backend.",
    )
    lm_eval_hf.add_argument("--model_id", default="ai21labs/AI21-Jamba-Reasoning-3B")
    lm_eval_hf.add_argument("--adapter", help="Optional PEFT adapter repo id or local path.")
    lm_eval_hf.add_argument(
        "--adapter_subfolder",
        default="",
        help="Optional Hub/local subfolder containing adapter_config.json before handing off to lm-eval.",
    )
    lm_eval_hf.add_argument(
        "--adapter_cache_dir",
        default="runs/lm_eval_adapter",
        help="Local cache target when --adapter is a Hub repo and --adapter_subfolder is set.",
    )
    lm_eval_hf.add_argument(
        "--tasks",
        default=None,
        help="Comma-separated lm-eval tasks. Optional if --suite is given (suite tasks then apply).",
    )
    lm_eval_hf.add_argument(
        "--suite",
        default="",
        help=(
            "Curated task preset (sets tasks + conventional few-shot/chat/gen defaults). "
            "Choices: smoke, reasoning_core, knowledge, math, instruction, truthful, leaderboard. "
            "Explicit flags override the preset."
        ),
    )
    # ---- Ouroboros latent configuration ------------------------------------
    lm_eval_hf.add_argument(
        "--stage_k",
        type=int,
        default=10,
        help="Number of Coconut latent passes for OuroborosLM (default: 10).",
    )
    lm_eval_hf.add_argument(
        "--max_seq_len",
        type=int,
        default=4096,
        help="Context budget (tokens) for OuroborosLM inference (default: 4096).",
    )
    # ---- multi-GPU placement (mirrors lm-eval docs) ----
    lm_eval_hf.add_argument(
        "--data_parallel",
        type=int,
        default=1,
        metavar="N",
        help="N>1 launches via `accelerate launch -m lm_eval` with one full model copy per GPU (data split across them).",
    )
    lm_eval_hf.add_argument(
        "--model_parallel",
        action="store_true",
        help="Shard ONE model copy across visible GPUs (run outside accelerate via parallelize=True).",
    )
    lm_eval_hf.add_argument(
        "--allow_dp_mp_hybrid",
        action="store_true",
        help="Confirm the advanced DP x MP hybrid when both --data_parallel>1 and --model_parallel are set.",
    )
    lm_eval_hf.add_argument(
        "--main_process_port",
        type=int,
        help="Optional accelerate rendezvous port for --data_parallel runs (avoids clashes).",
    )
    # ---- benchmarking knobs that make scores meaningful/comparable ----
    lm_eval_hf.add_argument("--num_fewshot", type=int, help="Few-shot examples per task (overrides any suite default).")
    lm_eval_hf.add_argument(
        "--apply_chat_template",
        dest="apply_chat_template",
        action="store_const",
        const=True,
        default=None,
        help="Apply the model's chat template (recommended for the instruct/reasoning model).",
    )
    lm_eval_hf.add_argument(
        "--no_apply_chat_template", dest="apply_chat_template", action="store_const", const=False
    )
    lm_eval_hf.add_argument(
        "--fewshot_as_multiturn",
        dest="fewshot_as_multiturn",
        action="store_const",
        const=True,
        default=None,
        help="Render few-shot shots as chat turns (requires chat template).",
    )
    lm_eval_hf.add_argument(
        "--no_fewshot_as_multiturn", dest="fewshot_as_multiturn", action="store_const", const=False
    )
    lm_eval_hf.add_argument("--system_instruction", default="", help="Optional system prompt passed to the harness.")
    lm_eval_hf.add_argument(
        "--gen_kwargs",
        default=None,
        help="Generation kwargs for generative tasks, e.g. 'max_gen_toks=512,do_sample=False'.",
    )
    lm_eval_hf.add_argument("--seed", default="0", help="lm-eval seed (single int or 'a,b,c,d').")
    lm_eval_hf.add_argument(
        "--log_samples",
        dest="log_samples",
        action="store_const",
        const=True,
        default=None,
        help="Persist per-sample outputs for inspection (needs --output_path). Default on.",
    )
    lm_eval_hf.add_argument("--no_log_samples", dest="log_samples", action="store_const", const=False)
    # ---- runtime / fast-path preflight ----
    lm_eval_hf.add_argument(
        "--bootstrap",
        action="store_true",
        help="Run the full Ouroboros runtime bootstrap (cached Mamba wheels + triton patch) before launching.",
    )
    lm_eval_hf.add_argument(
        "--require_fast_path",
        action="store_true",
        help="Hard-fail (instead of warn) if the Jamba/Mamba fast path is unavailable.",
    )
    lm_eval_hf.add_argument("--limit", type=int)
    lm_eval_hf.add_argument("--batch_size", default="auto")
    lm_eval_hf.add_argument(
        "--device",
        default="cuda:0",
        help="Kept for backward compatibility; OuroborosLM resolves device internally.",
    )
    lm_eval_hf.add_argument("--dtype", default="float16")
    lm_eval_hf.add_argument("--load_in_4bit", action="store_true")
    lm_eval_hf.add_argument("--trust_remote_code", action="store_true", default=True)
    lm_eval_hf.add_argument("--no_trust_remote_code", dest="trust_remote_code", action="store_false")
    lm_eval_hf.add_argument("--extra_model_args", default="")
    lm_eval_hf.add_argument("--output_path")
    lm_eval_hf.set_defaults(_handler="lm_eval_hf")

    readiness = subparsers.add_parser(
        "gate-experiment-readiness",
        help="Read eval artifacts and decide whether architecture experimentation is unblocked.",
    )
    readiness.add_argument("--comparison_dir", required=True, help="Artifact directory from compare-coconut-val.")
    readiness.add_argument("--lm_eval_dir", default="", help="Optional artifact directory from lm-eval-hf.")
    readiness.add_argument(
        "--require_lm_eval",
        action="store_true",
        help="Block unless lm-evaluation-harness wrote JSON result artifacts.",
    )
    readiness.add_argument(
        "--allow_diagnostic_score",
        action="store_true",
        help="Warn instead of blocking when the generated-answer score is diagnostic rather than full-release clean.",
    )
    readiness.add_argument(
        "--allow_candidate_regression",
        action="store_true",
        help="Warn instead of blocking when the candidate misses the configured baseline margin.",
    )
    readiness.add_argument(
        "--allow_halt_gate_suspect",
        action="store_true",
        help="Warn instead of blocking when HaltGate looks collapsed to one-latent behavior.",
    )
    readiness.add_argument(
        "--allow_not_ready",
        action="store_true",
        help="Always exit 0 after printing/writing the readiness report.",
    )
    readiness.add_argument("--output_path", help="Optional JSON report path.")
    readiness.set_defaults(_handler="gate_experiment_readiness")

    return parser


def main(argv: Iterable[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(list(argv) if argv is not None else None)
    if not getattr(args, "command", None):
        parser.print_help()
        return

    # Heavy imports stay behind subcommands so --help remains weight/model safe.
    if args._handler == "lm_eval_hf":
        from ouroboros.eval.lm_eval_bridge import run_lm_eval_hf

        handler = run_lm_eval_hf
    elif args._handler == "gate_experiment_readiness":
        from ouroboros.eval.readiness import run_experiment_readiness_gate

        handler = run_experiment_readiness_gate
    else:
        from ouroboros.eval import coconut_val

        handler = getattr(coconut_val, args._handler)
    handler(args)
