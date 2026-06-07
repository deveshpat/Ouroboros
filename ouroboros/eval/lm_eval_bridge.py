"""lm-evaluation-harness launcher (OuroborosLM backend).

This bridge drives EleutherAI's harness with the custom OuroborosLM model class
instead of the stock HF/PEFT backend, fixing both README boundaries:

  Boundary 1 — OuroborosLM uses load_components + run_single_prompt; correct
               vocab size (65537 with <|lat|>), latent passes, optional HaltGate.
  Boundary 2 — OuroborosLM.__init__ calls ensure_environment() inside each
               accelerate worker subprocess when bootstrap=True is in model_args.

Three launch modes mirror the harness' own documented options. Each runs through
the ``ouroboros.eval.lm_eval_runner`` wrapper (which registers OuroborosLM before
delegating to the harness) rather than bare ``lm_eval``:

* single GPU         -> ``python -m ouroboros.eval.lm_eval_runner ... --device <dev>``
* data parallel (DP) -> ``accelerate launch -m ouroboros.eval.lm_eval_runner ...``
  (one full model copy per GPU, data split across them) -- ``--device`` must NOT be passed.
* model parallel (MP) -> ``python -m ouroboros.eval.lm_eval_runner ... --model_args ...,parallelize=True``
  (one model sharded across GPUs) -- run outside the accelerate launcher.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import pathlib
import subprocess
import sys
from pathlib import Path
from typing import Any

from ouroboros.utils.runtime_env import resolve_hf_token


# Curated task suites for AI21-Jamba-Reasoning-3B. Each suite fixes the tasks
# plus the conventional few-shot / chat-template / generation defaults so a
# score is meaningful and comparable. Explicit CLI flags always override these.
#   tasks            : comma-separated lm-eval task or group names
#   num_fewshot      : conventional shot count for this suite
#   apply_chat_template / fewshot_as_multiturn : sensible for instruct/reasoning
#   gen_kwargs       : required for generative tasks (e.g. gsm8k, ifeval)
TASK_SUITES: dict[str, dict[str, Any]] = {
    "smoke": {
        "tasks": "arc_easy",
        "num_fewshot": 0,
        "apply_chat_template": False,
        "fewshot_as_multiturn": False,
        "gen_kwargs": "",
    },
    "reasoning_core": {
        "tasks": "arc_challenge,hellaswag,winogrande,piqa,openbookqa",
        "num_fewshot": 0,
        "apply_chat_template": False,
        "fewshot_as_multiturn": False,
        "gen_kwargs": "",
    },
    "knowledge": {
        "tasks": "mmlu",
        "num_fewshot": 5,
        "apply_chat_template": False,
        "fewshot_as_multiturn": False,
        "gen_kwargs": "",
    },
    "math": {
        "tasks": "gsm8k",
        "num_fewshot": 5,
        "apply_chat_template": True,
        "fewshot_as_multiturn": True,
        "gen_kwargs": "max_gen_toks=512,do_sample=False,temperature=0.0",
    },
    "instruction": {
        "tasks": "ifeval",
        "num_fewshot": 0,
        "apply_chat_template": True,
        "fewshot_as_multiturn": False,
        "gen_kwargs": "max_gen_toks=1280,do_sample=False,temperature=0.0",
    },
    "truthful": {
        "tasks": "truthfulqa_mc2",
        "num_fewshot": 0,
        "apply_chat_template": False,
        "fewshot_as_multiturn": False,
        "gen_kwargs": "",
    },
    # Open-LLM-Leaderboard-v1-style mix. NOTE: the official leaderboard uses
    # mixed per-task few-shot (arc 25 / hella 10 / mmlu 5 / wino 5 / gsm8k 5).
    # A single CLI --num_fewshot cannot express that; this suite leaves shots at
    # the harness/per-task default and is therefore approximate.
    "leaderboard": {
        "tasks": "arc_challenge,hellaswag,mmlu,truthfulqa_mc2,winogrande,gsm8k",
        "num_fewshot": None,
        "apply_chat_template": True,
        "fewshot_as_multiturn": True,
        "gen_kwargs": "max_gen_toks=512,do_sample=False,temperature=0.0",
    },
}


def _resolve_adapter_for_lm_eval(args: argparse.Namespace) -> str | None:
    """Return a local PEFT adapter path that OuroborosLM can pass to load_components.

    When an adapter lives in a Hub subfolder, resolve that subfolder first with
    Hugging Face Hub so the model class receives a normal local PEFT adapter
    directory containing ``adapter_config.json``.
    """
    adapter = (args.adapter or "").strip()
    if not adapter:
        return None

    subfolder = (getattr(args, "adapter_subfolder", "") or "").strip().strip("/")
    local_adapter = Path(adapter).expanduser()
    if local_adapter.exists():
        resolved = local_adapter / subfolder if subfolder else local_adapter
    elif subfolder:
        from huggingface_hub import snapshot_download

        cache_dir = Path(getattr(args, "adapter_cache_dir", "") or "runs/lm_eval_adapter").expanduser()
        snapshot_download(
            repo_id=adapter,
            token=resolve_hf_token(env=os.environ),
            local_dir=str(cache_dir),
            allow_patterns=[f"{subfolder}/*"],
        )
        resolved = cache_dir / subfolder
    else:
        return adapter

    if not (resolved / "adapter_config.json").exists():
        raise SystemExit(f"No PEFT adapter_config.json found at {resolved}.")
    return str(resolved)


def resolve_eval_plan(args: argparse.Namespace) -> dict[str, Any]:
    """Merge an optional --suite preset with explicit CLI flags (flags win).

    Tri-state flags (``apply_chat_template``/``fewshot_as_multiturn``/
    ``log_samples``) arrive as None when unset so a suite default can apply.
    """
    suite_name = (getattr(args, "suite", "") or "").strip()
    suite = TASK_SUITES.get(suite_name, {}) if suite_name else {}
    if suite_name and not suite:
        raise SystemExit(
            f"Unknown --suite {suite_name!r}. Choices: {', '.join(sorted(TASK_SUITES))}."
        )

    tasks = (getattr(args, "tasks", "") or "").strip() or suite.get("tasks")
    if not tasks:
        raise SystemExit("Provide --tasks or a known --suite.")

    def _pick(flag: str, suite_key: str, fallback: Any) -> Any:
        val = getattr(args, flag, None)
        if val is not None:
            return val
        if suite_key in suite:
            return suite[suite_key]
        return fallback

    num_fewshot = getattr(args, "num_fewshot", None)
    if num_fewshot is None:
        num_fewshot = suite.get("num_fewshot")  # may stay None -> harness default

    return {
        "tasks": tasks,
        "num_fewshot": num_fewshot,
        "apply_chat_template": bool(_pick("apply_chat_template", "apply_chat_template", False)),
        "fewshot_as_multiturn": bool(_pick("fewshot_as_multiturn", "fewshot_as_multiturn", False)),
        "gen_kwargs": _pick("gen_kwargs", "gen_kwargs", "") or "",
        "log_samples": bool(_pick("log_samples", "log_samples", True)),
        "suite": suite_name or None,
    }


def _ouroboros_model_args(args: argparse.Namespace, resolved_adapter: str | None) -> str:
    """Build model_args string for OuroborosLM (replaces stock HF model_args).

    OuroborosLM.__init__ accepts keyword arguments parsed from this string by
    lm-eval's model_args parser. Device placement is handled internally by
    load_components; do not pass --device alongside this model class.
    """
    pairs: list[tuple[str, str]] = [
        ("base_model", args.model_id),
        ("stage_k", str(getattr(args, "stage_k", 10))),
        ("use_halt_gate", "True"),
        ("dtype", args.dtype or "auto"),
        ("max_seq_len", str(getattr(args, "max_seq_len", 4096))),
    ]
    if resolved_adapter:
        # Pass resolved local path so OuroborosLM skips the Hub download.
        pairs.append(("adapter_dir", resolved_adapter))
    if bool(args.load_in_4bit):
        pairs.append(("use_4bit", "True"))
    if getattr(args, "bootstrap", False):
        # Propagate into each accelerate worker; OuroborosLM.__init__ calls
        # ensure_environment() there — this is Boundary 2 fix.
        pairs.append(("bootstrap", "True"))
    raw = (getattr(args, "extra_model_args", "") or "").strip().strip(",")
    extra = f",{raw}" if raw else ""
    return ",".join(f"{k}={v}" for k, v in pairs) + extra


def build_command(args: argparse.Namespace, resolved_adapter: str | None, plan: dict[str, Any]) -> list[str]:
    """Build the harness command for single / data-parallel / model-parallel.

    Pure function (no I/O) so it can be unit-tested without a GPU or weights.
    """
    data_parallel = int(getattr(args, "data_parallel", 1) or 1)
    model_parallel = bool(getattr(args, "model_parallel", False))
    if data_parallel > 1 and model_parallel:
        # The harness does support DP x MP hybrid, but it is an advanced footgun
        # on small models; require an explicit opt-in flag to combine them.
        if not getattr(args, "allow_dp_mp_hybrid", False):
            raise SystemExit(
                "--data_parallel > 1 with --model_parallel is the DP x MP hybrid. "
                "Pass --allow_dp_mp_hybrid to confirm, or pick one."
            )

    use_accelerate = data_parallel > 1

    # Launch the harness via our wrapper module, not bare ``lm_eval``. The wrapper
    # imports lm_eval_model first, running @register_model("ouroboros") in this
    # process (and in every accelerate worker), so --model ouroboros resolves.
    entrypoint = "ouroboros.eval.lm_eval_runner"

    if use_accelerate:
        launcher = ["accelerate", "launch", "--multi_gpu", "--num_processes", str(data_parallel)]
        port = getattr(args, "main_process_port", None)
        if port:
            launcher += ["--main_process_port", str(port)]
        command = launcher + ["-m", entrypoint]
    else:
        command = [sys.executable, "-m", entrypoint]

    # --include_path makes any task-config YAMLs placed in the eval package
    # discoverable. (Model registration is handled by the wrapper above, not by
    # --include_path, which only loads task YAMLs — never Python registrations.)
    import ouroboros.eval as _ouro_eval_pkg
    _eval_dir = str(pathlib.Path(_ouro_eval_pkg.__file__).parent)

    command += [
        "--include_path", _eval_dir,
        "--model", "ouroboros",
        "--model_args", _ouroboros_model_args(args, resolved_adapter),
        "--tasks", plan["tasks"],
        "--batch_size", str(args.batch_size),
    ]

    # NOTE: --device is intentionally omitted. OuroborosLM.load_components()
    # resolves device internally; passing --device alongside the ouroboros model
    # class would be ignored or cause confusion.

    if plan.get("num_fewshot") is not None:
        command += ["--num_fewshot", str(plan["num_fewshot"])]
    if plan.get("apply_chat_template"):
        command += ["--apply_chat_template"]
        if plan.get("fewshot_as_multiturn"):
            command += ["--fewshot_as_multiturn"]
    system_instruction = (getattr(args, "system_instruction", "") or "").strip()
    if system_instruction:
        command += ["--system_instruction", system_instruction]
    if plan.get("gen_kwargs"):
        command += ["--gen_kwargs", plan["gen_kwargs"]]
    seed = (getattr(args, "seed", "") or "").strip()
    if seed:
        command += ["--seed", seed]
    if args.trust_remote_code:
        command += ["--trust_remote_code"]
    if args.limit:
        command += ["--limit", str(args.limit)]
    if args.output_path:
        command += ["--output_path", args.output_path]
        if plan.get("log_samples"):
            command += ["--log_samples"]
    return command


def _preflight(args: argparse.Namespace) -> None:
    """Fail fast before a multi-GPU run instead of crashing mid-eval.

    Boundary 2 note: bootstrap in the parent process here ensures the on-disk
    Mamba wheel + Triton source patch exist before accelerate forks workers.
    OuroborosLM.__init__ then calls ensure_environment() again inside each worker
    (the actual Boundary 2 fix) to load the shims into that subprocess's address
    space.
    """
    if getattr(args, "bootstrap", False):
        from ouroboros.bootstrap import ensure_environment
        ensure_environment()
        # After bootstrap, continue preflight checks.

    if importlib.util.find_spec("lm_eval") is None:
        raise SystemExit(
            "lm-evaluation-harness not importable. Install with: pip install lm_eval>=0.4.5"
        )
    if int(getattr(args, "data_parallel", 1) or 1) > 1 and importlib.util.find_spec("accelerate") is None:
        raise SystemExit("--data_parallel needs accelerate. Install with: pip install accelerate>=1.0")

    # Light Mamba/Jamba fast-path check. The stock harness loads the model in a
    # fresh subprocess, so Ouroboros' import-time patches do not run there; the
    # on-disk wheel + source patch from a prior bootstrap must already be in the
    # env. Warn by default; hard-fail under --require_fast_path.
    if importlib.util.find_spec("mamba_ssm") is None:
        msg = (
            "[lm-eval] WARNING: mamba_ssm not importable; the Jamba/Mamba fast "
            "path is unavailable. On Kaggle run bootstrap first (set --bootstrap "
            "or run the notebook bootstrap cell)."
        )
        if getattr(args, "require_fast_path", False):
            raise SystemExit(msg.replace("WARNING", "ERROR"))
        print(msg)


def _write_run_config(
    args: argparse.Namespace, command: list[str], resolved_adapter: str | None, plan: dict[str, Any]
) -> None:
    if not args.output_path:
        return
    output = Path(args.output_path)
    output.mkdir(parents=True, exist_ok=True)
    if int(getattr(args, "data_parallel", 1) or 1) > 1:
        launch_mode = f"data_parallel x{args.data_parallel} (accelerate launch)"
    elif getattr(args, "model_parallel", False):
        launch_mode = "model_parallel (parallelize=True)"
    else:
        launch_mode = "single_process"
    config: dict[str, Any] = {
        "runtime": "ouroboros custom lm-eval model (OuroborosLM)",
        "launch_mode": launch_mode,
        "model_id": args.model_id,
        "adapter": args.adapter,
        "adapter_subfolder": getattr(args, "adapter_subfolder", ""),
        "resolved_adapter": resolved_adapter,
        "suite": plan.get("suite"),
        "tasks": plan["tasks"],
        "num_fewshot": plan.get("num_fewshot"),
        "apply_chat_template": plan.get("apply_chat_template"),
        "fewshot_as_multiturn": plan.get("fewshot_as_multiturn"),
        "gen_kwargs": plan.get("gen_kwargs"),
        "limit": args.limit,
        "batch_size": args.batch_size,
        "device": "auto (OuroborosLM resolves internally)",
        "dtype": args.dtype,
        "load_in_4bit": bool(args.load_in_4bit),
        "stage_k": getattr(args, "stage_k", 10),
        "max_seq_len": getattr(args, "max_seq_len", 4096),
        "command": command,
        "boundary_1": "OuroborosLM: load_components path with <|lat|> + latent passes.",
        "boundary_2": "OuroborosLM: bootstrap runs in each accelerate worker.",
    }
    (output / "ouroboros_lm_eval_run_config.json").write_text(
        json.dumps(config, indent=2), encoding="utf-8"
    )


def run_lm_eval_hf(args: argparse.Namespace) -> None:
    plan = resolve_eval_plan(args)
    _preflight(args)
    resolved_adapter = _resolve_adapter_for_lm_eval(args)
    command = build_command(args, resolved_adapter, plan)
    _write_run_config(args, command, resolved_adapter, plan)

    env = os.environ.copy()
    if args.trust_remote_code:
        # Some task datasets gate download behind this env in recent datasets.
        env.setdefault("HF_DATASETS_TRUST_REMOTE_CODE", "1")

    print("[lm-eval] " + " ".join(command))
    subprocess.run(command, check=True, env=env)


__all__ = ("run_lm_eval_hf", "build_command", "resolve_eval_plan", "TASK_SUITES")
