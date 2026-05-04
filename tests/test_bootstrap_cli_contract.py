from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
TRAINING_MONOLITH = REPO_ROOT / "jamba_coconut_finetune.py"
CLI_MODULE = REPO_ROOT / "ouroboros" / "cli.py"


def test_cli_module_import_is_bootstrap_safe_and_does_not_import_torch():
    env = os.environ.copy()
    env.setdefault("PYTHONPATH", str(REPO_ROOT))
    completed = subprocess.run(
        [
            sys.executable,
            "-c",
            "import sys; import ouroboros.cli; raise SystemExit('torch' in sys.modules)",
        ],
        cwd=str(REPO_ROOT),
        env=env,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        timeout=15,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr[:1000]


def test_critical_env_vars_are_set_before_train_imports():
    source = TRAINING_MONOLITH.read_text(encoding="utf-8")
    train_import = source.index("from ouroboros.train import run_cli")
    for env_name in (
        "PYTORCH_CUDA_ALLOC_CONF",
        "TORCH_NCCL_ASYNC_ERROR_HANDLING",
        "TOKENIZERS_PARALLELISM",
        "TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC",
        "NCCL_TIMEOUT",
    ):
        assert source.index(f'os.environ.setdefault("{env_name}"') < train_import
    assert "import torch" not in source


def test_bootstrap_call_stays_before_train_import_for_real_execution():
    source = TRAINING_MONOLITH.read_text(encoding="utf-8")
    help_guard = source.index('if any(arg in {"-h", "--help"}')
    bootstrap_call = source.index("    ensure_environment()")
    train_import = source.index("from ouroboros.train import run_cli")
    cli_help_call = source.index("print_bootstrap_free_help_and_exit", 0)
    assert cli_help_call < help_guard < bootstrap_call < train_import
    assert '{"-h", "--help"}' in source[help_guard:train_import]
    assert "from ouroboros.cli import print_bootstrap_free_help_and_exit" in source[:train_import]


def test_cli_help_exits_without_bootstrap_or_cuda_side_effects():
    env = os.environ.copy()
    env.setdefault("PYTHONPATH", str(REPO_ROOT))
    completed = subprocess.run(
        [sys.executable, str(TRAINING_MONOLITH), "--help"],
        cwd=str(REPO_ROOT),
        env=env,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        timeout=15,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr[:1000]
    assert "Jamba Reasoning 3B Coconut-Ouroboros fine-tuning" in completed.stdout
    assert "--diloco_worker_id" in completed.stdout
    assert "--val_batch_size" in completed.stdout
    assert "--gen_every_stage" in completed.stdout
    assert "pip install" not in completed.stdout + completed.stderr


def test_cli_defaults_and_worker_normalization_match_source_contract():
    source = CLI_MODULE.read_text(encoding="utf-8")
    for contract in (
        'parser.add_argument("--model_id", default=MODEL_ID)',
        'parser.add_argument("--max_seq_len", type=int, default=1024)',
        'parser.add_argument("--batch_size", type=int, default=2)',
        'parser.add_argument("--grad_accum", type=int, default=8)',
        'default=1.0,',
        '"--val_batch_size",',
        'default=1,',
        '_add_bool_arg(parser, "--gen_every_stage", True',
        'choices=["online", "offline", "disabled"]',
        'default="online"',
        'type=_parse_diloco_worker_id_cli',
        'choices=list(_VALID_DILOCO_WORKER_IDS)',
        'return worker_id',
        'raise argparse.ArgumentTypeError("DiLoCo worker id cannot be empty")',
        '"--resume_from_diloco_anchor",',
    ):
        assert contract in source

    monolith_source = TRAINING_MONOLITH.read_text(encoding="utf-8")
    assert "from ouroboros.cli import parse_args" in monolith_source
    assert "args = parse_args(argv)" in monolith_source
    assert "from ouroboros.train import run_cli" in monolith_source
