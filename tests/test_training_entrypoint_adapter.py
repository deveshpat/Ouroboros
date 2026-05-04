from __future__ import annotations

import ast
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
TRAINING_ADAPTER = REPO_ROOT / "jamba_coconut_finetune.py"
CLI_MODULE = REPO_ROOT / "ouroboros" / "cli.py"
TRAIN_MODULE = REPO_ROOT / "ouroboros" / "train.py"


_FORBIDDEN_TRAINING_SYMBOLS = (
    "def evaluate_stage(",
    "def run_generation_callback(",
    "def run_training_stages(",
    "def run_diloco_worker(",
    "class HaltGate",
    "AutoModelForCausalLM",
    "AdamW",
)


def test_root_training_file_is_a_thin_runtime_adapter():
    source = TRAINING_ADAPTER.read_text(encoding="utf-8")

    assert len(source.splitlines()) < 90
    for forbidden in _FORBIDDEN_TRAINING_SYMBOLS:
        assert forbidden not in source

    module = ast.parse(source)
    top_level_functions = [node.name for node in module.body if isinstance(node, ast.FunctionDef)]
    assert top_level_functions == ["_print_bootstrap_free_help_and_exit", "main"]


def test_adapter_preserves_bootstrap_before_train_import_ordering():
    source = TRAINING_ADAPTER.read_text(encoding="utf-8")

    first_env = source.index('os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF"')
    script_start = source.index("_SCRIPT_START = time.perf_counter()")
    help_guard = source.index('if any(arg in {"-h", "--help"}')
    ensure_environment = source.index("    ensure_environment()")
    train_import = source.index("from ouroboros.train import run_cli")
    run_cli_call = source.index("run_cli(args, script_start=_SCRIPT_START)")

    assert first_env < script_start < help_guard < ensure_environment < train_import < run_cli_call
    assert "import torch" not in source


def test_adapter_delegates_cli_and_training_to_packaged_modules():
    source = TRAINING_ADAPTER.read_text(encoding="utf-8")
    cli_source = CLI_MODULE.read_text(encoding="utf-8")
    train_source = TRAIN_MODULE.read_text(encoding="utf-8")

    assert "from ouroboros.cli import parse_args" in source
    assert "from ouroboros.train import run_cli" in source
    assert "def parse_args(argv: Optional[Sequence[str]] = None)" in cli_source
    assert "return parser.parse_args(argv)" in cli_source
    assert "def run_cli(args: argparse.Namespace, *, script_start: float) -> None:" in train_source


def test_adapter_help_remains_bootstrap_free_after_thinning():
    completed = subprocess.run(
        [sys.executable, str(TRAINING_ADAPTER), "--help"],
        cwd=str(REPO_ROOT),
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        timeout=15,
        check=False,
    )

    assert completed.returncode == 0, completed.stderr[:1000]
    assert "Jamba Reasoning 3B Coconut-Ouroboros fine-tuning" in completed.stdout
    assert "--resume_from_diloco_anchor" in completed.stdout
    assert "pip install" not in completed.stdout + completed.stderr
