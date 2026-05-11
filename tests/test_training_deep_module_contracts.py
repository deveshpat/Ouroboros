from __future__ import annotations

import ast
from pathlib import Path

from ouroboros import train as train_module
from ouroboros.training import checkpointing, evaluation, session, stage_runner
from ouroboros import latent as latent_module

REPO_ROOT = Path(__file__).resolve().parents[1]


def _imported_modules(path: Path) -> set[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    modules: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            modules.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            modules.add(node.module)
    return modules


def test_training_deep_modules_expose_new_preferred_interfaces():
    assert checkpointing.save_checkpoint.__module__ == "ouroboros.training.checkpointing"
    assert checkpointing.load_checkpoint.__module__ == "ouroboros.training.checkpointing"
    assert checkpointing.prune_epoch_checkpoints.__module__ == "ouroboros.training.checkpointing"
    assert evaluation.evaluate_stage.__module__ == "ouroboros.training.evaluation"
    assert evaluation.run_generation_callback.__module__ == "ouroboros.training.evaluation"
    assert latent_module.prepare_latent_runtime.__module__ == "ouroboros.latent"
    assert latent_module.forward_latent_batch.__module__ == "ouroboros.latent"
    assert latent_module.decode_from_latent_context.__module__ == "ouroboros.latent"
    assert stage_runner.run_training_stages.__module__ == "ouroboros.training.stage_runner"
    assert stage_runner.make_timeout_checker.__module__ == "ouroboros.training.stage_runner"
    assert session.run_training_session.__module__ == "ouroboros.training.session"


def test_train_module_remains_compatibility_facade():
    assert train_module.save_checkpoint is checkpointing.save_checkpoint
    assert train_module.load_checkpoint is checkpointing.load_checkpoint
    assert train_module.evaluate_stage is evaluation.evaluate_stage
    assert train_module.run_generation_callback is evaluation.run_generation_callback
    assert train_module.run_training_stages is stage_runner.run_training_stages
    assert train_module.run_cli.__module__ == "ouroboros.train"

    train_source = (REPO_ROOT / "ouroboros" / "train.py").read_text(encoding="utf-8")
    assert "def run_cli(args: argparse.Namespace, *, script_start: float) -> None:" in train_source
    assert "def run_training_stages(" not in train_source
    assert "def evaluate_stage(" not in train_source
    assert "def save_checkpoint(" not in train_source


def test_worker_no_longer_imports_train_module():
    worker_path = REPO_ROOT / "ouroboros" / "diloco" / "worker.py"
    imported = _imported_modules(worker_path)
    assert "ouroboros.train" not in imported
    assert "ouroboros.training.stage_runner" in imported
    assert "ouroboros.training.evaluation" in imported


def test_stage_runner_has_no_diloco_or_train_dependency():
    imported = _imported_modules(REPO_ROOT / "ouroboros" / "training" / "stage_runner.py")
    forbidden = {
        "ouroboros.train",
        "ouroboros.diloco.worker",
        "ouroboros.diloco.coordinator",
    }
    assert imported.isdisjoint(forbidden)


def test_latent_execution_ownership_has_moved_out_of_evaluation_and_dgac():
    evaluation_source = (REPO_ROOT / "ouroboros" / "training" / "evaluation.py").read_text(encoding="utf-8")
    dgac_source = (REPO_ROOT / "ouroboros" / "dgac.py").read_text(encoding="utf-8")
    latent_source = (REPO_ROOT / "ouroboros" / "latent.py").read_text(encoding="utf-8")

    for forbidden in [
        "_get_backbone",
        "_get_embed_tokens",
        "_get_lm_head",
        "_autocast_ctx",
        "_extract_last_hidden_state",
        "_run_latent_passes",
    ]:
        assert forbidden not in evaluation_source

    assert "def run_latent_passes(" in latent_source
    assert "def forward_latent_batch(" in latent_source
    assert "def decode_from_latent_context(" in latent_source
    assert "def _build_question_context(" not in dgac_source
    assert "def _collect_latent_hidden_sequences(" not in dgac_source
    assert "def _compute_ce_sum_and_count(" not in dgac_source
    assert "def _compute_ce_mean_by_row(" not in dgac_source
