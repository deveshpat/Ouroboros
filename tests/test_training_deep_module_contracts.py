from __future__ import annotations

import ast
from pathlib import Path

from ouroboros import coconut
from ouroboros.coconut import checkpointing, evaluation, session, stage_runner
from ouroboros.coconut import latent as latent_module

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


def test_coconut_package_exposes_training_latent_dgac_checkpoint_interface():
    assert coconut.save_checkpoint is checkpointing.save_checkpoint
    assert coconut.load_checkpoint is checkpointing.load_checkpoint
    assert coconut.evaluate_stage is evaluation.evaluate_stage
    assert coconut.run_generation_callback is evaluation.run_generation_callback
    assert coconut.run_training_stages is stage_runner.run_training_stages
    assert coconut.run_cli.__module__ == "ouroboros.coconut.runner"
    assert coconut.HaltGate.__module__ == "ouroboros.coconut.dgac"
    assert coconut.prepare_latent_runtime is latent_module.prepare_latent_runtime
    assert coconut.forward_latent_batch is latent_module.forward_latent_batch
    assert coconut.decode_from_latent_context is latent_module.decode_from_latent_context


def test_worker_imports_coconut_internals_not_retired_root_training_module():
    worker_path = REPO_ROOT / "ouroboros" / "coordinator" / "worker.py"
    imported = _imported_modules(worker_path)
    assert "ouroboros.train" not in imported
    assert "ouroboros.coconut.stage_runner" in imported
    assert "ouroboros.coconut.evaluation" in imported


def test_stage_runner_has_no_coordinator_dependency():
    imported = _imported_modules(REPO_ROOT / "ouroboros" / "coconut" / "stage_runner.py")
    forbidden = {
        "ouroboros.coordinator.worker",
        "ouroboros.coordinator.coordinator",
        "ouroboros.coordinator.dispatch",
    }
    assert imported.isdisjoint(forbidden)


def test_latent_execution_ownership_has_dedicated_coconut_module():
    evaluation_source = (REPO_ROOT / "ouroboros" / "coconut" / "evaluation.py").read_text(encoding="utf-8")
    dgac_source = (REPO_ROOT / "ouroboros" / "coconut" / "dgac.py").read_text(encoding="utf-8")
    latent_source = (REPO_ROOT / "ouroboros" / "coconut" / "latent.py").read_text(encoding="utf-8")

    for forbidden in [
        "_get_backbone",
        "_get_embed_tokens",
        "_get_lm_head",
        "_autocast_ctx",
        "_extract_last_hidden_state",
    ]:
        assert forbidden not in evaluation_source

    assert "def run_latent_passes(" in latent_source
    assert "def forward_latent_batch(" in latent_source
    assert "def decode_from_latent_context(" in latent_source
    assert "def _build_question_context(" not in dgac_source
    assert "def _collect_latent_hidden_sequences(" not in dgac_source
    assert "def _compute_ce_sum_and_count(" not in dgac_source
    assert "def _compute_ce_mean_by_row(" not in dgac_source
