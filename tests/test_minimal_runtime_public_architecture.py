"""Contract tests for the minimal seven-package Ouroboros runtime."""

from __future__ import annotations

import importlib
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
PUBLIC_PACKAGES = ("bootstrap", "coconut", "models", "inference", "eval", "coordinator", "utils")
RETIRED_ROOT_MODULES = (
    "azure_cost_guard.py",
    "benchmark_harness.py",
    "benchmark_multi_gpu.py",
    "cli.py",
    "coordinator_decision.py",
    "data.py",
    "dgac.py",
    "hard_lesson_guardrails.py",
    "hub.py",
    "inference.py",
    "kaggle.py",
    "kaggle_contract.py",
    "kaggle_launch_matrix.py",
    "kaggle_runtime.py",
    "latent.py",
    "lm_eval_bootstrap.py",
    "mac_dgac_fallback.py",
    "mac_jamba_fastpath.py",
    "model.py",
    "runtime_env.py",
    "train.py",
    "training_plan.py",
    "wandb_runtime.py",
    "worker_lifecycle.py",
    "workflow_validation.py",
    "workflow_validation_worker.py",
)


def test_seven_public_package_roots_exist_and_import_bootstrap_safe():
    for name in PUBLIC_PACKAGES:
        package = importlib.import_module(f"ouroboros.{name}")
        assert hasattr(package, "__all__"), f"ouroboros.{name} must define its public interface"


def test_root_workflow_adapters_are_removed_from_public_surface():
    assert not (REPO_ROOT / "jamba_coconut_finetune.py").exists()
    assert not (REPO_ROOT / "diloco_coordinator.py").exists()


def test_old_root_modules_are_collapsed_into_owning_packages():
    old_paths = [REPO_ROOT / "ouroboros" / filename for filename in RETIRED_ROOT_MODULES]
    assert [str(path.relative_to(REPO_ROOT)) for path in old_paths if path.exists()] == []


def test_operator_surface_uses_package_modules_not_root_wrappers():
    workflow = (REPO_ROOT / ".github" / "workflows" / "diloco_coordinator.yml").read_text(encoding="utf-8")
    assert "python -m ouroboros.coordinator" in workflow
    assert "python diloco_coordinator.py" not in workflow
    assert "python jamba_coconut_finetune.py" not in workflow


def test_blueprint_names_the_current_seven_package_model_only():
    text = (REPO_ROOT / "BLUEPRINT.md").read_text(encoding="utf-8")
    for name in ("Bootstrap", "Coconut", "Models", "Inference", "Eval", "Coordinator", "Utils"):
        assert name in text
    assert "thin compatibility adapter" not in text.lower()
    assert "CPU workflow validation" not in text
