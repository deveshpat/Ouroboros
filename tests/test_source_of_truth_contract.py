from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]


def test_training_entrypoint_has_graduated_to_compatibility_adapter():
    training_adapter = REPO_ROOT / "jamba_coconut_finetune.py"
    modular_train = REPO_ROOT / "ouroboros" / "train.py"

    assert training_adapter.exists(), "training root entrypoint must remain runnable"
    assert modular_train.exists(), "packaged training module owns extracted behavior"

    adapter_source = training_adapter.read_text(encoding="utf-8")
    modular_source = modular_train.read_text(encoding="utf-8")

    assert "def main(" in adapter_source
    assert "from ouroboros.train import run_cli" in adapter_source
    assert "def evaluate_stage(" not in adapter_source
    assert "def run_generation_callback(" not in adapter_source
    assert "def run_training_stages(" in modular_source
    assert "def run_cli(" in modular_source


def test_coordinator_entrypoint_has_graduated_to_compatibility_adapter():
    coordinator_adapter = REPO_ROOT / "diloco_coordinator.py"
    packaged_coordinator = REPO_ROOT / "ouroboros" / "diloco" / "coordinator.py"

    assert coordinator_adapter.exists(), "coordinator root entrypoint must remain runnable"
    assert packaged_coordinator.exists(), "packaged coordinator module owns extracted behavior"

    adapter_source = coordinator_adapter.read_text(encoding="utf-8")
    packaged_source = packaged_coordinator.read_text(encoding="utf-8")

    assert "from ouroboros.diloco.coordinator import main" in adapter_source
    assert "def weighted_average_deltas(" not in adapter_source
    assert "def trigger_kaggle_workers(" not in adapter_source
    assert "def parse_args(" in packaged_source
    assert "def main(" in packaged_source


def test_zero_drift_plan_and_adapter_transition_are_checked_in():
    zero_drift_plan = REPO_ROOT / "plans" / "zero-drift-monolith-extraction.md"
    adapter_plan = REPO_ROOT / "plans" / "monolith-adapter-thinning.md"
    coordinator_prd = REPO_ROOT / "plans" / "diloco-coordinator-zero-drift-extraction-prd.md"
    coordinator_plan = REPO_ROOT / "plans" / "diloco-coordinator-zero-drift-extraction-plan.md"

    assert zero_drift_plan.exists()
    assert adapter_plan.exists()
    assert coordinator_prd.exists()
    assert coordinator_plan.exists()

    zero_drift_text = zero_drift_plan.read_text(encoding="utf-8")
    adapter_text = adapter_plan.read_text(encoding="utf-8")
    coordinator_plan_text = coordinator_plan.read_text(encoding="utf-8")

    assert "Phase 1: Validation/OOM regression tracer bullet" in zero_drift_text
    assert "Kaggle GPU runs are final confidence validation only" in zero_drift_text
    assert "training root file is a compatibility adapter" in adapter_text
    assert "Kaggle notebook becomes the final thin adapter" in adapter_text
    assert "Phase 1: Aggregation characterization and extraction" in coordinator_plan_text
    assert "Phase 6: Root adapter thinning and guardrails" in coordinator_plan_text
