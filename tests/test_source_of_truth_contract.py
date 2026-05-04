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


def test_coordinator_monolith_remains_substantial_before_extraction():
    coordinator_source = (REPO_ROOT / "diloco_coordinator.py").read_text(encoding="utf-8")

    assert "def parse_args(" in coordinator_source
    assert "def main(" in coordinator_source
    assert len(coordinator_source.splitlines()) > 500, "coordinator extraction is still deferred"


def test_zero_drift_plan_and_adapter_transition_are_checked_in():
    zero_drift_plan = REPO_ROOT / "plans" / "zero-drift-monolith-extraction.md"
    adapter_plan = REPO_ROOT / "plans" / "monolith-adapter-thinning.md"

    assert zero_drift_plan.exists()
    assert adapter_plan.exists()

    zero_drift_text = zero_drift_plan.read_text(encoding="utf-8")
    adapter_text = adapter_plan.read_text(encoding="utf-8")

    assert "Phase 1: Validation/OOM regression tracer bullet" in zero_drift_text
    assert "Kaggle GPU runs are final confidence validation only" in zero_drift_text
    assert "training root file is a compatibility adapter" in adapter_text
    assert "Kaggle notebook becomes the final thin adapter" in adapter_text
