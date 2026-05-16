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
    stage_runner_source = (REPO_ROOT / "ouroboros" / "training" / "stage_runner.py").read_text(encoding="utf-8")
    evaluation_source = (REPO_ROOT / "ouroboros" / "training" / "evaluation.py").read_text(encoding="utf-8")

    assert "def evaluate_stage(" not in adapter_source
    assert "def run_generation_callback(" not in adapter_source
    assert "def run_training_stages(" not in modular_source
    assert "def run_cli(" in modular_source
    assert "def run_training_stages(" in stage_runner_source
    assert "def evaluate_stage(" in evaluation_source
    assert "def run_generation_callback(" in evaluation_source


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


def test_completed_extraction_plans_are_promoted_to_wiki_and_retired():
    architecture_record = REPO_ROOT / "wiki" / "Architecture-Extraction.md"
    workflow_record = REPO_ROOT / "wiki" / "Engineering-Workflow.md"

    assert architecture_record.exists()
    assert workflow_record.exists()

    architecture_text = architecture_record.read_text(encoding="utf-8")
    workflow_text = workflow_record.read_text(encoding="utf-8")

    assert "Completed Track: Training Monolith Extraction" in architecture_text
    assert "Kaggle launch remains IPython shell magic (`!{shell_command}`)" in architecture_text
    assert "Completed Track: Coordinator Zero-Drift Extraction" in architecture_text
    assert "`plans/zero-drift-monolith-extraction.md`" in architecture_text
    assert "`plans/diloco-coordinator-zero-drift-extraction-plan.md`" in architecture_text
    assert "latest PRD" in workflow_text
    assert "choose one tracer bullet" in workflow_text
    assert "Delete obsolete files from `prds/` and `plans/`" in workflow_text

    obsolete_paths = [
        REPO_ROOT / "plans" / "zero-drift-monolith-extraction.md",
        REPO_ROOT / "plans" / "monolith-adapter-thinning.md",
        REPO_ROOT / "plans" / "diloco-coordinator-zero-drift-extraction-prd.md",
        REPO_ROOT / "plans" / "diloco-coordinator-zero-drift-extraction-plan.md",
    ]
    assert not any(path.exists() for path in obsolete_paths)


def test_runtime_signal_artifacts_are_ignored_but_signal_directory_is_kept():
    gitignore = (REPO_ROOT / ".gitignore").read_text(encoding="utf-8")

    assert "signals/*.json" in gitignore
    assert "!signals/.gitkeep" in gitignore
    assert (REPO_ROOT / "signals" / ".gitkeep").exists()
    assert list((REPO_ROOT / "signals").glob("*.json")) == []


def test_runtime_signal_mechanism_is_retained_as_coordinator_doorbell():
    workflow = (REPO_ROOT / ".github" / "workflows" / "diloco_coordinator.yml").read_text(encoding="utf-8")
    worker_source = (REPO_ROOT / "ouroboros" / "diloco" / "worker.py").read_text(encoding="utf-8")

    assert "signals/*.json" in workflow
    assert "schedule:" not in workflow
    assert "cron:" not in workflow
    assert "def diloco_push_signal" in worker_source
    assert "signals/worker_{worker_id}_stage_{stage_k}_round_{round_n}.json" in worker_source


def test_workflow_dispatch_exposes_cpu_smoke_end_to_end_validation_gate():
    workflow = (REPO_ROOT / ".github" / "workflows" / "diloco_coordinator.yml").read_text(encoding="utf-8")

    assert "workflow_validate" in workflow
    assert "cpu-smoke" in workflow
    assert "OUROBOROS_WORKFLOW_VALIDATE" in workflow
    assert "OUROBOROS_WORKFLOW_VALIDATION_RUN_ID" in workflow
    assert "--workflow_validate" in workflow
    assert "--workflow_validation_run_id" in workflow



def test_workflow_dispatch_exposes_dgac_anchor_eval_and_train_gates():
    workflow = (REPO_ROOT / ".github" / "workflows" / "diloco_coordinator.yml").read_text(encoding="utf-8")
    notebook = (REPO_ROOT / "kaggle-utils.ipynb").read_text(encoding="utf-8")
    launch_matrix = (REPO_ROOT / "ouroboros" / "kaggle_launch_matrix.py").read_text(encoding="utf-8")
    kaggle_commands = (REPO_ROOT / "ouroboros" / "kaggle.py").read_text(encoding="utf-8")

    assert "kaggle_run_mode" in workflow
    assert "dgac-anchor-eval" in workflow
    assert "dgac-train" in workflow
    assert "dgac-diloco" in workflow
    assert "OUROBOROS_KAGGLE_RUN_MODE" in workflow
    assert "--kaggle_run_mode" in workflow
    assert "OUROBOROS_KAGGLE_RUN_MODE" in notebook
    assert "build_launch_command(run_mode, os.environ, worker_id=worker_id)" in notebook
    assert "!{shell_command}" in notebook
    assert "--resume_from_diloco_anchor" in kaggle_commands
    assert "--eval_only" in kaggle_commands
    assert "runs/stage3_dgac" in launch_matrix
    assert "DGAC_DILOCO_RUN_MODE" in launch_matrix
    assert "runs/dgac_dedicated" in launch_matrix


def test_completed_cpu_smoke_prd_and_plan_are_promoted_to_wiki_and_retired():
    workflow_validation_record = REPO_ROOT / "wiki" / "Kaggle-CPU-API-Workflow-Validation.md"
    status_record = REPO_ROOT / "wiki" / "STATUS.md"

    assert workflow_validation_record.exists()
    assert status_record.exists()

    workflow_validation_text = workflow_validation_record.read_text(encoding="utf-8")
    status_text = status_record.read_text(encoding="utf-8")

    assert "Durable record for the completed PRD" in workflow_validation_text
    assert "Live Gate Evidence" in workflow_validation_text
    assert "coordinate #272" in workflow_validation_text
    assert "diloco_state/workflow_validation/25377312407-1/" in workflow_validation_text
    assert "removed from `prds/` and `plans/`" in status_text

    obsolete_paths = [
        REPO_ROOT / "prds" / "dgac-readiness-cpu-smoke.md",
        REPO_ROOT / "plans" / "dgac-readiness-cpu-smoke.md",
    ]
    assert not any(path.exists() for path in obsolete_paths)

def test_stage_10_terminal_gate_is_reflected_in_source_of_truth_docs():
    blueprint = (REPO_ROOT / "BLUEPRINT.md").read_text(encoding="utf-8")
    status = (REPO_ROOT / "wiki" / "STATUS.md").read_text(encoding="utf-8")
    session_log = (REPO_ROOT / "wiki" / "SessionLog.md").read_text(encoding="utf-8")
    terminal_log = (REPO_ROOT / "terminal_log.md").read_text(encoding="utf-8")

    assert "| Stage 10 | ✅ COMPLETE" in blueprint
    assert "| DGAC | ✅ COMPLETE" in blueprint
    assert "36,906/36,906" in blueprint
    assert "waiting on Kaggle GPU quota" not in blueprint

    curriculum_section = status.split("## Curriculum Progress", 1)[1].split("## Engineering Architecture Status", 1)[0]
    assert "| 10 — 10 latent passes | ✅ COMPLETE" in curriculum_section
    assert "36,906/36,906" in curriculum_section
    assert "Current gate" in curriculum_section
    assert "IN PROGRESS" not in curriculum_section
    assert "Current blocker" not in curriculum_section

    assert "mode=dgac-complete" in status
    assert "dgac_diloco_complete=true" in status
    assert "Stage 10 terminal aggregation" in status
    assert "Stage 10 terminal anchor eval-only" in status
    assert "post-DGAC `dgac-anchor-eval`" in status
    assert "Stage 10 terminal aggregation → DGAC manual gate" in session_log
    assert "Stage 10 terminal anchor eval-only → DGAC cleared" in session_log
    assert "Azure H100 corrected DGAC epoch-0 checkpoint" in terminal_log
    assert "Loaded halt gate from diloco_state/anchor/halt_gate.pt" in terminal_log
    assert "runs/azure_h100_dgac/stage_10/checkpoint-0001154" in terminal_log
