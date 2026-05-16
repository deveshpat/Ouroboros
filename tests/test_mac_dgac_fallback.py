from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

from ouroboros.coordinator import coordinator
from ouroboros.coordinator.mac_dgac_fallback import (
    CANARY_LATEST_DIRNAME,
    DEFAULT_MAC_DGAC_WORKERS,
    MAC_DGAC_CLAIM_PATH,
    ExpectedRoundState,
    MacRuntimeProbe,
    build_mac_failed_claim,
    build_mac_failure_round_state,
    build_local_dgac_aggregation_command,
    build_local_dgac_canary_command,
    build_local_dgac_worker_command,
    build_mac_controlled_round_state,
    evaluate_mac_preflight,
    is_active_mac_claim,
    promote_canary_output,
    prune_canary_outputs,
)
from ouroboros.models import _amp_dtype, _autocast_ctx


def _round_state(**overrides):
    state = {
        "stage_k": 10,
        "round_n": 3,
        "mode": "waiting",
        "triggered_workers": [],
        "attendance_workers": ["A", "B", "C"],
        "triggered_at": 1000.0,
        "anchor_path": "diloco_state/anchor",
        "projected_shards": {"A": 4475, "B": 4475, "C": 4475},
        "total_samples_seen": {"10": 23481},
        "completed_stages": [10],
        "dgac_diloco": True,
        "seed": 42,
    }
    state.update(overrides)
    return state


def _passing_probe(**overrides):
    probe = {
        "platform_system": "Darwin",
        "platform_machine": "arm64",
        "mps_available": True,
        "cuda_available": False,
        "mamba_ssm_macos_available": True,
        "mamba_probe_passed": True,
        "jamba_forward_backward_passed": True,
        "anchor_adapter_loaded": True,
        "halt_gate_loaded": True,
        "use_4bit_requested": False,
    }
    probe.update(overrides)
    return MacRuntimeProbe(**probe)


def test_default_mac_worker_set_is_single_local_worker():
    assert DEFAULT_MAC_DGAC_WORKERS == ("A",)


def _args(**overrides):
    values = dict(
        repo_id="fake/repo",
        hf_token="hf_fake",
        min_shard_samples=1,
        outer_lr=0.7,
        worker_timeout_hours=13.0,
        attendance_join_grace_minutes=5.0,
        force_worker_ids=None,
        kaggle_username_a="user-a",
        kaggle_key_a="key-a",
        kaggle_username_b="user-b",
        kaggle_key_b="key-b",
        kaggle_username_c="user-c",
        kaggle_key_c="key-c",
        kaggle_notebook_path=str(Path("kaggle-utils.ipynb")),
        skip_trigger=False,
        dry_run=False,
        wandb_key=None,
        wandb_project="project",
        wandb_entity=None,
        total_train_samples=36906,
        workflow_validate=None,
        workflow_validation_run_id=None,
        workflow_validation_timeout_s=900.0,
        workflow_validation_poll_s=10.0,
        kaggle_run_mode="dgac-diloco",
        mac_claim_id=None,
    )
    values.update(overrides)
    return argparse.Namespace(**values)


def test_mac_preflight_allows_canonical_writes_only_after_strict_runtime_and_no_drift():
    report = evaluate_mac_preflight(
        state=_round_state(),
        expected=ExpectedRoundState(),
        runtime_probe=_passing_probe(),
    )

    assert report.canonical_writes_allowed is True
    assert report.quantization == "mps-fp16-lora-no-4bit"
    assert report.parity_classification == "canonical-write-eligible"
    assert report.reasons == []


def test_mac_training_dtype_is_fp16_on_mps_to_avoid_strict_fallback_oom():
    assert _amp_dtype(torch.device("mps")) == torch.float16


def test_mac_training_autocast_uses_mps_when_available(monkeypatch):
    calls = []

    class DummyContext:
        def __enter__(self):
            return None

        def __exit__(self, exc_type, exc, tb):
            return False

    monkeypatch.setattr(
        torch.amp.autocast_mode,
        "is_autocast_available",
        lambda device_type: device_type == "mps",
    )
    monkeypatch.setattr(
        torch,
        "autocast",
        lambda *, device_type, dtype: calls.append((device_type, dtype)) or DummyContext(),
    )

    with _autocast_ctx(torch.device("mps"), torch.float16):
        pass

    assert calls == [("mps", torch.float16)]


def test_mac_preflight_refuses_when_live_round_state_drifted():
    report = evaluate_mac_preflight(
        state=_round_state(total_samples_seen={"10": 24000}),
        expected=ExpectedRoundState(),
        runtime_probe=_passing_probe(),
    )

    assert report.canonical_writes_allowed is False
    assert report.parity_classification == "refuse-canonical-write"
    assert any("total_samples_seen[10]" in reason for reason in report.reasons)


def test_mac_preflight_refuses_4bit_quantization_on_mps():
    report = evaluate_mac_preflight(
        state=_round_state(),
        expected=ExpectedRoundState(),
        runtime_probe=_passing_probe(use_4bit_requested=True),
    )

    assert report.canonical_writes_allowed is False
    assert report.quantization == "unsupported-4bit-on-mps"
    assert any("--use_4bit" in reason for reason in report.reasons)


def test_mac_controlled_round_state_reactivates_waiting_workers_without_changing_progress():
    state = build_mac_controlled_round_state(
        state=_round_state(),
        worker_ids=["A", "B", "C"],
        claim_id="mac-claim-123",
        now=2000.0,
    )

    assert state["mode"] == "dgac-diloco"
    assert state["triggered_workers"] == ["A", "B", "C"]
    assert state["attendance_workers"] == []
    assert state["total_samples_seen"] == {"10": 23481}
    assert state["projected_shards"] == {"A": 4475, "B": 4475, "C": 4475}
    assert state["mac_dgac_claim_id"] == "mac-claim-123"
    assert state["triggered_at"] == 2000.0


def test_mac_failure_state_preserves_waiting_round_and_marks_failed_claim():
    original = _round_state()

    failed_claim = build_mac_failed_claim(
        claim={"claim_id": "mac-claim-123", "status": "running"},
        error="worker A failed",
        now=3000.0,
    )
    failure_state = build_mac_failure_round_state(
        state=original,
        claim_id="mac-claim-123",
        error="worker A failed",
        now=3000.0,
    )

    assert failed_claim["status"] == "failed"
    assert failed_claim["failure"] == "worker A failed"
    assert failure_state["mode"] == "waiting"
    assert failure_state["attendance_workers"] == ["A", "B", "C"]
    assert failure_state["triggered_workers"] == []
    assert failure_state["total_samples_seen"] == {"10": 23481}
    assert failure_state["mac_dgac_claim_id"] == "mac-claim-123"
    assert failure_state["mac_dgac_failure"] == "worker A failed"


def test_local_worker_command_is_sequential_mps_safe_and_never_uses_4bit_or_github_signal():
    command = build_local_dgac_worker_command(
        worker_id="B",
        repo_id="WeirdRunner/Ouroboros",
        output_root="runs/mac_dgac_fallback",
        outer_lr=0.7,
        wandb_project="project",
    )

    assert command[:3] == [sys.executable, "-m", "ouroboros.coconut"]
    assert "torchrun" not in command
    assert "--use_4bit" not in command
    assert "--use_halt_gate" in command
    assert "--resume_from_diloco_anchor" in command
    assert "--mac_mps_mamba_kernels" in command
    assert "--latent_cache" not in command
    assert command[command.index("--diloco_worker_id") + 1] == "B"
    assert command[command.index("--diloco_signal_repo") + 1] == ""
    assert command[command.index("--output_dir") + 1] == "runs/mac_dgac_fallback/worker_B"
    assert command[command.index("--wandb_mode") + 1] == "online"
    assert command[command.index("--wandb_project") + 1] == "project"
    assert command[command.index("--max_seq_len") + 1] == "384"
    assert command[command.index("--grad_accum") + 1] == "8"
    assert command[command.index("--dgac_halt_probe_steps") + 1] == "stage_k"
    assert "--profile_training_timing" in command
    assert command[command.index("--log_every") + 1] == "1"


def test_local_canary_command_profiles_without_diloco_upload_or_aggregation():
    command = build_local_dgac_canary_command(
        repo_id="WeirdRunner/Ouroboros",
        output_root="runs/mac_dgac_fallback",
        grad_accum=2,
        max_train_steps=1,
        max_samples=12,
        wandb_project="project",
    )

    assert command[:3] == [sys.executable, "-m", "ouroboros.coconut"]
    assert "--diloco_mode" not in command
    assert "--push_to_hub" not in command
    assert "--resume_from_diloco_anchor" in command
    assert "--profile_training_timing" in command
    assert "--mac_mps_mamba_kernels" in command
    assert "--latent_cache" not in command
    assert "--no-grad_checkpoint" not in command
    assert "--no-gen_every_stage" in command
    assert command[command.index("--diloco_state_repo") + 1] == "WeirdRunner/Ouroboros"
    assert command[command.index("--grad_accum") + 1] == "2"
    assert command[command.index("--max_train_steps") + 1] == "1"
    assert command[command.index("--max_samples") + 1] == "12"
    assert command[command.index("--max_seq_len") + 1] == "384"
    assert command[command.index("--dgac_halt_probe_steps") + 1] == "stage_k"
    assert command[command.index("--log_every") + 1] == "1"
    assert command[command.index("--output_dir") + 1] == f"runs/mac_dgac_fallback/{CANARY_LATEST_DIRNAME}"
    assert command[command.index("--wandb_project") + 1] == "project"

    no_checkpoint_command = build_local_dgac_canary_command(
        repo_id="WeirdRunner/Ouroboros",
        disable_grad_checkpoint=True,
    )
    assert "--no-grad_checkpoint" in no_checkpoint_command


def test_canary_output_promotion_keeps_only_latest_weights(tmp_path):
    output_root = tmp_path / "mac_dgac_fallback"
    output_root.mkdir()
    old_canary = output_root / "canary_old"
    old_canary.mkdir()
    (old_canary / "weights.bin").write_text("old", encoding="utf-8")
    current_latest = output_root / CANARY_LATEST_DIRNAME
    current_latest.mkdir()
    (current_latest / "weights.bin").write_text("previous", encoding="utf-8")
    active = output_root / "canary_active_abc"
    active.mkdir()
    (active / "weights.bin").write_text("new", encoding="utf-8")

    promote_canary_output(active, current_latest)
    prune_canary_outputs(output_root, keep_latest=True)

    assert sorted(path.name for path in output_root.iterdir()) == [CANARY_LATEST_DIRNAME]
    assert (current_latest / "weights.bin").read_text(encoding="utf-8") == "new"


def test_local_aggregation_command_uses_skip_trigger_and_matching_claim():
    command = build_local_dgac_aggregation_command(
        repo_id="WeirdRunner/Ouroboros",
        claim_id="mac-claim-123",
        total_train_samples=36906,
        outer_lr=0.7,
        wandb_project="project",
    )

    assert command[:3] == [sys.executable, "-m", "ouroboros.coordinator"]
    assert "--skip_trigger" in command
    assert "--mac_claim_id" in command
    assert command[command.index("--mac_claim_id") + 1] == "mac-claim-123"
    assert command[command.index("--force_worker_ids") + 1] == "A"
    assert command[command.index("--wandb_project") + 1] == "project"
    assert "--hf_token" not in command
    assert "--wandb_key" not in command
    assert "--kaggle_run_mode" not in command


def test_active_mac_claim_blocks_github_coordinator_dispatch(monkeypatch):
    monkeypatch.setattr(coordinator, "parse_args", lambda: _args())
    triggered = []
    uploads = []

    def fake_download(repo_id, path, token):
        if path == MAC_DGAC_CLAIM_PATH:
            return {
                "claim_id": "mac-claim-123",
                "status": "running",
                "created_at": 1000.0,
                "expires_at": 5000.0,
            }
        raise AssertionError("coordinator must not read round_state while a foreign Mac claim is active")

    monkeypatch.setattr(coordinator.time, "time", lambda: 2000.0)
    monkeypatch.setattr(coordinator, "hub_download_json", fake_download)
    monkeypatch.setattr(coordinator, "hub_upload_json", lambda *args, **kwargs: uploads.append(args))
    monkeypatch.setattr(coordinator, "trigger_kaggle_workers", lambda *args, **kwargs: triggered.append(kwargs))

    coordinator.main()

    assert triggered == []
    assert uploads == []


def test_matching_mac_claim_is_active_until_expiry():
    claim = {
        "claim_id": "mac-claim-123",
        "status": "running",
        "created_at": 1000.0,
        "expires_at": 5000.0,
    }

    assert is_active_mac_claim(claim, now=4999.0) is True
    assert is_active_mac_claim(claim, now=5000.0) is False
