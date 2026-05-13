from __future__ import annotations

import argparse
import ast
import time
from pathlib import Path

import pytest
import torch

from ouroboros.diloco import worker as worker_module
from ouroboros.diloco.shared import RoundState, ordered_unique_workers
from tests.fakes.eval_fakes import FakeCausalLM, FakeHaltGate, FakeTokenizer

REPO_ROOT = Path(__file__).resolve().parents[1]


def _function_ast_dump(path: Path, function_name: str) -> str:
    module = ast.parse(path.read_text(encoding="utf-8"))
    for node in module.body:
        if isinstance(node, ast.FunctionDef) and node.name == function_name:
            return ast.dump(node, include_attributes=False)
    raise AssertionError(f"{function_name} not found in {path}")


def test_diloco_worker_core_ast_matches_monolith_source_of_truth():
    monolith_path = REPO_ROOT / "tests" / "fixtures" / "training_monolith_source.py"
    modular_path = REPO_ROOT / "ouroboros" / "diloco" / "worker.py"
    for function_name in ("_partition_contiguous_range", "diloco_get_shard"):
        assert _function_ast_dump(modular_path, function_name) == _function_ast_dump(monolith_path, function_name)

    modular_source = modular_path.read_text(encoding="utf-8")
    for contract in (
        "samples_already_seen",
        "triggered_workers",
        "attendance_workers",
        "samples_seen",
        "weights_path",
        "diloco_signal_repo",
        "DGAC DiLoCo requires --resume_from_diloco_anchor",
    ):
        assert contract in modular_source


def _args(worker_id: str = "A", **overrides) -> argparse.Namespace:
    values = dict(
        use_halt_gate=False,
        diloco_worker_id=worker_id,
        resume_from=None,
        resume_from_diloco_anchor=False,
        diloco_state_repo="fake/state",
        diloco_signal_repo="fake/signal",
        seed=42,
        push_to_hub=False,
        stage_0_epochs=None,
        epochs_per_stage=1,
        gen_every_stage=False,
        diloco_run_val=False,
        batch_size=1,
        grad_accum=1,
        max_seq_len=64,
        lr=1e-3,
        min_lr_ratio=0.1,
        warmup_steps=0,
        weight_decay=0.0,
        max_grad_norm=1.0,
        session_timeout_hours=100.0,
        graceful_exit_buffer_minutes=1.0,
        val_skip_buffer_minutes=0.0,
        log_every=1,
        val_batch_size=1,
        gen_max_tokens=2,
        wandb_mode="disabled",
        wandb_project="fake",
        wandb_entity=None,
        model_id="fake/model",
        keep_checkpoints_per_stage=1,
        _resolved_hf_token="hf_fake",
        hf_repo_id="fake/state",
        hf_stage_subdir="runs/stage3",
        halt_threshold=0.9,
        dgac_warmup_steps=0,
        dgac_ramp_steps=1,
        dgac_lambda_ponder_max=0.0,
        dgac_tau=0.95,
        dgac_lambda_diversity=0.0,
    )
    values.update(overrides)
    return argparse.Namespace(**values)


def test_dgac_dedicated_round_wandb_identity_avoids_stage_round_collision():
    identity = worker_module._diloco_wandb_identity(
        _args(worker_id="A", use_halt_gate=True, resume_from_diloco_anchor=True),
        stage_k=10,
        round_n=1,
        is_dgac_diloco=True,
    )

    assert identity["id"] == "dgac-a-r0001"
    assert identity["group"] == "dgac-dedicated-r0001"
    assert identity["name"] == "DGAC Worker A | Dedicated Round 001"
    assert identity["config"]["mode"] == "dgac-dedicated-round"
    assert identity["config"]["dgac_round_n"] == 1


def test_normal_diloco_wandb_identity_keeps_stage_round_shape():
    identity = worker_module._diloco_wandb_identity(
        _args(worker_id="B"),
        stage_k=7,
        round_n=3,
        is_dgac_diloco=False,
    )

    assert identity["id"] == "diloco-b-s7-r3"
    assert identity["group"] == "diloco-b-s7"
    assert identity["name"] == "Worker B | Stage 7 | Round 3"
    assert identity["config"]["mode"] == "diloco"


def test_diloco_shard_determinism_and_partition_contract():
    samples = [{"id": i} for i in range(10)]
    shards = {
        worker_id: worker_module.diloco_get_shard(
            samples,
            worker_id=worker_id,
            stage_k=2,
            round_n=3,
            seed=11,
            samples_already_seen=1,
        )
        for worker_id in ("A", "B", "C")
    }
    repeat = worker_module.diloco_get_shard(samples, "B", stage_k=2, round_n=3, seed=11, samples_already_seen=1)

    assert shards["B"] == repeat
    ids_by_worker = {wid: [item["id"] for item in shard] for wid, shard in shards.items()}
    flattened = [item for ids in ids_by_worker.values() for item in ids]
    assert len(flattened) == len(set(flattened)) == 9
    assert max(len(shards["A"]), len(shards["B"]), len(shards["C"])) - min(
        len(shards["A"]), len(shards["B"]), len(shards["C"])
    ) <= 1

    assert worker_module.diloco_get_shard(samples, "A", 2, 3, 11, samples_already_seen=10) == []
    assert worker_module.diloco_get_shard([], "A", 2, 3, 11) == []


def test_round_state_parsing_normalizes_triggered_and_attendance_workers():
    state = RoundState.from_dict(
        {
            "stage_k": "2",
            "round_n": "3",
            "total_samples_seen": {"2": "7"},
            "completed_stages": ["0", "1"],
            "triggered_workers": ["b", "A", "B", "z"],
            "attendance_workers": ["a", "C", "C"],
            "seed": "99",
            "extra_key": "preserved",
        }
    ).to_dict()

    assert state["stage_k"] == 2
    assert state["round_n"] == 3
    assert state["total_samples_seen"] == {"2": 7}
    assert state["completed_stages"] == [0, 1]
    assert state["triggered_workers"] == ["B", "A"]
    assert state["attendance_workers"] == ["C"]
    assert state["seed"] == 99
    assert state["extra_key"] == "preserved"
    assert ordered_unique_workers(["a", "A", "x"], ["B", "a", "C"]) == ["A", "B", "C"]


def test_anchor_download_fallback_leaves_model_weights_intact(monkeypatch):
    model = FakeCausalLM()
    before = model.lm_head.weight.detach().clone()

    def boom(*args, **kwargs):
        raise RuntimeError("missing anchor")

    monkeypatch.setitem(__import__("sys").modules, "huggingface_hub", type("HF", (), {"hf_hub_download": boom}))
    # Import inside the function will use the fake module above and then fail;
    # the worker must swallow the anchor miss and leave weights untouched.
    worker_module.diloco_download_anchor(model, "hf", "repo", "anchor", torch.device("cpu"))
    assert torch.equal(model.lm_head.weight.detach(), before)


def test_required_anchor_download_raises_instead_of_training_from_fresh_weights(monkeypatch):
    model = FakeCausalLM()

    def boom(*args, **kwargs):
        raise RuntimeError("missing anchor")

    monkeypatch.setitem(__import__("sys").modules, "huggingface_hub", type("HF", (), {"hf_hub_download": boom}))

    with pytest.raises(RuntimeError, match="Required DiLoCo anchor load failed"):
        worker_module.diloco_download_anchor(
            model,
            "hf",
            "repo",
            "anchor",
            torch.device("cpu"),
            required=True,
        )


def test_attendance_only_worker_uploads_zero_sample_status_without_training(monkeypatch, tmp_path):
    calls: dict[str, list] = {"download": [], "upload": [], "signal": []}
    monkeypatch.setattr(worker_module, "barrier", lambda: None)
    monkeypatch.setattr(worker_module, "diloco_read_round_state", lambda hf_token, repo_id: {
        "stage_k": 0,
        "round_n": 5,
        "anchor_path": "anchor/path",
        "triggered_workers": ["A"],
        "attendance_workers": ["B"],
        "total_samples_seen": {},
        "seed": 42,
    })
    monkeypatch.setattr(worker_module, "diloco_download_anchor", lambda *args, **kwargs: calls["download"].append((args, kwargs)))
    monkeypatch.setattr(worker_module, "diloco_upload_worker_state", lambda **kwargs: calls["upload"].append(kwargs))
    monkeypatch.setattr(worker_module, "_resolve_github_token_common", lambda: "gh_fake")
    monkeypatch.setattr(worker_module, "diloco_push_signal", lambda *args: calls["signal"].append(args))

    result = worker_module.run_diloco_worker(
        model=FakeCausalLM(),
        tokenizer=FakeTokenizer(),
        halt_gate=None,
        train_samples=[{"question": "Q", "steps": ["s"], "answer_full": "a", "answer_norm": "a"}],
        val_samples=[],
        curriculum_max_stage=0,
        lat_token_id=6,
        pad_id=0,
        args=_args(worker_id="B"),
        device=torch.device("cpu"),
        output_dir=tmp_path,
        session_start=time.perf_counter(),
        wandb_run=None,
        hf_token="hf_fake",
    )

    assert result["samples_seen"] == 0
    assert calls["download"]
    assert calls["upload"][0]["samples_seen"] == 0
    assert calls["upload"][0]["worker_id"] == "B"
    assert calls["signal"]


def test_diloco_worker_fake_smoke_trains_uploads_status_and_never_uses_network(monkeypatch, tmp_path):
    calls: dict[str, list] = {"download": [], "upload": [], "signal": []}
    monkeypatch.setattr(worker_module, "barrier", lambda: None)
    monkeypatch.setattr(worker_module, "diloco_read_round_state", lambda hf_token, repo_id: {
        "stage_k": 0,
        "round_n": 1,
        "anchor_path": "anchor/path",
        "triggered_workers": ["A"],
        "attendance_workers": [],
        "total_samples_seen": {},
        "seed": 42,
    })
    monkeypatch.setattr(worker_module, "diloco_download_anchor", lambda *args, **kwargs: calls["download"].append((args, kwargs)))
    monkeypatch.setattr(worker_module, "diloco_upload_worker_state", lambda **kwargs: calls["upload"].append(kwargs))
    monkeypatch.setattr(worker_module, "_resolve_github_token_common", lambda: "gh_fake")
    monkeypatch.setattr(worker_module, "diloco_push_signal", lambda *args: calls["signal"].append(args))

    result = worker_module.run_diloco_worker(
        model=FakeCausalLM(),
        tokenizer=FakeTokenizer(),
        halt_gate=None,
        train_samples=[
            {"question": "Q1", "steps": ["s"], "answer_full": "a", "answer_norm": "a"},
            {"question": "Q2", "steps": ["s"], "answer_full": "b", "answer_norm": "b"},
            {"question": "Q3", "steps": ["s"], "answer_full": "c", "answer_norm": "c"},
        ],
        val_samples=[],
        curriculum_max_stage=0,
        lat_token_id=6,
        pad_id=0,
        args=_args(worker_id="A"),
        device=torch.device("cpu"),
        output_dir=tmp_path,
        session_start=time.perf_counter(),
        wandb_run=None,
        hf_token="hf_fake",
    )

    assert result["stage_k"] == 0
    assert result["round_n"] == 1
    assert result["samples_seen"] == 1
    assert result["global_step"] >= 1
    assert calls["download"]
    assert calls["upload"][0]["worker_id"] == "A"
    assert calls["upload"][0]["samples_seen"] == 1
    assert (tmp_path / "diloco_worker_upload" / "worker_A_stage_0_round_1" / "adapter_model.safetensors").exists()
    assert calls["signal"]


def test_dgac_diloco_requires_resume_from_anchor():
    with pytest.raises(ValueError, match="DGAC DiLoCo requires --resume_from_diloco_anchor"):
        worker_module.run_diloco_worker(
            model=FakeCausalLM(),
            tokenizer=FakeTokenizer(),
            halt_gate=FakeHaltGate(),
            train_samples=[],
            val_samples=[],
            curriculum_max_stage=0,
            lat_token_id=6,
            pad_id=0,
            args=_args(worker_id="A", use_halt_gate=True),
            device=torch.device("cpu"),
            output_dir=Path("/tmp/unused"),
            session_start=time.perf_counter(),
            wandb_run=None,
            hf_token="hf_fake",
        )


def test_dgac_diloco_worker_forces_one_local_epoch_and_uploads_halt_gate(monkeypatch, tmp_path):
    calls: dict[str, list] = {"download": [], "upload": [], "signal": [], "train": []}
    monkeypatch.setattr(worker_module, "barrier", lambda: None)
    monkeypatch.setattr(worker_module, "diloco_read_round_state", lambda hf_token, repo_id: {
        "stage_k": 10,
        "round_n": 0,
        "mode": "dgac-diloco",
        "anchor_path": "diloco_state/anchor",
        "triggered_workers": ["A", "B", "C"],
        "attendance_workers": [],
        "total_samples_seen": {"10": 0},
        "dgac_diloco": True,
        "seed": 42,
    })
    monkeypatch.setattr(worker_module, "diloco_download_anchor", lambda *args, **kwargs: calls["download"].append((args, kwargs)))
    monkeypatch.setattr(worker_module, "diloco_upload_worker_state", lambda **kwargs: calls["upload"].append(kwargs))
    monkeypatch.setattr(worker_module, "_resolve_github_token_common", lambda: "gh_fake")
    monkeypatch.setattr(worker_module, "diloco_push_signal", lambda *args: calls["signal"].append(args))

    from ouroboros.training import stage_runner as stage_runner_module

    def fake_run_training_stages(**kwargs):
        calls["train"].append(kwargs)
        assert kwargs["halt_gate"] is not None
        assert kwargs["args"].epochs_per_stage == 1
        return {
            "samples_seen": len(kwargs["train_samples"]),
            "global_step": 1,
            "timeout_triggered": False,
            "val_budget_triggered": False,
            "stages": [10],
        }

    monkeypatch.setattr(stage_runner_module, "run_training_stages", fake_run_training_stages)

    result = worker_module.run_diloco_worker(
        model=FakeCausalLM(),
        tokenizer=FakeTokenizer(),
        halt_gate=FakeHaltGate(),
        train_samples=[
            {"question": f"Q{i}", "steps": ["s"], "answer_full": "a", "answer_norm": "a"}
            for i in range(6)
        ],
        val_samples=[],
        curriculum_max_stage=10,
        lat_token_id=6,
        pad_id=0,
        args=_args(worker_id="A", use_halt_gate=True, resume_from_diloco_anchor=True, epochs_per_stage=3),
        device=torch.device("cpu"),
        output_dir=tmp_path,
        session_start=time.perf_counter(),
        wandb_run=None,
        hf_token="hf_fake",
    )

    assert result["stage_k"] == 10
    assert result["samples_seen"] > 0
    assert calls["download"][0][1]["halt_gate"] is not None
    assert calls["train"]
    assert calls["upload"][0]["halt_gate"] is not None
    assert calls["signal"]


def test_dgac_diloco_worker_runs_leader_pre_val_and_enables_generation_when_requested(monkeypatch, tmp_path):
    calls: dict[str, list] = {"download": [], "upload": [], "signal": [], "train": [], "eval": []}
    monkeypatch.setattr(worker_module, "barrier", lambda: None)
    monkeypatch.setattr(worker_module, "diloco_read_round_state", lambda hf_token, repo_id: {
        "stage_k": 10,
        "round_n": 3,
        "mode": "dgac-diloco",
        "anchor_path": "diloco_state/anchor",
        "triggered_workers": ["A", "B", "C"],
        "attendance_workers": [],
        "total_samples_seen": {"10": 0},
        "dgac_diloco": True,
        "seed": 42,
    })
    monkeypatch.setattr(worker_module, "diloco_download_anchor", lambda *args, **kwargs: calls["download"].append((args, kwargs)))
    monkeypatch.setattr(worker_module, "diloco_upload_worker_state", lambda **kwargs: calls["upload"].append(kwargs))
    monkeypatch.setattr(worker_module, "_resolve_github_token_common", lambda: "gh_fake")
    monkeypatch.setattr(worker_module, "diloco_push_signal", lambda *args: calls["signal"].append(args))

    from ouroboros.training import evaluation as evaluation_module
    from ouroboros.training import stage_runner as stage_runner_module

    monkeypatch.setattr(evaluation_module, "evaluate_stage", lambda **kwargs: (calls["eval"].append(kwargs) or (0.5, 0.25)))

    def fake_run_training_stages(**kwargs):
        calls["train"].append(kwargs)
        return {
            "samples_seen": len(kwargs["train_samples"]),
            "global_step": 1,
            "timeout_triggered": False,
            "val_budget_triggered": False,
            "stages": [10],
        }

    monkeypatch.setattr(stage_runner_module, "run_training_stages", fake_run_training_stages)

    worker_module.run_diloco_worker(
        model=FakeCausalLM(),
        tokenizer=FakeTokenizer(),
        halt_gate=FakeHaltGate(),
        train_samples=[
            {"question": f"Q{i}", "steps": ["s"], "answer_full": "a", "answer_norm": "a"}
            for i in range(12)
        ],
        val_samples=[{"question": "V", "steps": ["s"], "answer_full": "a", "answer_norm": "a"}],
        curriculum_max_stage=10,
        lat_token_id=6,
        pad_id=0,
        args=_args(
            worker_id="A",
            use_halt_gate=True,
            resume_from_diloco_anchor=True,
            epochs_per_stage=1,
            diloco_run_val=True,
            gen_every_stage=True,
        ),
        device=torch.device("cpu"),
        output_dir=tmp_path,
        session_start=time.perf_counter(),
        wandb_run=None,
        hf_token="hf_fake",
    )

    assert calls["eval"]
    assert calls["train"]
    assert calls["train"][0]["run_generation_at_stage_end"] is True
    assert calls["train"][0]["run_epoch_end_val"] is False


def test_dgac_diloco_worker_skips_duplicate_pre_val_for_non_leader(monkeypatch, tmp_path):
    calls: dict[str, list] = {"download": [], "upload": [], "signal": [], "train": [], "eval": []}
    monkeypatch.setattr(worker_module, "barrier", lambda: None)
    monkeypatch.setattr(worker_module, "diloco_read_round_state", lambda hf_token, repo_id: {
        "stage_k": 10,
        "round_n": 3,
        "mode": "dgac-diloco",
        "anchor_path": "diloco_state/anchor",
        "triggered_workers": ["A", "B", "C"],
        "attendance_workers": [],
        "total_samples_seen": {"10": 0},
        "dgac_diloco": True,
        "seed": 42,
    })
    monkeypatch.setattr(worker_module, "diloco_download_anchor", lambda *args, **kwargs: calls["download"].append((args, kwargs)))
    monkeypatch.setattr(worker_module, "diloco_upload_worker_state", lambda **kwargs: calls["upload"].append(kwargs))
    monkeypatch.setattr(worker_module, "_resolve_github_token_common", lambda: "gh_fake")
    monkeypatch.setattr(worker_module, "diloco_push_signal", lambda *args: calls["signal"].append(args))

    from ouroboros.training import evaluation as evaluation_module
    from ouroboros.training import stage_runner as stage_runner_module

    monkeypatch.setattr(evaluation_module, "evaluate_stage", lambda **kwargs: (calls["eval"].append(kwargs) or (0.5, 0.25)))

    def fake_run_training_stages(**kwargs):
        calls["train"].append(kwargs)
        return {
            "samples_seen": len(kwargs["train_samples"]),
            "global_step": 1,
            "timeout_triggered": False,
            "val_budget_triggered": False,
            "stages": [10],
        }

    monkeypatch.setattr(stage_runner_module, "run_training_stages", fake_run_training_stages)

    worker_module.run_diloco_worker(
        model=FakeCausalLM(),
        tokenizer=FakeTokenizer(),
        halt_gate=FakeHaltGate(),
        train_samples=[
            {"question": f"Q{i}", "steps": ["s"], "answer_full": "a", "answer_norm": "a"}
            for i in range(12)
        ],
        val_samples=[{"question": "V", "steps": ["s"], "answer_full": "a", "answer_norm": "a"}],
        curriculum_max_stage=10,
        lat_token_id=6,
        pad_id=0,
        args=_args(
            worker_id="B",
            use_halt_gate=True,
            resume_from_diloco_anchor=True,
            epochs_per_stage=1,
            diloco_run_val=True,
            gen_every_stage=True,
        ),
        device=torch.device("cpu"),
        output_dir=tmp_path,
        session_start=time.perf_counter(),
        wandb_run=None,
        hf_token="hf_fake",
    )

    assert calls["train"]
    assert calls["eval"] == []
