from __future__ import annotations

import argparse
import ast
import time
from pathlib import Path

import torch

from ouroboros.train import load_checkpoint, run_training_stages, save_checkpoint
from tests.fakes.eval_fakes import FakeCausalLM, FakeTokenizer

REPO_ROOT = Path(__file__).resolve().parents[1]


def _function_ast_dump(path: Path, function_name: str) -> str:
    module = ast.parse(path.read_text(encoding="utf-8"))
    for node in module.body:
        if isinstance(node, ast.FunctionDef) and node.name == function_name:
            return ast.dump(node, include_attributes=False)
    raise AssertionError(f"{function_name} not found in {path}")


def test_checkpoint_and_timeout_core_ast_match_monolith_source_of_truth():
    monolith_path = REPO_ROOT / "tests" / "fixtures" / "training_monolith_source.py"
    owner_paths = {
        "save_checkpoint": REPO_ROOT / "ouroboros" / "training" / "checkpointing.py",
        "load_checkpoint": REPO_ROOT / "ouroboros" / "training" / "checkpointing.py",
        "prune_epoch_checkpoints": REPO_ROOT / "ouroboros" / "training" / "checkpointing.py",
        "make_timeout_checker": REPO_ROOT / "ouroboros" / "training" / "stage_runner.py",
        "run_training_stages": REPO_ROOT / "ouroboros" / "training" / "stage_runner.py",
    }
    for function_name, owner_path in owner_paths.items():
        assert _function_ast_dump(owner_path, function_name) == _function_ast_dump(monolith_path, function_name)


def _training_args(**overrides) -> argparse.Namespace:
    values = dict(
        use_halt_gate=False,
        model_id="fake/model",
        push_to_hub=False,
        _resolved_hf_token=None,
        hf_repo_id="fake/repo",
        hf_stage_subdir="runs/stage3",
        keep_checkpoints_per_stage=2,
        stage_0_epochs=None,
        epochs_per_stage=1,
        batch_size=1,
        grad_accum=1,
        max_seq_len=64,
        lr=1e-3,
        min_lr_ratio=0.1,
        warmup_steps=0,
        weight_decay=0.0,
        max_grad_norm=1.0,
        seed=123,
        session_timeout_hours=100.0,
        graceful_exit_buffer_minutes=1.0,
        val_skip_buffer_minutes=0.0,
        log_every=1,
        val_batch_size=1,
        gen_every_stage=True,
        gen_max_tokens=2,
        halt_threshold=0.9,
        dgac_warmup_steps=0,
        dgac_ramp_steps=1,
        dgac_lambda_ponder_max=0.0,
        dgac_tau=0.95,
        dgac_lambda_diversity=0.0,
    )
    values.update(overrides)
    return argparse.Namespace(**values)


def test_save_checkpoint_preserves_adapter_layout_training_state_and_atomic_replace(tmp_path):
    model = FakeCausalLM()
    args = _training_args()
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda _: 1.0)

    ckpt = save_checkpoint(
        output_dir=tmp_path,
        step=7,
        epoch=1,
        step_in_epoch=2,
        step_in_phase=3,
        stage_k=4,
        model=model,
        halt_gate=None,
        optimizer=optimizer,
        scheduler=scheduler,
        args=args,
        val_ce=1.25,
        val_acc=0.5,
        tag="",
    )

    assert ckpt == tmp_path / "stage_4" / "checkpoint-0000007"
    assert (ckpt / "adapter_model" / "adapter_model.safetensors").exists()
    assert (ckpt / "adapter_model" / "adapter_config.json").exists()
    assert not (tmp_path / "stage_4" / "checkpoint-0000007.tmp").exists()

    state = torch.load(ckpt / "training_state.pt", map_location="cpu")
    assert state["stage_k"] == 4
    assert state["step"] == 7
    assert state["epoch"] == 1
    assert state["step_in_epoch"] == 2
    assert state["step_in_phase"] == 3
    assert state["val_ce"] == 1.25
    assert state["val_acc"] == 0.5
    assert state["optimizer"] is not None
    assert state["scheduler"] is not None
    assert state["use_halt_gate"] is False
    assert state["model_id"] == "fake/model"


def test_load_checkpoint_tolerates_optimizer_scheduler_mismatch_without_adapter_import(tmp_path, capsys):
    ckpt = tmp_path / "stage_0" / "checkpoint-0000001"
    ckpt.mkdir(parents=True)
    torch.save(
        {
            "stage_k": 0,
            "step": 1,
            "epoch": 0,
            "step_in_epoch": 0,
            "step_in_phase": 0,
            "val_ce": None,
            "val_acc": None,
            "optimizer": {"bad": "state"},
            "scheduler": {"bad": "state"},
            "use_halt_gate": False,
            "model_id": "fake/model",
        },
        ckpt / "training_state.pt",
    )
    model = FakeCausalLM()
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda _: 1.0)

    state = load_checkpoint(
        ckpt,
        model=model,
        halt_gate=None,
        optimizer=optimizer,
        scheduler=scheduler,
        device=torch.device("cpu"),
        verbose=True,
    )

    assert state["step"] == 1
    out = capsys.readouterr().out
    assert "optimizer state mismatch" in out
    assert "scheduler state mismatch" in out


def test_one_stage_fake_smoke_training_run_exercises_public_loop_and_checkpointing(tmp_path):
    tokenizer = FakeTokenizer()
    model = FakeCausalLM()
    args = _training_args()
    train_samples = [
        {"question": "What is 1+1?", "steps": ["Add."], "answer_full": "2", "answer_norm": "2"},
        {"question": "What is 2+2?", "steps": ["Add."], "answer_full": "4", "answer_norm": "4"},
    ]
    val_samples = [
        {"question": "What is 3+3?", "steps": ["Add."], "answer_full": "6", "answer_norm": "6"},
    ]

    result = run_training_stages(
        model=model,
        tokenizer=tokenizer,
        halt_gate=None,
        train_samples=train_samples,
        val_samples=val_samples,
        lat_token_id=6,
        pad_id=tokenizer.pad_token_id,
        args=args,
        device=torch.device("cpu"),
        output_dir=tmp_path,
        session_start=time.perf_counter(),
        wandb_run=None,
        stages=[0],
        curriculum_max_stage=0,
        load_best_between_stages=False,
        run_generation_at_stage_end=True,
        run_epoch_end_val=True,
    )

    assert result["global_step"] >= 1
    assert result["timeout_triggered"] is False
    assert result["val_budget_triggered"] is False
    assert result["samples_seen"] == len(train_samples)
    assert result["stages"] == [0]
    assert (tmp_path / "stage_0" / "checkpoint-0000002" / "training_state.pt").exists()
    assert (tmp_path / "stage_0" / "best" / "training_state.pt").exists()
    assert model.model.grad_enabled_observations
    assert set(model.model.device_type_observations) == {"cpu"}
