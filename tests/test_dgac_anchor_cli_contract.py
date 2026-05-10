from __future__ import annotations

import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
TRAINING_MONOLITH = REPO_ROOT / "jamba_coconut_finetune.py"
MODULAR_TRAIN = REPO_ROOT / "ouroboros" / "train.py"
BLUEPRINT = REPO_ROOT / "BLUEPRINT.md"
CLI_MODULE = REPO_ROOT / "ouroboros" / "cli.py"


def test_dgac_anchor_flag_is_exposed_by_bootstrap_free_help():
    completed = subprocess.run(
        [sys.executable, str(TRAINING_MONOLITH), "--help"],
        cwd=str(REPO_ROOT),
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        timeout=15,
        check=False,
    )

    assert completed.returncode == 0, completed.stderr[:1000]
    assert "--resume_from_diloco_anchor" in completed.stdout


def test_dgac_anchor_launch_contract_matches_root_modular_and_cli_entrypoints():
    blueprint_source = BLUEPRINT.read_text(encoding="utf-8")
    monolith_source = TRAINING_MONOLITH.read_text(encoding="utf-8")
    modular_source = MODULAR_TRAIN.read_text(encoding="utf-8")
    cli_source = CLI_MODULE.read_text(encoding="utf-8")

    assert "--use_halt_gate --resume_from_diloco_anchor" in blueprint_source
    assert '"--resume_from_diloco_anchor",' in cli_source

    assert "from ouroboros.train import run_cli" in monolith_source
    assert "from ouroboros.cli import parse_args" in monolith_source

    assert "--resume_from_diloco_anchor" in modular_source
    assert "diloco_download_anchor" in modular_source
    assert "Loading DiLoCo anchor" in modular_source
    assert "requires --use_halt_gate" in modular_source
    assert "requires an HF token" in modular_source


def test_eval_only_flag_is_exposed_by_bootstrap_free_help_and_cli_parser():
    from ouroboros.cli import bootstrap_free_help_text, parse_args

    help_text = bootstrap_free_help_text()
    assert "--eval_only" in help_text
    assert "--dgac_diagnostics" in help_text

    args = parse_args([
        "--use_halt_gate",
        "--resume_from_diloco_anchor",
        "--eval_only",
        "--dgac_diagnostics",
    ])
    assert args.use_halt_gate is True
    assert args.resume_from_diloco_anchor is True
    assert args.eval_only is True
    assert args.dgac_diagnostics is True


def test_dgac_anchor_eval_only_loads_anchor_evaluates_and_skips_training(monkeypatch, tmp_path, capsys):
    import time

    from ouroboros.cli import parse_args
    from ouroboros import train as train_module
    from tests.fakes.eval_fakes import FakeCausalLM, FakeTokenizer

    calls = []

    monkeypatch.setattr(
        train_module,
        "load_canonical_dataset",
        lambda data_dir, max_samples: (
            [{"question": "train", "steps": ["s"], "answer_full": "1", "answer_norm": "1"}],
            [{"question": "val", "steps": ["s"], "answer_full": "1", "answer_norm": "1"}],
            {"n_steps_median": 10},
        ),
    )
    monkeypatch.setattr(train_module, "get_max_stage", lambda args, stats: 10)
    def fake_load_model_and_tokenizer(args, device):
        tokenizer = FakeTokenizer()
        tokenizer.save_pretrained = lambda path: Path(path).mkdir(parents=True, exist_ok=True)
        return FakeCausalLM(), tokenizer, 8, 6

    monkeypatch.setattr(train_module, "load_model_and_tokenizer", fake_load_model_and_tokenizer)
    monkeypatch.setattr(train_module, "_resolve_hf_token", lambda token: "hf_fake")
    monkeypatch.setattr(train_module, "_wandb_credentials_available", lambda: False)
    def fake_diloco_download_anchor(model, hf_token, repo, subdir, device, *, halt_gate=None):
        calls.append(("anchor", hf_token, repo, subdir, halt_gate is not None))

    monkeypatch.setattr(
        train_module,
        "diloco_download_anchor",
        fake_diloco_download_anchor,
    )
    monkeypatch.setattr(
        train_module,
        "evaluate_stage",
        lambda **kwargs: calls.append(("evaluate", kwargs["stage_k"], kwargs["halt_gate"] is not None)) or (0.42, 0.12),
    )
    monkeypatch.setattr(
        train_module,
        "run_generation_callback",
        lambda **kwargs: calls.append(("generation", kwargs["stage_k"], kwargs["halt_gate"] is not None)) or 0.75,
    )
    monkeypatch.setattr(
        train_module,
        "run_training_stages",
        lambda **kwargs: (_ for _ in ()).throw(AssertionError("eval-only must not train")),
    )

    args = parse_args([
        "--use_halt_gate",
        "--resume_from_diloco_anchor",
        "--eval_only",
        "--diloco_state_repo", "fake/state",
        "--output_dir", str(tmp_path),
        "--wandb_mode", "disabled",
    ])

    train_module.run_cli(args, script_start=time.perf_counter())

    assert calls == [
        ("anchor", "hf_fake", "fake/state", "diloco_state/anchor", True),
        ("evaluate", 10, True),
        ("generation", 10, True),
    ]
    out = capsys.readouterr().out
    assert "[DGAC] Loading DiLoCo anchor from fake/state/diloco_state/anchor" in out
    assert "[eval-only] stage=10 val_ce=0.4200 val_acc=0.1200" in out


def test_dgac_anchor_eval_only_runs_halt_gate_diagnostics_when_requested(monkeypatch, tmp_path):
    import time

    from ouroboros.cli import parse_args
    from ouroboros import train as train_module
    from tests.fakes.eval_fakes import FakeCausalLM, FakeTokenizer

    calls = []

    monkeypatch.setattr(
        train_module,
        "load_canonical_dataset",
        lambda data_dir, max_samples: (
            [{"question": "train", "steps": ["s"], "answer_full": "1", "answer_norm": "1"}],
            [{"question": "val", "steps": ["s"], "answer_full": "1", "answer_norm": "1"}],
            {"n_steps_median": 10},
        ),
    )
    monkeypatch.setattr(train_module, "get_max_stage", lambda args, stats: 10)

    def fake_load_model_and_tokenizer(args, device):
        tokenizer = FakeTokenizer()
        tokenizer.save_pretrained = lambda path: Path(path).mkdir(parents=True, exist_ok=True)
        return FakeCausalLM(), tokenizer, 8, 6

    monkeypatch.setattr(train_module, "load_model_and_tokenizer", fake_load_model_and_tokenizer)
    monkeypatch.setattr(train_module, "_resolve_hf_token", lambda token: "hf_fake")
    monkeypatch.setattr(train_module, "_wandb_credentials_available", lambda: False)
    monkeypatch.setattr(
        train_module,
        "diloco_download_anchor",
        lambda model, hf_token, repo, subdir, device, *, halt_gate=None: calls.append(
            ("anchor", halt_gate is not None)
        ),
    )
    monkeypatch.setattr(
        train_module,
        "evaluate_stage",
        lambda **kwargs: calls.append(("evaluate", kwargs["halt_gate"] is not None)) or (0.42, 0.12),
    )
    monkeypatch.setattr(
        train_module,
        "run_generation_callback",
        lambda **kwargs: calls.append(("generation", kwargs["halt_gate"] is not None)) or 0.75,
    )
    monkeypatch.setattr(
        train_module,
        "run_dgac_diagnostics",
        lambda **kwargs: calls.append(
            (
                "diagnostics",
                kwargs["halt_gate"] is not None,
                kwargs["stage_k"],
                kwargs["val_ce_forced_kmax"],
            )
        ) or {"dgac_diag/k_mean": 1.0},
    )
    monkeypatch.setattr(
        train_module,
        "run_training_stages",
        lambda **kwargs: (_ for _ in ()).throw(AssertionError("eval-only must not train")),
    )

    args = parse_args([
        "--use_halt_gate",
        "--resume_from_diloco_anchor",
        "--eval_only",
        "--dgac_diagnostics",
        "--diloco_state_repo", "fake/state",
        "--output_dir", str(tmp_path),
        "--wandb_mode", "disabled",
    ])

    train_module.run_cli(args, script_start=time.perf_counter())

    assert calls == [
        ("anchor", True),
        ("evaluate", True),
        ("generation", True),
        ("diagnostics", True, 10, 0.42),
    ]
