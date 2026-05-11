from __future__ import annotations

import importlib
import sys


def test_runtime_env_aliases_are_stdlib_safe_and_normalize_workers():
    before = set(sys.modules)
    runtime_env = importlib.import_module("ouroboros.runtime_env")

    assert runtime_env.normalize_worker_id(" b ") == "B"
    assert runtime_env.normalize_worker_id("x") is None
    assert runtime_env.require_worker_id({"WORKER_ID": "c"}) == "C"
    assert runtime_env.parse_worker_id_list("a, B, nope, a") == ["A", "B"]
    assert "torch" not in set(sys.modules) - before


def test_runtime_env_token_and_credential_aliases_fail_safe():
    from ouroboros.runtime_env import (
        env_bool,
        env_int,
        resolve_env_alias,
        resolve_hf_token,
        resolve_github_token,
        resolve_kaggle_credentials,
        resolve_wandb_key,
    )

    env = {
        "HUGGINGFACE_HUB_TOKEN": " hf ",
        "GH_TOKEN": " gh ",
        "WANDB_KEY": " wb ",
        "KAGGLE_USERNAME_A": " userA ",
        "KAGGLE_KEY_A": " keyA ",
        "FLAG": "yes",
        "COUNT": "7",
    }

    assert resolve_env_alias(env, ("HF_TOKEN", "HUGGINGFACE_HUB_TOKEN")) == "hf"
    assert resolve_hf_token(env=env) == "hf"
    assert resolve_github_token(env=env) == "gh"
    assert resolve_wandb_key(env=env) == "wb"
    assert resolve_kaggle_credentials(env, "A") == ("userA", "keyA")
    assert resolve_kaggle_credentials(env, "B") == (None, None)
    assert env_bool(env, "FLAG") is True
    assert env_bool({"FLAG": "0"}, "FLAG", default=True) is False
    assert env_int(env, "COUNT", default=3) == 7
    assert env_int({"COUNT": "bad"}, "COUNT", default=3) == 3
