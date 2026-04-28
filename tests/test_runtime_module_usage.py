from __future__ import annotations

import ast
from pathlib import Path

from ouroboros.diloco import coordinator_runtime


def test_finetune_runtime_uses_curriculum_module_calls() -> None:
    src = Path("ouroboros/coconut/finetune_runtime.py").read_text(encoding="utf-8")
    tree = ast.parse(src)

    has_curriculum_import = any(
        isinstance(node, ast.ImportFrom)
        and node.module == "ouroboros.coconut.curriculum"
        and {"build_stage_sample"}.issubset({alias.name for alias in node.names})
        for node in tree.body
    )
    assert has_curriculum_import

    has_dataset_runtime_import = any(
        isinstance(node, ast.ImportFrom)
        and node.module == "ouroboros.coconut.dataset_runtime"
        and {
            "download_dataset_from_hub",
            "get_max_stage",
            "load_canonical_dataset",
        }.issubset({alias.name for alias in node.names})
        for node in tree.body
    )
    assert has_dataset_runtime_import

    called_names = {
        node.func.id
        for node in ast.walk(tree)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
    }
    assert "build_stage_sample" in called_names
    assert "_download_dataset_from_hub_module" in called_names
    assert "_load_canonical_dataset_module" in called_names
    assert "_get_max_stage_module" in called_names


def test_collect_ready_workers_uses_hub_state_paths(monkeypatch):
    seen_paths: list[str] = []

    def fake_download(repo_id: str, path: str, token: str):
        seen_paths.append(path)
        return {"worker_id": "A", "stage_k": 1, "round_n": 2, "status": "done", "samples_seen": 10}

    monkeypatch.setattr(coordinator_runtime, "hub_download_json", fake_download)
    ready = coordinator_runtime.collect_ready_workers("repo", "token", stage_k=1, round_n=2, expected_workers=["A"])
    assert ready and ready[0]["worker_id"] == "A"
    assert seen_paths == ["diloco_state/workers/A/status.json"]
