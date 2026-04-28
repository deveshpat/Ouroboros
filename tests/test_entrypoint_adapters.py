from __future__ import annotations

from pathlib import Path


def test_repo_entrypoints_are_thin_adapters() -> None:
    root = Path(__file__).resolve().parents[1]
    diloco_lines = (root / "diloco_coordinator.py").read_text(encoding="utf-8").splitlines()
    jamba_lines = (root / "jamba_coconut_finetune.py").read_text(encoding="utf-8").splitlines()

    assert len(diloco_lines) <= 40
    assert len(jamba_lines) <= 40


def test_diloco_adapter_preserves_legacy_import_surface() -> None:
    import diloco_coordinator

    assert diloco_coordinator.WORKER_IDS == ["A", "B", "C"]
    assert diloco_coordinator._ordered_unique_worker_ids(["a", "B", "a"]) == ["A", "B"]
