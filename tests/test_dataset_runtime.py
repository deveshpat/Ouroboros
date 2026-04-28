from __future__ import annotations

import json
from pathlib import Path

from ouroboros.coconut.dataset_runtime import get_max_stage, load_canonical_dataset


def test_load_canonical_dataset_normalizes_steps(tmp_path: Path) -> None:
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    (data_dir / "train.jsonl").write_text(
        json.dumps({"question": "q", "steps": '["a","b"]', "answer_full": "ans"}) + "\n",
        encoding="utf-8",
    )
    (data_dir / "val.jsonl").write_text("", encoding="utf-8")
    (data_dir / "stats.json").write_text(json.dumps({"train": {"n_steps_median": 2}}), encoding="utf-8")

    train, val, stats = load_canonical_dataset(
        data_dir,
        max_samples=None,
        is_main_process=lambda: False,
        download_from_hub=lambda _: None,
    )
    assert train[0]["steps"] == ["a", "b"]
    assert val == []
    assert stats["train"]["n_steps_median"] == 2


def test_get_max_stage_prefers_stats_when_arg_missing() -> None:
    stage = get_max_stage(None, {"train": {"n_steps_median": 7}}, is_main_process=lambda: False)
    assert stage == 7
