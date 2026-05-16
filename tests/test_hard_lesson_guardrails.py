from __future__ import annotations

from pathlib import Path

from ouroboros.bootstrap.guardrails import (
    HARD_LESSON_GUARDRAILS,
    classify_failure_log,
    duplicate_guardrail_symptoms,
    triage_failure_log,
    unguarded_documented_lessons,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
LESSONS = REPO_ROOT / "wiki" / "Lessons-Learned.md"


def test_every_hard_lesson_row_has_a_machine_readable_guardrail_record():
    missing = unguarded_documented_lessons(LESSONS.read_text(encoding="utf-8"))

    assert missing == (), (
        "Hard lessons cannot be passive prose. Add a HardLessonGuardrail entry "
        f"before adding Lessons-Learned rows. Missing: {missing!r}"
    )
    assert duplicate_guardrail_symptoms() == ()


def test_hard_lesson_guardrail_refs_point_to_repo_artifacts():
    for guardrail in HARD_LESSON_GUARDRAILS:
        assert guardrail.kind
        assert guardrail.guardrail
        assert guardrail.remediation
        assert guardrail.refs, guardrail.symptom
        for ref in guardrail.refs:
            path_text = ref.split("::", 1)[0]
            assert (REPO_ROOT / path_text).exists(), f"{guardrail.symptom}: missing guardrail ref {ref}"


def test_dgac_diagnostics_oom_log_triages_to_inference_mode_guardrail():
    log = """
    [DGAC diagnostic] CE start: rank0_pairs=1940 batch_size=1
    File "/kaggle/working/ouroboros/training/evaluation.py", line 484, in run_dgac_diagnostics
      gated_ce = _evaluate_ce_for_sample_stage_pairs(...)
    File "/usr/local/lib/python3.12/site-packages/mamba_ssm/ops/selective_scan_interface.py", line 118, in selective_scan_fn
      return SelectiveScanFn.apply(...)
    torch.OutOfMemoryError: CUDA out of memory. Tried to allocate 20.00 MiB.
    """

    matches = classify_failure_log(log)

    assert matches
    assert matches[0].symptom == "DGAC diagnostics eval-only OOM in `selective_scan_fn`/bitsandbytes dequantization"
    note = triage_failure_log(log)
    assert "torch.inference_mode" in note
    assert "tests/test_dgac_diagnostics.py" in note
