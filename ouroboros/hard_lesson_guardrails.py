"""Executable guardrail registry for recurring Ouroboros hard lessons.

This module is intentionally stdlib-only so it can be imported by CPU smoke,
Kaggle preflight, and local log triage without torch/CUDA/bootstrap side effects.
Docs are not the source of enforcement by themselves: every row in
``wiki/Lessons-Learned.md`` must have a matching registry entry here, and tests
ratchet that contract so new lessons cannot be added as passive prose only.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence


@dataclass(frozen=True)
class HardLessonGuardrail:
    """Machine-readable guardrail backing a documented recurring failure."""

    symptom: str
    kind: str
    guardrail: str
    refs: tuple[str, ...]
    remediation: str
    signature_patterns: tuple[str, ...] = ()

    def matches(self, text: str) -> bool:
        """Return True when this lesson's error signature is present in text."""
        if not self.signature_patterns:
            return False
        return all(re.search(pattern, text, flags=re.IGNORECASE | re.DOTALL) for pattern in self.signature_patterns)


HARD_LESSON_GUARDRAILS: tuple[HardLessonGuardrail, ...] = (
    HardLessonGuardrail(
        symptom="`kaggle kernels pull` → 403 in CI",
        kind="workflow-test",
        guardrail="Notebook/dispatch tests exercise push-only Kaggle publishing; CI never needs a pull step.",
        refs=("tests/test_kaggle_notebook_contract.py", "tests/test_diloco_coordinator_dispatch.py"),
        remediation="Use local kernel metadata plus kernels push; do not introduce a kernels pull dependency in CI.",
        signature_patterns=(r"kaggle kernels pull", r"403"),
    ),
    HardLessonGuardrail(
        symptom="W&B step collision between rounds",
        kind="regression-test",
        guardrail="Coordinator and worker identity tests reserve non-overlapping step spans and per-round run identity.",
        refs=("tests/test_diloco_worker_fake.py", "tests/test_diloco_coordinator_aggregation.py"),
        remediation="Keep round_step_span larger than the shard step estimate and keep per-round W&B IDs grouped by stage.",
        signature_patterns=(r"wandb", r"step", r"collision|non-monotonic|history"),
    ),
    HardLessonGuardrail(
        symptom="Fixed `min_workers` causes deadlock when B has empty shard",
        kind="coordinator-test",
        guardrail="Coordinator decision tests compute active shards before deciding who must report.",
        refs=("tests/test_deep_module_contracts.py", "tests/test_diloco_coordinator_orchestration.py"),
        remediation="Base required workers on projected shard work, not a fixed min_workers count.",
        signature_patterns=(r"min_workers", r"empty shard|deadlock"),
    ),
    HardLessonGuardrail(
        symptom="Stage never closes with geometric remainder",
        kind="coordinator-test",
        guardrail="Coordinator decision tests close a stage when remaining samples fall below min_shard_samples.",
        refs=("tests/test_deep_module_contracts.py", "tests/test_diloco_coordinator_state.py"),
        remediation="If remaining < min_shard_samples, declare stage complete instead of dispatching impossible shards.",
        signature_patterns=(r"remaining", r"min_shard_samples", r"stage never closes|waiting"),
    ),
    HardLessonGuardrail(
        symptom="Coordinator triggers all workers even when some have nothing",
        kind="coordinator-test",
        guardrail="Dispatch/orchestration tests assert only active projected workers are triggered.",
        refs=("tests/test_diloco_coordinator_dispatch.py", "tests/test_diloco_coordinator_orchestration.py"),
        remediation="Precompute projected shards and trigger only workers with real work or explicit attendance duty.",
        signature_patterns=(r"triggered_workers", r"empty shard|nothing to train|zero samples"),
    ),
    HardLessonGuardrail(
        symptom="Solo mode with outer_lr=0.7 blends stale anchor into new weights",
        kind="aggregation-test",
        guardrail="Aggregation tests keep solo-worker promotion on the direct-promotion path instead of outer-LR blending.",
        refs=("tests/test_diloco_coordinator_aggregation.py", "ouroboros/diloco/aggregation.py"),
        remediation="When only one worker contributes, promote its weights directly and skip the outer update blend.",
        signature_patterns=(r"solo", r"outer_lr", r"stale anchor|blend"),
    ),
    HardLessonGuardrail(
        symptom="`kaggle kernels push --accelerator` → unrecognized argument",
        kind="dependency-preflight",
        guardrail="Requirements and workflow tests pin kaggle>=1.8.4 before accelerator push flags are used.",
        refs=("tests/requirements.sh", ".github/workflows/diloco_coordinator.yml", "tests/test_kaggle_launch_matrix.py"),
        remediation="Upgrade Kaggle CLI to >=1.8.4 or remove the flag only in CPU smoke mode.",
        signature_patterns=(r"--accelerator", r"unrecognized argument"),
    ),
    HardLessonGuardrail(
        symptom="Worker C quota exhausted → coordinator stalls forever",
        kind="coordinator-timeout",
        guardrail="Coordinator state tests demote timed-out attendance/workers via triggered_at and timeout policy.",
        refs=("tests/test_diloco_coordinator_state.py", "tests/test_deep_module_contracts.py"),
        remediation="Use triggered_at plus timeout/attendance reconciliation; do not wait forever for exhausted accounts.",
        signature_patterns=(r"quota", r"Worker C|worker C", r"stall|waiting"),
    ),
    HardLessonGuardrail(
        symptom="Coordinator writes `triggered_workers` but push fails silently",
        kind="dispatch-reconcile-test",
        guardrail="Dispatch classification requires successful push output and coordinator re-dispatches triggered_at=0 states.",
        refs=("tests/test_diloco_coordinator_dispatch.py", "tests/test_diloco_coordinator_state.py"),
        remediation="Mark failed dispatches with triggered_at=0 so the next coordinator pass re-dispatches immediately.",
        signature_patterns=(r"triggered_workers", r"push failed|failed dispatch|triggered_at=0"),
    ),
    HardLessonGuardrail(
        symptom="Kaggle CLI prints `Kernel push error`/quota text with non-fatal process behavior",
        kind="error-signature-test",
        guardrail="Dispatch tests classify Kaggle stdout/stderr strictly and require a success marker.",
        refs=("tests/test_diloco_coordinator_dispatch.py", "ouroboros/diloco/dispatch.py"),
        remediation="Treat Kernel push error/quota markers as failed dispatch even if the process return code looks benign.",
        signature_patterns=(r"Kernel push error|quota", r"kaggle"),
    ),
    HardLessonGuardrail(
        symptom='`kaggle==1.6.17` + `"accelerator": "nvidiaTeslaT4"` → still P100',
        kind="runtime-fast-fail",
        guardrail="Kaggle metadata, CLI accelerator flag, and runtime GPU capability guard all require T4-or-better.",
        refs=("tests/test_kaggle_launch_matrix.py", "tests/test_kaggle_notebook_contract.py", "jamba_coconut_finetune.py"),
        remediation="Use kaggle>=1.8.4, accelerator=NvidiaTeslaT4, and let cc < 7.5 runtime fast-fail reset dispatch.",
        signature_patterns=(r"P100|sm60", r"nvidiaTeslaT4|accelerator|kaggle==1\.6\.17"),
    ),
    HardLessonGuardrail(
        symptom='`wandb==0.25.0` `resume="allow"` on finished run → ephemeral run',
        kind="identity-test",
        guardrail="Worker W&B identity tests require unique per-round run IDs and stage grouping.",
        refs=("tests/test_diloco_worker_fake.py", "ouroboros/wandb_runtime.py"),
        remediation="Use deterministic per-round IDs with group=stage; do not resume finished runs with resume=allow.",
        signature_patterns=(r"wandb", r"resume=.allow.|ephemeral|finished run"),
    ),
    HardLessonGuardrail(
        symptom="W&B dashboard unreadable with many overlapping runs",
        kind="identity-test",
        guardrail="Worker tests validate run id/group shape for normal, DGAC, and local fallback runs.",
        refs=("tests/test_diloco_worker_fake.py", "ouroboros/wandb_runtime.py"),
        remediation="Keep unique IDs per round and group related runs by stage or DGAC round.",
        signature_patterns=(r"wandb", r"overlapping runs|dashboard"),
    ),
    HardLessonGuardrail(
        symptom="`--use_halt_gate` starts from random LoRA weights in DiLoCo path",
        kind="cli-contract-test",
        guardrail="Training plan and worker lifecycle tests require resume_from_diloco_anchor for DGAC DiLoCo.",
        refs=("tests/test_deep_module_contracts.py", "tests/test_dgac_anchor_cli_contract.py", "ouroboros/training_plan.py"),
        remediation="Pair --use_halt_gate with --resume_from_diloco_anchor for DGAC anchor-start paths.",
        signature_patterns=(r"--use_halt_gate", r"random LoRA|resume_from_diloco_anchor"),
    ),
    HardLessonGuardrail(
        symptom="`--resume_from_diloco_anchor` used when evaluating/resuming a numbered DGAC checkpoint",
        kind="cli-contract-test",
        guardrail="Checkpoint docs and CLI tests distinguish anchor-start from numbered checkpoint resume.",
        refs=("tests/test_dgac_anchor_cli_contract.py", "wiki/Checkpoint-Hub-Sync.md"),
        remediation="For numbered checkpoints, omit --resume_from_diloco_anchor and use --resume_from or hf_stage_subdir discovery.",
        signature_patterns=(r"resume_from_diloco_anchor", r"checkpoint-\d+|numbered DGAC checkpoint"),
    ),
    HardLessonGuardrail(
        symptom="H100 run reaches epoch end but does not produce val/gen",
        kind="workflow-contract",
        guardrail="Session/status docs mark skipped val/gen checkpoints as training evidence until separate eval passes.",
        refs=("wiki/STATUS.md", "wiki/SessionLog.md", "tests/test_dgac_anchor_cli_contract.py"),
        remediation="Do not treat an H100 checkpoint as quality-gated if val/gen were skipped by timeout buffer.",
        signature_patterns=(r"H100", r"val/gen|validation|generation", r"skipped|buffer"),
    ),
    HardLessonGuardrail(
        symptom="Completed PRDs/plans linger in `prds/` or `plans/` and compete with docs",
        kind="workflow-doc-contract",
        guardrail="Engineering workflow docs define PRDs/plans as temporary artifacts promoted into wiki decisions.",
        refs=("wiki/Engineering-Workflow.md",),
        remediation="Promote durable decisions to wiki and delete obsolete PRD/plan files after implementation.",
        signature_patterns=(r"prds/|plans/", r"obsolete|compete|linger"),
    ),
    HardLessonGuardrail(
        symptom="Runtime `signals/*.json` appears in source control",
        kind="source-control-guard",
        guardrail="Gitignore and workflow tests keep generated signal JSONs untracked while preserving signals/.gitkeep.",
        refs=(".gitignore", "tests/test_kaggle_launch_matrix.py"),
        remediation="Track only signals/.gitkeep; never commit generated signals/*.json runtime files.",
        signature_patterns=(r"signals/.*\.json", r"source control|git"),
    ),
    HardLessonGuardrail(
        symptom="Root scripts become safer but then risk regrowing into monoliths",
        kind="deep-module-test",
        guardrail="Entrypoint adapter tests assert root scripts delegate into package modules after bootstrap guards.",
        refs=("tests/test_training_entrypoint_adapter.py", "tests/test_deep_module_contracts.py"),
        remediation="Keep root scripts as thin adapters and put behavior in tested package modules.",
        signature_patterns=(r"root scripts|entrypoint", r"monolith"),
    ),
    HardLessonGuardrail(
        symptom="Test assumes a clean shell but developer/CI has real tokens exported",
        kind="test-hygiene",
        guardrail="Tests use monkeypatch.delenv/setenv around ambient token/runtime variables.",
        refs=("tests/test_kaggle_runtime_contract.py", "tests/test_runtime_env_aliases.py"),
        remediation="Clear ambient token/runtime env before asserting fake payloads or secret presence.",
        signature_patterns=(r"HF_TOKEN|WANDB|GITHUB_TOKEN|GH_TOKEN", r"ambient|clean shell|real tokens"),
    ),
    HardLessonGuardrail(
        symptom="TDD loop stalls after each tiny red/green step or swings into bulk refactor",
        kind="process-ratchet",
        guardrail="Skill-Binder TDD guidance is phase-scoped: one vertical slice at a time, loop to phase completion.",
        refs=("wiki/Engineering-Workflow.md",),
        remediation="Use tracer-bullet vertical slices without stopping after each tiny step or batching speculative refactors.",
        signature_patterns=(r"TDD", r"stalls|bulk refactor|red/green"),
    ),
    HardLessonGuardrail(
        symptom="`attn_implementation` crash",
        kind="model-load-fallback-test",
        guardrail="Model loading strips unsupported attn_implementation and retries safe eager/auto fallback paths.",
        refs=("ouroboros/model.py", "tests/test_model_memory_policy.py"),
        remediation="Retry model load without attn_implementation when the installed transformers/model combo rejects it.",
        signature_patterns=(r"attn_implementation", r"crash|unexpected|TypeError|ValueError"),
    ),
    HardLessonGuardrail(
        symptom="`use_mamba_kernels` old TF",
        kind="model-load-fallback-test",
        guardrail="Model loading strips unsupported use_mamba_kernels and retries when transformers rejects the key.",
        refs=("ouroboros/model.py", "tests/test_model_memory_policy.py"),
        remediation="Retry without use_mamba_kernels for older transformers/model configs.",
        signature_patterns=(r"use_mamba_kernels", r"unexpected|old TF|TypeError|ValueError"),
    ),
    HardLessonGuardrail(
        symptom="`last_hidden_state` None",
        kind="runtime-assertion",
        guardrail="Latent/model seams assert last_hidden_state is present in every forward path.",
        refs=("ouroboros/model.py", "ouroboros/latent.py", "tests/test_validation_no_drift.py"),
        remediation="Fail immediately at the forward seam with context instead of propagating None into later math.",
        signature_patterns=(r"last_hidden_state", r"None"),
    ),
    HardLessonGuardrail(
        symptom="Graceful session timeout",
        kind="runtime-guard",
        guardrail="Training session helpers centralize timeout checks using script start time and graceful buffer.",
        refs=("ouroboros/training/stage_runner.py", "tests/test_training_deep_module_contracts.py"),
        remediation="Use make_timeout_checker and stop before Kaggle kills the kernel mid-upload/eval.",
        signature_patterns=(r"session timeout|graceful", r"Kaggle|timeout"),
    ),
    HardLessonGuardrail(
        symptom="`conv1d` in LoRA",
        kind="model-config-guard",
        guardrail="LoRA target selection excludes conv1d/Mamba convolution modules from adapter injection.",
        refs=("ouroboros/model.py", "tests/test_model_memory_policy.py"),
        remediation="Keep conv1d out of LORA_TARGET_MODULES; target projection/MLP layers only.",
        signature_patterns=(r"conv1d", r"LoRA|adapter"),
    ),
    HardLessonGuardrail(
        symptom="OOM at val",
        kind="eval-memory-guard",
        guardrail="Validation/generation/DGAC diagnostics run under no_grad or inference_mode and microbatch eval work.",
        refs=("ouroboros/training/evaluation.py", "tests/test_validation_no_drift.py", "tests/test_dgac_diagnostics.py"),
        remediation="Keep eval paths inference-only, empty CUDA cache before eval, and use small validation/diagnostic batches.",
        signature_patterns=(r"outofmemoryerror|CUDA out of memory|OOM", r"val|eval|validation"),
    ),
    HardLessonGuardrail(
        symptom="DGAC diagnostics eval-only OOM in `selective_scan_fn`/bitsandbytes dequantization",
        kind="eval-memory-guard",
        guardrail="DGAC diagnostics public entrypoint and CE helper are torch.inference_mode() guarded; tests assert fake forwards see grad disabled.",
        refs=("ouroboros/training/evaluation.py", "tests/test_dgac_diagnostics.py", "tests/test_hard_lesson_guardrails.py"),
        remediation="Run DGAC diagnostics under torch.inference_mode(); if this signature appears again, do not spend GPU time diagnosing before checking the inference guard.",
        signature_patterns=(r"run_dgac_diagnostics|DGAC diagnostic", r"OutOfMemoryError|CUDA out of memory", r"selective_scan_fn|bitsandbytes|dequantize"),
    ),
    HardLessonGuardrail(
        symptom="Stage 1+ samples filtered by short seq_len",
        kind="cli-default-contract",
        guardrail="Kaggle launch commands set max_seq_len through the training defaults/CLI contract for Coconut samples.",
        refs=("ouroboros/cli.py", "tests/test_dgac_anchor_cli_contract.py"),
        remediation="Use max_seq_len=1024 for real stage-1+ Coconut data unless a bounded smoke intentionally overrides it.",
        signature_patterns=(r"filtered", r"short seq_len|max_seq_len", r"stage 1|stage 1\+"),
    ),
    HardLessonGuardrail(
        symptom="Exploding gradients k≥2",
        kind="launch-contract",
        guardrail="DGAC launch commands include max_grad_norm=0.3 and tests assert the contract.",
        refs=("ouroboros/kaggle.py", "tests/test_kaggle_launch_contract.py", "tests/test_model_memory_policy.py"),
        remediation="Keep --max_grad_norm 0.3 for DGAC k>=2 training and do not loosen without a canary.",
        signature_patterns=(r"exploding gradients|grad norm|NaN", r"k≥2|k>=2|stage"),
    ),
    HardLessonGuardrail(
        symptom="mamba-ssm 2.x API break",
        kind="bootstrap-contract",
        guardrail="Bootstrap pins/installs the known-good mamba-ssm fast-path wheel and tests cover bootstrap CLI contract.",
        refs=("ouroboros/bootstrap.py", "tests/test_bootstrap_cli_contract.py", "tests/requirements.sh"),
        remediation="Use mamba-ssm 1.2.2 for this Jamba path until the 2.x API is explicitly migrated and tested.",
        signature_patterns=(r"mamba[-_]ssm", r"2\.x|API break|undefined symbol|module has no attribute|signature mismatch"),
    ),
    HardLessonGuardrail(
        symptom="NCCL watchdog kills DDP val",
        kind="bootstrap-env-guard",
        guardrail="Root entrypoint sets NCCL watchdog/heartbeat timeout env vars before torch imports.",
        refs=("jamba_coconut_finetune.py", "tests/test_bootstrap_cli_contract.py"),
        remediation="Set TORCH_NCCL_* and NCCL_TIMEOUT before importing torch/distributed.",
        signature_patterns=(r"NCCL", r"watchdog|heartbeat|timeout", r"DDP|val|validation"),
    ),
    HardLessonGuardrail(
        symptom="BF16 emulation on T4",
        kind="dtype-runtime-guard",
        guardrail="AMP dtype selection uses BF16 only on sm80+ and FP16 on T4/V100.",
        refs=("ouroboros/model.py", "tests/test_bootstrap_mac_mps.py", "tests/test_model_memory_policy.py"),
        remediation="Use float16 on T4 sm75; reserve bfloat16 for Ampere/Hopper or equivalent native BF16 hardware.",
        signature_patterns=(r"BF16|bfloat16", r"T4|sm75|emulation"),
    ),
)


def documented_hard_lesson_symptoms(markdown: str) -> tuple[str, ...]:
    """Extract symptoms from the Lessons-Learned markdown table."""
    symptoms: list[str] = []
    for raw_line in markdown.splitlines():
        line = raw_line.strip()
        if not line.startswith("|"):
            continue
        if line.startswith("|---") or line.startswith("| Symptom"):
            continue
        cells = [cell.strip() for cell in line.strip("|").split("|")]
        if len(cells) >= 2 and cells[0]:
            symptoms.append(cells[0])
    return tuple(symptoms)


def guardrail_by_symptom() -> dict[str, HardLessonGuardrail]:
    """Return guardrails keyed by the exact Lessons-Learned symptom text."""
    return {guardrail.symptom: guardrail for guardrail in HARD_LESSON_GUARDRAILS}


def unguarded_documented_lessons(markdown: str) -> tuple[str, ...]:
    """Return documented lesson symptoms missing executable guardrail records."""
    backed = guardrail_by_symptom()
    return tuple(symptom for symptom in documented_hard_lesson_symptoms(markdown) if symptom not in backed)


def duplicate_guardrail_symptoms() -> tuple[str, ...]:
    seen: set[str] = set()
    duplicates: list[str] = []
    for guardrail in HARD_LESSON_GUARDRAILS:
        if guardrail.symptom in seen:
            duplicates.append(guardrail.symptom)
        seen.add(guardrail.symptom)
    return tuple(duplicates)


def classify_failure_log(text: str) -> tuple[HardLessonGuardrail, ...]:
    """Classify a log or traceback against known hard-lesson signatures."""
    matches = [guardrail for guardrail in HARD_LESSON_GUARDRAILS if guardrail.matches(text)]
    matches.sort(key=lambda guardrail: (-len(guardrail.signature_patterns), guardrail.symptom))
    return tuple(matches)


def format_triage(matches: Sequence[HardLessonGuardrail]) -> str:
    """Render known-failure matches as a compact human-readable triage note."""
    if not matches:
        return "No known hard-lesson signature matched this log. Diagnose before adding a new lesson."
    lines = ["Known hard-lesson signature matched:"]
    for match in matches:
        lines.extend(
            [
                f"- {match.symptom}",
                f"  guardrail: {match.guardrail}",
                f"  remediation: {match.remediation}",
                f"  refs: {', '.join(match.refs)}",
            ]
        )
    return "\n".join(lines)


def triage_failure_log(text: str) -> str:
    return format_triage(classify_failure_log(text))


def triage_failure_log_path(path: str | Path) -> str:
    return triage_failure_log(Path(path).read_text(encoding="utf-8", errors="replace"))


__all__ = [
    "HARD_LESSON_GUARDRAILS",
    "HardLessonGuardrail",
    "classify_failure_log",
    "documented_hard_lesson_symptoms",
    "duplicate_guardrail_symptoms",
    "format_triage",
    "guardrail_by_symptom",
    "triage_failure_log",
    "triage_failure_log_path",
    "unguarded_documented_lessons",
]
