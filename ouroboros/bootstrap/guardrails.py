"""Executable guardrail registry for recurring Ouroboros hard lessons.

This module is intentionally stdlib-only so it can be imported by Kaggle
preflight and local log triage without torch/CUDA/bootstrap side effects.
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
        symptom='`kaggle kernels pull` → 403 in CI',
        kind='workflow-test',
        guardrail='Notebook/dispatch tests exercise push-only Kaggle publishing; CI never needs a pull step.',
        refs=('tests/test_diloco_coordinator_dispatch.py',),
        remediation='Use local kernel metadata plus kernels push; do not introduce a kernels pull dependency in CI.',
        signature_patterns=('kaggle kernels pull', '403'),
    ),
    HardLessonGuardrail(
        symptom='`kaggle kernels push --accelerator` → unrecognized argument',
        kind='dependency-preflight',
        guardrail='Requirements and workflow tests pin kaggle>=1.8.4 before accelerator push flags are used.',
        refs=('tests/requirements.sh', '.github/workflows/diloco_coordinator.yml', 'tests/test_kaggle_launch_matrix.py'),
        remediation='Upgrade Kaggle CLI to >=1.8.4 before using accelerator push flags.',
        signature_patterns=('--accelerator', 'unrecognized argument'),
    ),
    HardLessonGuardrail(
        symptom='Kaggle CLI prints `Kernel push error`/quota text with non-fatal process behavior',
        kind='error-signature-test',
        guardrail='Dispatch tests classify Kaggle stdout/stderr strictly and require a success marker.',
        refs=('tests/test_diloco_coordinator_dispatch.py', 'ouroboros/coordinator/dispatch.py'),
        remediation='Treat Kernel push error/quota markers as failed dispatch even if the process return code looks benign.',
        signature_patterns=('Kernel push error|quota', 'kaggle'),
    ),
    HardLessonGuardrail(
        symptom='`kaggle==1.6.17` + `"accelerator": "nvidiaTeslaT4"` → still P100',
        kind='runtime-fast-fail',
        guardrail='Kaggle metadata, CLI accelerator flag, and runtime GPU capability guard all require T4-or-better.',
        refs=('tests/test_kaggle_launch_matrix.py', 'ouroboros/coconut/__main__.py'),
        remediation='Use kaggle>=1.8.4, accelerator=NvidiaTeslaT4, and let cc < 7.5 runtime fast-fail reset dispatch.',
        signature_patterns=('P100|sm60', 'nvidiaTeslaT4|accelerator|kaggle==1\\.6\\.17'),
    ),
    HardLessonGuardrail(
        symptom='Coordinator writes `triggered_workers` but push fails silently',
        kind='dispatch-reconcile-test',
        guardrail='Dispatch classification requires successful push output and coordinator re-dispatches triggered_at=0 states.',
        refs=('tests/test_diloco_coordinator_dispatch.py', 'tests/test_diloco_coordinator_state.py'),
        remediation='Mark failed dispatches with triggered_at=0 so the next coordinator pass re-dispatches immediately.',
        signature_patterns=('triggered_workers', 'push failed|failed dispatch|triggered_at=0'),
    ),
    HardLessonGuardrail(
        symptom='Solo mode with outer_lr=0.7 blends stale anchor into new weights',
        kind='aggregation-test',
        guardrail='Aggregation tests keep solo-worker promotion on the direct-promotion path instead of outer-LR blending.',
        refs=('tests/test_diloco_coordinator_aggregation.py', 'ouroboros/coordinator/aggregation.py'),
        remediation='When only one worker contributes, promote its weights directly and skip the outer update blend.',
        signature_patterns=('solo', 'outer_lr', 'stale anchor|blend'),
    ),
    HardLessonGuardrail(
        symptom='Worker C quota exhausted → coordinator stalls forever',
        kind='coordinator-timeout',
        guardrail='Coordinator state tests demote timed-out attendance/workers via triggered_at and timeout policy.',
        refs=('tests/test_diloco_coordinator_state.py', 'tests/test_deep_module_contracts.py'),
        remediation='Use triggered_at plus timeout/attendance reconciliation; do not wait forever for exhausted accounts.',
        signature_patterns=('quota', 'Worker C|worker C', 'stall|waiting'),
    ),
    HardLessonGuardrail(
        symptom='OOM at val',
        kind='eval-memory-guard',
        guardrail='Validation/generation run under no_grad or inference_mode and microbatch eval work.',
        refs=('ouroboros/coconut/evaluation.py', 'tests/test_validation_no_drift.py'),
        remediation='Keep eval paths inference-only, empty CUDA cache before eval, and use small validation batches.',
        signature_patterns=('outofmemoryerror|CUDA out of memory|OOM', 'val|eval|validation'),
    ),
    HardLessonGuardrail(
        symptom='`last_hidden_state` None',
        kind='runtime-assertion',
        guardrail='Latent/model seams assert last_hidden_state is present in every forward path.',
        refs=('ouroboros/models/loading.py', 'ouroboros/coconut/latent.py', 'tests/test_validation_no_drift.py'),
        remediation='Fail immediately at the forward seam with context instead of propagating None into later math.',
        signature_patterns=('last_hidden_state', 'None'),
    ),
    HardLessonGuardrail(
        symptom='BF16 emulation on T4',
        kind='dtype-runtime-guard',
        guardrail='AMP dtype selection uses BF16 only on sm80+ and FP16 on T4/V100.',
        refs=('ouroboros/models/loading.py', 'tests/test_model_memory_policy.py'),
        remediation='Use float16 on T4 sm75; reserve bfloat16 for Ampere/Hopper or equivalent native BF16 hardware.',
        signature_patterns=('BF16|bfloat16', 'T4|sm75|emulation'),
    ),
    HardLessonGuardrail(
        symptom='NCCL watchdog kills DDP val',
        kind='bootstrap-env-guard',
        guardrail='Root entrypoint sets NCCL watchdog/heartbeat timeout env vars before torch imports.',
        refs=('ouroboros/coconut/__main__.py', 'tests/test_bootstrap_cli_contract.py'),
        remediation='Set TORCH_NCCL_* and NCCL_TIMEOUT before importing torch/distributed.',
        signature_patterns=('NCCL', 'watchdog|heartbeat|timeout', 'DDP|val|validation'),
    ),
    HardLessonGuardrail(
        symptom='mamba-ssm 2.x API break',
        kind='bootstrap-contract',
        guardrail='Bootstrap pins/installs the known-good mamba-ssm fast-path wheel and tests cover bootstrap CLI contract.',
        refs=('ouroboros/bootstrap/runtime.py', 'tests/test_bootstrap_cli_contract.py', 'tests/requirements.sh'),
        remediation='Use mamba-ssm 1.2.2 for this Jamba path until the 2.x API is explicitly migrated and tested.',
        signature_patterns=('mamba[-_]ssm', '2\\.x|API break|undefined symbol|module has no attribute|signature mismatch'),
    ),
    HardLessonGuardrail(
        symptom='`--use_halt_gate` starts from random LoRA weights in DiLoCo path',
        kind='cli-contract-test',
        guardrail='Training plan and worker lifecycle tests require resume_from_diloco_anchor for DGAC DiLoCo.',
        refs=('tests/test_deep_module_contracts.py', 'tests/test_dgac_anchor_cli_contract.py', 'ouroboros/coconut/training_plan.py'),
        remediation='Pair --use_halt_gate with --resume_from_diloco_anchor for DGAC anchor-start paths.',
        signature_patterns=('--use_halt_gate', 'random LoRA|resume_from_diloco_anchor'),
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
