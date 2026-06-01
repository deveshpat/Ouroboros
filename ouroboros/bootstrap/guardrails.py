"""Executable guardrail registry for recurring Ouroboros hard lessons.

This module is intentionally stdlib-only so it can be imported by Kaggle
preflight and local log triage without torch/CUDA/bootstrap side effects.
Docs are not the source of enforcement by themselves: every row in
``wiki/Lessons-Learned.md`` must have a matching registry entry here. Local
validation can use temporary inline commands instead of committed test files.
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
    validation_command: str = ""

    def matches(self, text: str) -> bool:
        """Return True when this lesson's error signature is present in text."""
        if not self.signature_patterns:
            return False
        return all(re.search(pattern, text, flags=re.IGNORECASE | re.DOTALL) for pattern in self.signature_patterns)


HARD_LESSON_GUARDRAILS: tuple[HardLessonGuardrail, ...] = (
    HardLessonGuardrail(
        symptom='`kaggle kernels pull` → 403 in CI',
        kind='legacy-kaggle-publish',
        guardrail='If Kaggle publishing is reintroduced, use push-only publishing; CI never needs a pull step.',
        refs=('kaggle-utils.ipynb', 'wiki/Lessons-Learned.md'),
        remediation='Use local kernel metadata plus kernels push; do not introduce a kernels pull dependency.',
        signature_patterns=('kaggle kernels pull', '403'),
    ),
    HardLessonGuardrail(
        symptom='`kaggle kernels push --accelerator` → unrecognized argument',
        kind='dependency-preflight',
        guardrail='Workflow dependency install pins kaggle>=1.8.4 before accelerator push flags are used.',
        refs=('kaggle-utils.ipynb',),
        remediation='Upgrade Kaggle CLI to >=1.8.4 before using accelerator push flags.',
        signature_patterns=('--accelerator', 'unrecognized argument'),
    ),
    HardLessonGuardrail(
        symptom='Kaggle CLI prints `Kernel push error`/quota text with non-fatal process behavior',
        kind='error-signature-test',
        guardrail='If Kaggle publishing is reintroduced, stdout/stderr must be classified strictly and require a success marker.',
        refs=('wiki/Lessons-Learned.md',),
        remediation='Treat Kernel push error/quota markers as failed publishing even if the process return code looks benign.',
        signature_patterns=('Kernel push error|quota', 'kaggle'),
    ),
    HardLessonGuardrail(
        symptom='`kaggle==1.6.17` + `"accelerator": "nvidiaTeslaT4"` → still P100',
        kind='runtime-fast-fail',
        guardrail='Kaggle metadata, CLI accelerator flag, and runtime GPU capability guard all require T4-or-better.',
        refs=('kaggle-utils.ipynb', 'ouroboros/coconut/__main__.py'),
        remediation='Use kaggle>=1.8.4, accelerator=NvidiaTeslaT4, and let cc < 7.5 runtime fast-fail before training.',
        signature_patterns=('P100|sm60', 'nvidiaTeslaT4|accelerator|kaggle==1\\.6\\.17'),
    ),
    HardLessonGuardrail(
        symptom='Legacy launch state writes `triggered_workers` but push fails silently',
        kind='legacy-launch-reconcile',
        guardrail='Archived launch logic required successful push output before marking work launched.',
        refs=('wiki/Lessons-Learned.md',),
        remediation='Do not restore stateful launch bookkeeping unless failed pushes are explicitly reconciled.',
        signature_patterns=('triggered_workers', 'push failed|failed launch|triggered_at=0'),
    ),
    HardLessonGuardrail(
        symptom='Solo mode with outer_lr=0.7 blends stale anchor into new weights',
        kind='legacy-aggregation-test',
        guardrail='Archived aggregation logic kept solo-worker promotion on the direct-promotion path instead of outer-LR blending.',
        refs=('wiki/Lessons-Learned.md',),
        remediation='When only one worker contributes, promote its weights directly and skip the outer update blend.',
        signature_patterns=('solo', 'outer_lr', 'stale anchor|blend'),
    ),
    HardLessonGuardrail(
        symptom='Legacy worker quota exhausted → launch loop stalls forever',
        kind='legacy-launch-timeout',
        guardrail='Archived launch loops required timeout/attendance reconciliation instead of waiting forever.',
        refs=('wiki/Lessons-Learned.md',),
        remediation='Keep current Kaggle launch manual and stateless; if automation returns, add timeout reconciliation first.',
        signature_patterns=('quota', 'Worker C|worker C', 'stall|waiting'),
    ),
    HardLessonGuardrail(
        symptom='OOM at val',
        kind='eval-memory-guard',
        guardrail='Validation/generation run under no_grad or inference_mode and microbatch eval work.',
        refs=('ouroboros/coconut/evaluation.py', 'wiki/GPU-Guardrails.md'),
        remediation='Keep eval paths inference-only, empty CUDA cache before eval, and use small validation batches.',
        signature_patterns=('outofmemoryerror|CUDA out of memory|OOM', 'val|eval|validation'),
    ),
    HardLessonGuardrail(
        symptom='`last_hidden_state` None',
        kind='runtime-assertion',
        guardrail='Latent/model seams assert last_hidden_state is present in every forward path.',
        refs=('ouroboros/models/loading.py', 'ouroboros/coconut/latent.py'),
        remediation='Fail immediately at the forward seam with context instead of propagating None into later math.',
        signature_patterns=('last_hidden_state', 'None'),
    ),
    HardLessonGuardrail(
        symptom='BF16 emulation on T4',
        kind='dtype-runtime-guard',
        guardrail='AMP dtype selection uses BF16 only on sm80+ and FP16 on T4/V100.',
        refs=('ouroboros/models/loading.py', 'wiki/GPU-Guardrails.md'),
        remediation='Use float16 on T4 sm75; reserve bfloat16 for Ampere/Hopper or equivalent native BF16 hardware.',
        signature_patterns=('BF16|bfloat16', 'T4|sm75|emulation'),
    ),
    HardLessonGuardrail(
        symptom='NCCL watchdog kills DDP val',
        kind='bootstrap-env-guard',
        guardrail='Root entrypoint sets NCCL watchdog/heartbeat timeout env vars before torch imports.',
        refs=('ouroboros/coconut/__main__.py', 'wiki/GPU-Guardrails.md'),
        remediation='Set TORCH_NCCL_* and NCCL_TIMEOUT before importing torch/distributed.',
        signature_patterns=('NCCL', 'watchdog|heartbeat|timeout', 'DDP|val|validation'),
    ),
    HardLessonGuardrail(
        symptom='mamba-ssm 2.x API break',
        kind='bootstrap-contract',
        guardrail='Bootstrap pins/installs the known-good mamba-ssm fast-path wheel and tests cover bootstrap CLI contract.',
        refs=('ouroboros/bootstrap/runtime.py', 'requirements.sh', 'wiki/Mamba-Bootstrap.md'),
        remediation='Use mamba-ssm 1.2.2 for this Jamba path until the 2.x API is explicitly migrated and tested.',
        signature_patterns=('mamba[-_]ssm', '2\\.x|API break|undefined symbol|module has no attribute|signature mismatch'),
    ),
    HardLessonGuardrail(
        symptom='`--use_halt_gate` starts from random LoRA weights instead of the Hub anchor',
        kind='cli-contract-test',
        guardrail='Training plan requires --resume_from_anchor for DGAC anchor-start paths.',
        refs=('ouroboros/coconut/training_plan.py', 'wiki/Lessons-Learned.md'),
        remediation='Pair --use_halt_gate with --resume_from_anchor when continuing from the Hub anchor.',
        signature_patterns=('--use_halt_gate', 'random LoRA|resume_from_anchor'),
    ),

    HardLessonGuardrail(
        symptom='Kaggle command hidden behind launch-mode modules',
        kind='workflow-visibility',
        guardrail='kaggle-utils.ipynb owns the visible launch command.',
        refs=('kaggle-utils.ipynb',),
        remediation='Keep the notebook launch cell readable and avoid hiding launch policy inside one-off helper modules.',
        signature_patterns=('launch-mode|torchrun|kaggle-utils', 'hidden|not visible|command'),
        validation_command='PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m compileall -q ouroboros',
    ),
    HardLessonGuardrail(
        symptom='Jamba fast path declared active but generation raises `Fast Mamba kernels are not available`',
        kind='runtime-readiness-contract',
        guardrail='All Jamba model-load paths must pass the shared post-load runtime probe before logging the Mamba fast path as active.',
        refs=('ouroboros/models/runtime.py', 'ouroboros/models/loading.py', 'ouroboros/inference/generation.py', 'ouroboros/eval/generation_runtime.py', 'wiki/Mamba-Bootstrap.md'),
        remediation='Route baseline, candidate, and inference loading through the shared model runtime readiness seam and fail before the evaluation loop if the probe cannot pass.',
        signature_patterns=('mamba CUDA kernels: fast path ACTIVE', 'Fast Mamba kernels are not available'),
        validation_command='python - <<\'PY\'\nfrom ouroboros.bootstrap.guardrails import triage_failure_log_path\nprint(triage_failure_log_path("/mnt/data/kaggle-utils.log.txt"))\nPY',
    ),
    HardLessonGuardrail(
        symptom='HaltGate target can look good under teacher-forced CE while generated answers degrade',
        kind='eval-claim-boundary',
        guardrail='Teacher-forced CE is a health metric only; generated-answer artifacts gate quality claims.',
        refs=('ouroboros/eval/comparison.py', 'wiki/STATUS.md'),
        remediation='Inspect generated answers before treating CE/token-accuracy as model-quality progress.',
        signature_patterns=('teacher-forced|CE|token', 'generated answers degrade|degenerate|over-stopped'),
    ),
    HardLessonGuardrail(
        symptom='Fixed-depth ablation can pass a small slice but fail the hardest slice',
        kind='eval-slice-boundary',
        guardrail='Fixed-depth runs are labeled diagnostic unless release-valid artifacts pass broader slices.',
        refs=('ouroboros/eval/comparison.py', 'wiki/STATUS.md'),
        remediation='Treat first-slice wins as diagnostics until longest/full runs and raw generations are checked.',
        signature_patterns=('fixed-depth|disable_candidate_halt_gate', 'small slice|hardest slice|longest'),
    ),
    HardLessonGuardrail(
        symptom='PEFT adapter config loaded with ignored keys',
        kind='runtime-version-boundary',
        guardrail='PEFT/Transformers versions must be recorded in eval artifacts before public claims.',
        refs=('ouroboros/eval/comparison.py', 'ouroboros/models/loading.py'),
        remediation='Align PEFT version with training/runtime or reproduce both paths before making claims.',
        signature_patterns=('PEFT|adapter config', 'ignored keys|unexpected keys'),
    ),
    HardLessonGuardrail(
        symptom='OOM fixes can make eval complete without proving model quality',
        kind='eval-interpretation-boundary',
        guardrail='Memory-stability success is reported separately from generated-answer quality.',
        refs=('ouroboros/eval/comparison.py', 'wiki/GPU-Guardrails.md'),
        remediation='Record completion/OOM status separately from candidate-vs-baseline quality gates.',
        signature_patterns=('OOM|out of memory|memory', 'eval complete|completed', 'quality|candidate'),
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
