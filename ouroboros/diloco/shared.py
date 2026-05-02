"""
Shared DiLoCo primitives.
Imported by both diloco_coordinator.py and jamba_coconut_finetune.py.
Zero third-party dependencies at import time (stdlib only).
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple, TypeVar

WORKER_IDS: Tuple[str, ...] = ("A", "B", "C")

T = TypeVar("T")


def normalize_text(value: Optional[Any], *, uppercase: bool = False) -> Optional[str]:
    """Canonical text normalization. Replaces _normalize_optional_text in both files."""
    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return None
    return text.upper() if uppercase else text


def ordered_unique_workers(*groups: Optional[List[str]]) -> List[str]:
    """Canonical worker ID deduplication. Replaces _ordered_unique_* in both files."""
    ordered: List[str] = []
    seen = set()
    for group in groups:
        for worker_id in group or []:
            wid = str(worker_id).upper()
            if wid not in WORKER_IDS or wid in seen:
                continue
            ordered.append(wid)
            seen.add(wid)
    return ordered


def retry_io(
    label: str,
    fn: Callable[[], T],
    *,
    attempts: int = 3,
    base_delay_s: float = 1.5,
    swallow: bool = False,
    default: Optional[T] = None,
    verbose: bool = True,
) -> Optional[T]:
    """
    Unified retry primitive. Replaces _retry_io (coordinator) and
    _retry_diloco_io (worker). verbose=False suppresses rank>0 noise.
    """
    last_exc: Optional[Exception] = None
    attempts = max(int(attempts), 1)
    for attempt in range(1, attempts + 1):
        try:
            return fn()
        except Exception as exc:  # noqa: BLE001 - callers decide which transient I/O to retry
            last_exc = exc
            if attempt >= attempts:
                if swallow:
                    if verbose:
                        print(
                            f"{label} failed after {attempts} attempts: "
                            f"{type(exc).__name__}: {exc}"
                        )
                    return default
                raise
            delay = base_delay_s * (2 ** (attempt - 1))
            if verbose:
                print(
                    f"{label} failed (attempt {attempt}/{attempts}): "
                    f"{type(exc).__name__}: {exc}. Retrying in {delay:.1f}s..."
                )
            time.sleep(delay)
    if swallow:
        return default
    assert last_exc is not None
    raise last_exc


@dataclass
class RoundState:
    """Typed, validated round state. Replaces inline dict parsing in both files."""

    stage_k: int = 0
    round_n: int = 0
    anchor_path: str = "diloco_state/anchor"
    total_samples_seen: Dict[str, int] = field(default_factory=dict)
    completed_stages: List[int] = field(default_factory=list)
    triggered_workers: List[str] = field(default_factory=list)
    attendance_workers: List[str] = field(default_factory=list)
    triggered_at: float = 0.0
    mode: str = "diloco"
    seed: int = 42
    extra: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, raw: Dict[str, Any]) -> "RoundState":
        state = dict(raw or {})
        triggered_workers = ordered_unique_workers(state.get("triggered_workers"))
        attendance_workers = [
            w
            for w in ordered_unique_workers(state.get("attendance_workers"))
            if w not in set(triggered_workers)
        ]
        known = {
            "stage_k",
            "round_n",
            "anchor_path",
            "total_samples_seen",
            "completed_stages",
            "triggered_workers",
            "attendance_workers",
            "triggered_at",
            "mode",
            "seed",
        }
        return cls(
            stage_k=int(state.get("stage_k", 0)),
            round_n=int(state.get("round_n", 0)),
            anchor_path=str(state.get("anchor_path", "diloco_state/anchor")),
            total_samples_seen={
                str(k): int(v)
                for k, v in dict(state.get("total_samples_seen", {})).items()
            },
            completed_stages=[int(x) for x in state.get("completed_stages", [])],
            triggered_workers=triggered_workers,
            attendance_workers=attendance_workers,
            triggered_at=float(state.get("triggered_at", 0.0)),
            mode=str(state.get("mode", "diloco")),
            seed=int(state.get("seed", 42)),
            extra={k: v for k, v in state.items() if k not in known},
        )

    def to_dict(self) -> Dict[str, Any]:
        data: Dict[str, Any] = {
            "stage_k": int(self.stage_k),
            "round_n": int(self.round_n),
            "anchor_path": self.anchor_path,
            "total_samples_seen": {str(k): int(v) for k, v in self.total_samples_seen.items()},
            "completed_stages": [int(x) for x in self.completed_stages],
            "triggered_workers": ordered_unique_workers(self.triggered_workers),
            "attendance_workers": [
                w
                for w in ordered_unique_workers(self.attendance_workers)
                if w not in set(ordered_unique_workers(self.triggered_workers))
            ],
            "triggered_at": float(self.triggered_at),
            "mode": self.mode,
            "seed": int(self.seed),
        }
        data.update(self.extra)
        return data
