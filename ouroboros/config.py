"""Configuration defaults for the compact Ouroboros research core."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

DEFAULT_BASE_MODEL = "ai21labs/AI21-Jamba-Reasoning-3B"
DEFAULT_LATENT_TOKEN = "<|lat|>"
DEFAULT_LORA_TARGETS = (
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",
    "in_proj",
    "x_proj",
    "dt_proj",
    "out_proj",
)


@dataclass(frozen=True)
class ModelConfig:
    base_model: str = DEFAULT_BASE_MODEL
    latent_token: str = DEFAULT_LATENT_TOKEN
    lora_r: int = 32
    lora_alpha: int = 64
    lora_dropout: float = 0.05
    target_modules: tuple[str, ...] = DEFAULT_LORA_TARGETS
    load_in_4bit: bool = False
    trust_remote_code: bool = True


@dataclass(frozen=True)
class CurriculumConfig:
    stages: tuple[int, ...] = tuple(range(0, 11))
    epochs_per_stage: int = 1


@dataclass(frozen=True)
class DgacConfig:
    enabled: bool = False
    halt_threshold: float = 0.5
    lambda_ponder_max: float = 0.01
    lambda_diversity: float = 0.1
    tau: float = 0.9
    warmup_steps: int = 200
    ramp_steps: int = 300


def split_csv(value: str | Iterable[str]) -> list[str]:
    if isinstance(value, str):
        return [part.strip() for part in value.split(",") if part.strip()]
    return [str(part).strip() for part in value if str(part).strip()]


def parse_stage_spec(value: str | None, *, max_stage: int | None = None) -> tuple[int, ...]:
    """Parse ``0,1,2`` or ``0-10`` into an explicit curriculum tuple."""
    if value:
        stages: list[int] = []
        for part in split_csv(value):
            if "-" in part:
                left, right = part.split("-", 1)
                start, end = int(left), int(right)
                step = 1 if end >= start else -1
                stages.extend(range(start, end + step, step))
            else:
                stages.append(int(part))
        return tuple(dict.fromkeys(max(0, stage) for stage in stages))
    end = 10 if max_stage is None else int(max_stage)
    return tuple(range(0, max(0, end) + 1))
