"""Ouroboros Coconut adapter workflow."""

from ouroboros.coconut import (
    DEFAULT_BASE_MODEL,
    DEFAULT_LATENT_TOKEN,
    DEFAULT_LORA_TARGETS,
    CoconutCollator,
    GenerationResult,
    JsonlCoconutDataset,
    OuroborosCoconutForCausalLM,
    build_features,
    generate,
    load_lora_coconut,
    load_published_coconut,
    load_rows,
    push_release_bundle,
    save_release_bundle,
)

__all__ = (
    "DEFAULT_BASE_MODEL",
    "DEFAULT_LATENT_TOKEN",
    "DEFAULT_LORA_TARGETS",
    "CoconutCollator",
    "GenerationResult",
    "JsonlCoconutDataset",
    "OuroborosCoconutForCausalLM",
    "build_features",
    "generate",
    "load_lora_coconut",
    "load_published_coconut",
    "load_rows",
    "push_release_bundle",
    "save_release_bundle",
)
