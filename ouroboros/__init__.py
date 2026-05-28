"""Ouroboros: a compact latent-reasoning research core."""

from ouroboros.callbacks import push_release_bundle, save_release_bundle
from ouroboros.config import DEFAULT_BASE_MODEL, DEFAULT_LATENT_TOKEN, DEFAULT_LORA_TARGETS
from ouroboros.data import CoconutCollator, JsonlCoconutDataset, PromptFeature, build_features, load_rows
from ouroboros.generation import GenerationResult, generate
from ouroboros.latent import HaltGate, OuroborosCoconutForCausalLM, load_lora_coconut, load_published_coconut

__all__ = (
    "DEFAULT_BASE_MODEL",
    "DEFAULT_LATENT_TOKEN",
    "DEFAULT_LORA_TARGETS",
    "CoconutCollator",
    "GenerationResult",
    "HaltGate",
    "JsonlCoconutDataset",
    "OuroborosCoconutForCausalLM",
    "PromptFeature",
    "build_features",
    "generate",
    "load_lora_coconut",
    "load_published_coconut",
    "load_rows",
    "push_release_bundle",
    "save_release_bundle",
)
