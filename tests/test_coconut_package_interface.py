from __future__ import annotations

from ouroboros import coconut
from ouroboros.coconut import checkpointing, evaluation, session, stage_runner
from ouroboros.coconut import latent as latent_module


def test_coconut_package_exposes_training_latent_dgac_checkpoint_interface():
    assert coconut.save_checkpoint is checkpointing.save_checkpoint
    assert coconut.load_checkpoint is checkpointing.load_checkpoint
    assert coconut.evaluate_stage is evaluation.evaluate_stage
    assert coconut.run_generation_callback is evaluation.run_generation_callback
    assert coconut.run_training_stages is stage_runner.run_training_stages
    assert coconut.run_cli is session.run_training_session
    assert coconut.HaltGate.__module__ == "ouroboros.coconut.dgac"
    assert coconut.prepare_latent_runtime is latent_module.prepare_latent_runtime
    assert coconut.forward_latent_batch is latent_module.forward_latent_batch
    assert coconut.decode_from_latent_context is latent_module.decode_from_latent_context
