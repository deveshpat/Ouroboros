from ouroboros.diloco.observability import coordinator_wandb_identity, redact_mapping, worker_wandb_identity


def test_worker_identity_is_unique_per_round_but_grouped_per_stage_worker():
    r0 = worker_wandb_identity(worker_id="A", stage_k=3, round_n=0, shard_step_estimate=100, project="p")
    r1 = worker_wandb_identity(worker_id="A", stage_k=3, round_n=1, shard_step_estimate=100, project="p")
    assert r0.run_id != r1.run_id
    assert r0.group == r1.group == "diloco-worker-A-s3"
    assert r1.step_offset == 100


def test_coordinator_identity_is_stage_scoped():
    ident = coordinator_wandb_identity(stage_k=4, project="p", entity="e")
    assert ident.run_id == "diloco-coordinator-s4"
    assert ident.entity == "e"


def test_redaction_masks_secret_like_keys():
    assert redact_mapping({"HF_TOKEN": "abc", "normal": 1}) == {"HF_TOKEN": "***", "normal": 1}
