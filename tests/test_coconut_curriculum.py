from ouroboros.coconut.curriculum import build_stage_sample, partition_stage_shard


def test_build_stage_sample_splits_latent_and_visible_steps():
    sample = build_stage_sample({"question": "q", "steps": ["s1", "s2", "s3"], "answer_full": "a"}, stage_k=2)
    assert sample.latent_steps == ["s1", "s2"]
    assert sample.visible_steps == ["s3"]
    assert sample.answer_full == "a"


def test_partition_stage_shard_matches_three_way_projection():
    shard_a = partition_stage_shard(total_samples=10, total_seen=0, worker_id="A")
    shard_b = partition_stage_shard(total_samples=10, total_seen=0, worker_id="B")
    shard_c = partition_stage_shard(total_samples=10, total_seen=0, worker_id="C")
    assert (shard_a.start, shard_a.end, shard_a.size) == (0, 4, 4)
    assert (shard_b.start, shard_b.end, shard_b.size) == (4, 7, 3)
    assert (shard_c.start, shard_c.end, shard_c.size) == (7, 10, 3)
