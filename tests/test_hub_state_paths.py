from ouroboros.diloco.hub_state import InMemoryHubStateStore, worker_status_path, worker_weights_prefix


def test_worker_paths_are_canonicalized():
    assert worker_status_path("b") == "diloco_state/workers/B/status.json"
    assert worker_weights_prefix("c", 3, 9) == "diloco_state/workers/C/stage_3/round_9"


def test_in_memory_store_round_state_and_statuses():
    store = InMemoryHubStateStore()
    store.save_round_state({"stage_k": 3, "round_n": 0, "mode": "diloco", "total_samples_seen": {"3": 0}})
    store.upload_worker_status({"worker_id": "A", "stage_k": 3, "round_n": 0, "status": "done", "samples_seen": 10})
    assert store.load_round_state().stage_k == 3
    statuses = store.load_worker_statuses(["A", "B"])
    assert len(statuses) == 1
    assert statuses[0].worker_id == "A"
