import numpy as np

from zgq.search import compute_ground_truth, compute_recall


def test_recall_floor_vs_brute_force(built_index, data):
    vectors, queries = data
    k = 10
    gt_ids, _ = compute_ground_truth(vectors, queries, k=k)
    ids, _ = built_index.batch_search(queries, k=k, ef_search=128)
    assert compute_recall(ids, gt_ids, k=k) >= 90.0


def test_zone_path_recall_floor(built_index, data):
    vectors, queries = data
    k = 10
    gt_ids, _ = compute_ground_truth(vectors, queries, k=k)
    ids = np.stack([built_index.search(q, k=k, n_probe=8)[0] for q in queries])
    assert compute_recall(ids, gt_ids, k=k) >= 70.0
