"""Specs for the hnswlib file parser, graph statistics and instrumented search."""

import json

import hnswlib
import numpy as np
import pytest
from benchmarks.graph_stats import (
    analyze_graph,
    cross_zone_fraction,
    edge_length_ratios,
    reachable_count,
    read_hnsw_index,
    search_hops,
)

DIM, M = 16, 8


def _clustered(n, nq, seed=0):
    rng = np.random.RandomState(seed)
    centers = rng.randn(10, DIM).astype(np.float32) * 4
    x = centers[rng.randint(0, 10, n)] + rng.randn(n, DIM).astype(np.float32)
    q = centers[rng.randint(0, 10, nq)] + rng.randn(nq, DIM).astype(np.float32)
    return x.astype(np.float32), q.astype(np.float32)


@pytest.fixture(scope="module")
def built(tmp_path_factory):
    x, q = _clustered(2000, 100)
    order = np.random.RandomState(1).permutation(len(x))
    idx = hnswlib.Index(space="l2", dim=DIM)
    idx.init_index(max_elements=len(x), M=M, ef_construction=100, random_seed=7)
    idx.set_num_threads(1)
    idx.add_items(x[order], order)
    path = str(tmp_path_factory.mktemp("g") / "idx.bin")
    idx.save_index(path)
    return idx, read_hnsw_index(path), x, q, order


def test_parser_round_trip(built):
    idx, g, x, _, order = built
    n = len(x)
    assert g.n == n
    assert g.M == M and g.max_m0 == 2 * M
    assert set(g.labels.tolist()) == set(idx.get_ids_list())
    assert np.array_equal(g.labels, order)  # internal id = insertion rank
    assert g.counts.max() <= 2 * M and g.counts.min() >= 1
    valid = g.level0[g.level0 >= 0]
    assert valid.min() >= 0 and valid.max() < n
    assert np.array_equal(g.vectors, x[order])
    assert 0 <= g.entry_point < n
    assert g.max_level == g.levels.max() >= 1
    assert g.levels[g.entry_point] == g.max_level
    # Upper neighbors are valid nodes that exist at that level.
    for node in np.nonzero(g.levels)[0][:50]:
        for lv in range(1, g.levels[node] + 1):
            nb = g.level_neighbors(node, lv)
            assert len(nb) <= M
            assert np.all(g.levels[nb] >= lv)


def test_instrumented_search_matches_hnswlib_recall(built):
    idx, g, x, q, _ = built
    k, ef = 10, 64
    d = ((q[:, None, :] - x[None, :, :]) ** 2).sum(-1)
    gt = np.argsort(d, axis=1)[:, :k]
    idx.set_ef(ef)
    lib_ids, _ = idx.knn_query(q, k=k)
    res = search_hops(g, None, q, ef=ef, k=k)

    def recall(ids):
        return np.mean([len(set(a) & set(b)) / k for a, b in zip(ids, gt)])

    assert abs(recall(res["ids"]) - recall(lib_ids)) <= 0.02
    assert (res["hops_level0"] > 0).all() and (
        res["dist_comps"] > res["hops_level0"]
    ).all()
    assert res["summary"]["dist_comps"]["p95"] >= res["summary"]["dist_comps"]["median"]


def test_reachability_fixture():
    # 0->1->2, 3->0, 4 isolated, 2->1 (cycle). From 0: {0,1,2}.
    nb = np.array([[1, -1], [2, -1], [1, -1], [0, -1], [-1, -1]], dtype=np.int32)
    counts = np.array([1, 1, 1, 1, 0], dtype=np.int32)
    assert reachable_count(nb, counts, 0) == 3
    assert reachable_count(nb, counts, 3) == 4
    assert reachable_count(nb, counts, 4) == 1


def test_cross_zone_fraction_fixture():
    nb = np.array([[1, 2], [0, -1], [3, -1], [2, -1]], dtype=np.int32)
    counts = np.array([2, 1, 1, 1], dtype=np.int32)
    zones = np.array([0, 0, 1, 1])
    # edges: 0-1 same, 0-2 cross, 1-0 same, 2-3 same, 3-2 same -> 1/5
    assert cross_zone_fraction(nb, counts, zones) == pytest.approx(0.2)


def test_edge_length_ratio_definition():
    # node A: edges 1,3 with 10th-NN distance 2 -> 0.5, 1.5; node B: edge 8, kth 4 -> 2
    edge = np.array([[1.0, 3.0], [8.0, np.nan]])
    kth = np.array([2.0, 4.0])
    r = edge_length_ratios(edge, kth)
    assert sorted(r.tolist()) == [0.5, 1.5, 2.0]


def test_analyze_graph_is_json_serializable(built):
    _, g, x, _, order = built
    zones_by_label = np.arange(len(x)) % 5
    stats = analyze_graph(g, g.vectors, zones_by_label[g.labels], seed=0, n_sample=200)
    json.dumps(stats)
    assert 0.0 <= stats["reachable_fraction_from_entry"] <= 1.0
    assert stats["n_weak_components"] >= 1
    assert stats["out_degree"]["p99"] <= 2 * M
    assert 0.0 <= stats["in_degree_gini"] <= 1.0
    assert stats["nodes_per_level"][0] == len(x)
    assert stats["entry_point_zone"] in range(5)
    assert stats["edge_length_ratio"]["p50"] > 0
