"""Default-config results must be bit-identical to the pre-change golden."""
import json
import os

import numpy as np
import pytest

from zgq import ZGQIndex

from zgq import ZGQConfig


@pytest.fixture(scope="module")
def built_index(data):
    # Own index: search mutates shared hnswlib ef state, so do not reuse the session one
    cfg = ZGQConfig(n_zones=8, M=16, ef_construction=100, ef_search=64, random_state=0)
    return ZGQIndex(cfg).build(data[0])


GOLDEN = json.load(open(os.path.join(os.path.dirname(__file__), "fixtures", "golden_default.json")))


def test_new_options_default_off():
    c = ZGQConfig()
    assert c.insertion_order == "zone"
    assert c.zone_replicas == 1
    assert c.adaptive_probe is False
    assert c.search_mode == "graph"


def test_default_zone_assignments_unchanged(built_index):
    assert built_index.zones.assignments.tolist() == GOLDEN["assign"]
    assert built_index.zones.replica_lists == {}


def test_default_search_identical(built_index, data):
    _, queries = data
    assert [built_index.search(q, k=10)[0].tolist() for q in queries] == GOLDEN["search"]
    results = [built_index.search(q, k=10, n_probe=4)[0].tolist() for q in queries]
    assert results == GOLDEN["search_np4"]
    assert [
        built_index.search(q, k=10, use_pq_rerank=False)[0].tolist() for q in queries
    ] == GOLDEN["search_nopq"]


def test_default_batch_search_identical(built_index, data):
    _, queries = data
    # hnswlib's multithreaded add_items makes the graph (and so the fast path)
    # vary run to run even before this change, so compare by overlap, not equality.
    fast = built_index.batch_search(queries, k=10)[0]
    overlap = np.mean([len(set(a) & set(b)) / 10 for a, b in zip(fast.tolist(), GOLDEN["fast"])])
    assert overlap >= 0.97
    assert built_index.batch_search(queries, k=10, fast_mode=False)[0].tolist() == GOLDEN["zone"]


def test_batch_zone_search_matches_search(built_index, data):
    _, queries = data
    ids, _ = built_index.batch_zone_search(queries, k=10)
    assert ids.tolist() == GOLDEN["search"]
    ids, _ = built_index.batch_zone_search(queries, k=10, n_probe=4, n_jobs=2)
    assert ids.tolist() == GOLDEN["search_np4"]
    ids, _ = built_index.batch_zone_search(queries, k=10, use_pq_rerank=False)
    assert ids.tolist() == GOLDEN["search_nopq"]
