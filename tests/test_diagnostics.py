import numpy as np
import pytest

from benchmarks.diagnostics import (
    fraction_outside_zones,
    oracle_zone_coverage,
    nearest_centroid_zones,
)
from zgq.core.graph import GraphConfig, ZoneGuidedGraph


def test_fraction_outside_zones_basic():
    assignments = np.array([0, 0, 1, 1, 2, 2])
    gt = np.array([[0, 1], [0, 2], [4, 5]])
    probed = [np.array([0]), np.array([0, 1]), np.array([0])]
    # q0: both in zone 0 -> 0 outside; q1: ids 0 (z0), 2 (z1) both probed -> 0;
    # q2: ids 4, 5 in zone 2, probed {0} -> 1.0 outside
    assert fraction_outside_zones(gt, assignments, probed) == pytest.approx(1.0 / 3)


def test_fraction_outside_zones_partial():
    assignments = np.array([0, 1, 1, 2])
    gt = np.array([[0, 1, 2, 3]])
    probed = [np.array([1])]
    assert fraction_outside_zones(gt, assignments, probed) == pytest.approx(0.5)


def test_oracle_zone_coverage():
    assignments = np.array([0, 0, 1, 1])
    gt = np.array([[0, 1], [2, 0]])
    # q0: zone of gt[0] is 0, both in zone 0 -> 1.0; q1: zone 1, 1 of 2 -> 0.5
    assert oracle_zone_coverage(gt, assignments) == pytest.approx(0.75)


def test_nearest_centroid_zones():
    centroids = np.array([[0.0, 0.0], [10.0, 0.0], [0.0, 10.0]], dtype=np.float32)
    queries = np.array([[9.0, 1.0], [0.5, 0.5]], dtype=np.float32)
    out = nearest_centroid_zones(queries, centroids, 2)
    assert out.shape == (2, 2)
    assert out[0, 0] == 1 and out[1, 0] == 0


def _graph(order, n=300, d=8, zones=5):
    rng = np.random.RandomState(1)
    v = rng.randn(n, d).astype(np.float32)
    assign = rng.randint(0, zones, n)
    cents = np.stack([v[assign == z].mean(0) for z in range(zones)])
    cfg = GraphConfig(M=8, ef_construction=50, insertion_order=order)
    return ZoneGuidedGraph(cfg).build(v, assign, cents), v


def test_insertion_order_default_is_zone():
    assert GraphConfig().insertion_order == "zone"


def test_random_insertion_builds_and_finds_self():
    g, v = _graph("random")
    ids, _ = g.search_batch(v[:20], 1, ef_search=50)
    assert (ids[:, 0] == np.arange(20)).mean() > 0.9


def test_invalid_insertion_order_rejected():
    with pytest.raises(ValueError):
        _graph("bogus")
