import numpy as np
import pytest

from zgq import ZGQConfig, ZGQIndex


@pytest.fixture(scope="module")
def boundary_data():
    # Dense blobs with queries on the segment between blob centers
    rng = np.random.RandomState(1)
    centers = rng.randn(8, 16).astype(np.float32) * 3
    vecs = np.concatenate([c + 0.9 * rng.randn(150, 16) for c in centers]).astype(np.float32)
    a, b = rng.randint(0, 8, 60), rng.randint(0, 8, 60)
    qs = (0.5 * (centers[a] + centers[b]) + 0.3 * rng.randn(60, 16)).astype(np.float32)
    return vecs, qs


def _truth(vecs, qs, k):
    d = ((qs[:, None, :] - vecs[None]) ** 2).sum(-1)
    return np.argsort(d, axis=1)[:, :k]


def _recall(ids, truth):
    return np.mean([len(set(a) & set(b)) / len(b) for a, b in zip(ids, truth)])


def _build(vecs, **kw):
    return ZGQIndex(ZGQConfig(n_zones=8, use_hierarchy=False, ef_construction=100,
                              random_state=0, search_mode="zone_scan", **kw)).build(vecs)


def test_replicas_membership(boundary_data):
    vecs, _ = boundary_data
    idx = _build(vecs, zone_replicas=2)
    z = idx.zones
    extra = z.replica_assignments
    assert extra.shape == (len(vecs), 1)
    assert not np.any(extra[:, 0] == z.assignments)  # secondary differs from primary
    ov = z.get_replica_overhead()
    assert ov["extra_ids"] == len(vecs) and ov["extra_bytes"] == 4 * len(vecs)
    assert sum(len(v) for v in z.replica_lists.values()) == len(vecs)
    # primary assignments and inverted lists unchanged by replication
    base = _build(vecs)
    assert np.array_equal(base.zones.assignments, z.assignments)


def test_replicas_increase_recall_and_dedupe(boundary_data):
    vecs, qs = boundary_data
    truth = _truth(vecs, qs, 10)
    r1 = _recall(_build(vecs).batch_zone_search(qs, 10, n_probe=1, use_pq_rerank=False)[0], truth)
    idx2 = _build(vecs, zone_replicas=2)
    ids2, _ = idx2.batch_zone_search(qs, 10, n_probe=1, use_pq_rerank=False)
    assert _recall(ids2, truth) > r1 + 0.02
    for row in ids2:
        assert len(set(row.tolist())) == len(row)  # no duplicate ids from replicas
    cand = idx2.zones.get_zone_candidates([0, 1, 2])
    assert len(cand) == len(np.unique(cand))


def test_replicas_save_load(boundary_data, tmp_path):
    vecs, qs = boundary_data
    idx = _build(vecs, zone_replicas=2)
    p = str(tmp_path / "r.zgq")
    idx.save(p)
    loaded = ZGQIndex.load(p)
    assert loaded.config.zone_replicas == 2 and loaded.config.search_mode == "zone_scan"
    assert np.array_equal(loaded.zones.replica_assignments, idx.zones.replica_assignments)
    a = idx.batch_zone_search(qs, 10, n_probe=2)[0]
    b = loaded.batch_zone_search(qs, 10, n_probe=2)[0]
    assert np.array_equal(a, b)


def test_invalid_options():
    with pytest.raises(ValueError):
        ZGQConfig(zone_replicas=0)
    with pytest.raises(ValueError):
        ZGQConfig(search_mode="bogus")
    with pytest.raises(ValueError):
        ZGQConfig(insertion_order="bogus")
