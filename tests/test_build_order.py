import numpy as np
import pytest

from zgq.core.build_order import STRATEGIES, estimate_lid, insertion_order
from zgq import ZGQConfig, ZGQIndex


def _legacy_zone_order(vectors, assignments, centroids):
    zone_order = []
    for zone_id in range(len(centroids)):
        zone_vids = np.where(assignments == zone_id)[0]
        if len(zone_vids) > 0:
            zone_vectors = vectors[zone_vids]
            d = np.sum((zone_vectors - centroids[zone_id]) ** 2, axis=1)
            zone_order.extend(zone_vids[np.argsort(d)].tolist())
    return np.array(zone_order, dtype=np.int64)


@pytest.fixture(scope="module")
def toy():
    rng = np.random.RandomState(1)
    n, d, z = 300, 32, 6
    vectors = rng.randn(n, d).astype(np.float32)
    assignments = rng.randint(0, z, size=n)
    centroids = np.stack(
        [vectors[assignments == i].mean(axis=0) for i in range(z)]
    ).astype(np.float32)
    return vectors, assignments, centroids


def _order(name, toy, seed=3, **kw):
    v, a, c = toy
    return insertion_order(name, v, a, c, seed, **kw)


@pytest.mark.parametrize("name", STRATEGIES)
def test_every_strategy_is_permutation(name, toy):
    p = _order(name, toy)
    assert p.dtype == np.int64
    assert np.array_equal(np.sort(p), np.arange(len(toy[0])))


def test_zone_and_random_match_legacy(toy):
    v, a, c = toy
    assert np.array_equal(_order("zone", toy), _legacy_zone_order(v, a, c))
    assert np.array_equal(
        _order("random", toy, seed=7), np.random.RandomState(7).permutation(len(v))
    )


def test_skeleton(toy):
    v, a, c = toy
    n, seed, f = len(v), 5, 0.1
    perm = np.random.RandomState(seed).permutation(n)
    k = round(f * n)
    p = _order("skeleton", toy, seed=seed, skeleton_fraction=f)
    assert np.array_equal(p[:k], perm[:k])
    zone = _legacy_zone_order(v, a, c)
    assert np.array_equal(p[k:], zone[~np.isin(zone, perm[:k])])
    assert np.array_equal(_order("skeleton", toy, seed=seed, skeleton_fraction=0.0), zone)
    assert np.array_equal(
        _order("skeleton", toy, seed=seed, skeleton_fraction=1.0), perm
    )


def test_roundrobin(toy):
    p = _order("roundrobin", toy)
    z = len(toy[2])
    assert len(set(toy[1][p[:z]].tolist())) == z
    assert np.array_equal(p, _order("roundrobin", toy))


@pytest.fixture(scope="module")
def mixed():
    rng = np.random.RandomState(2)
    plane = np.zeros((150, 32), dtype=np.float32)
    plane[:, :2] = rng.randn(150, 2)
    full = rng.randn(150, 32).astype(np.float32) + 10.0
    return np.vstack([plane, full])


def test_lid_separates_plane_from_gaussian(mixed):
    lid = estimate_lid(mixed, k=20)
    assert lid.shape == (300,)
    assert lid[:150].mean() < lid[150:].mean()
    assert lid[:150].mean() < 5 < lid[150:].mean()


def test_lid_orders(mixed):
    n = len(mixed)
    a = np.zeros(n, dtype=np.int64)
    c = mixed[:1].copy()
    lid = estimate_lid(mixed, k=20)
    desc = insertion_order("lid_desc", mixed, a, c, 0, lid=lid)
    asc = insertion_order("lid_asc", mixed, a, c, 0, lid=lid)
    assert np.array_equal(asc, np.argsort(lid, kind="stable"))
    assert np.array_equal(desc, np.argsort(-lid, kind="stable"))
    assert lid[desc[0]] >= lid[desc[-1]]
    # computed internally when lid is not passed
    assert np.array_equal(insertion_order("lid_desc", mixed, a, c, 0), desc)


def test_deterministic_and_validation(toy):
    for name in STRATEGIES:
        assert np.array_equal(_order(name, toy), _order(name, toy))
    with pytest.raises(ValueError):
        _order("bogus", toy)
    for f in (-0.1, 1.5):
        with pytest.raises(ValueError):
            _order("skeleton", toy, skeleton_fraction=f)


def test_skeleton_index_builds_and_defaults_unchanged(data):
    vectors, queries = data
    cfg = ZGQConfig(n_zones=8, M=16, ef_construction=100, ef_search=64,
                    random_state=0, insertion_order="skeleton", skeleton_fraction=0.1)
    idx = ZGQIndex(cfg).build(vectors)
    ids, _ = idx.search(queries[0], k=5)
    assert len(ids) == 5
    d = ZGQConfig()
    assert d.insertion_order == "zone" and d.skeleton_fraction == 0.05
    with pytest.raises(ValueError):
        ZGQConfig(skeleton_fraction=2.0)
