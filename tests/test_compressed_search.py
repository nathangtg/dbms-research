import numpy as np
import pytest
from zgq import ZGQConfig, ZGQIndex
from zgq import memory_model as mm
from zgq.core import compressed as cz
from zgq.search import compute_recall

N, D, K = 3000, 32, 10


@pytest.fixture(scope="module")
def data():
    rng = np.random.RandomState(3)
    centers = rng.randn(30, D).astype(np.float32) * 2
    x = centers[rng.randint(0, 30, N)] + rng.randn(N, D).astype(np.float32) * 0.5
    q = centers[rng.randint(0, 30, 100)] + rng.randn(100, D).astype(np.float32) * 0.5
    d2 = ((q[:, None, :] - x[None]) ** 2).sum(-1)
    gt = np.argsort(d2, axis=1)[:, :K]
    return x, q, gt


def _cfg(**kw):
    base = dict(
        M=16,
        ef_construction=100,
        random_state=0,
        compressed_search=True,
        insertion_order="random",
    )
    base.update(kw)
    return ZGQConfig(**base)


@pytest.fixture(scope="module")
def index(data):
    return ZGQIndex(_cfg(rerank_budget=40)).build(data[0])


def test_defaults_are_off():
    c = ZGQConfig()
    assert c.compressed_search is False
    assert c.drop_full_vectors is False
    assert c.rerank_budget == 0


def test_config_validation():
    with pytest.raises(ValueError):
        ZGQConfig(drop_full_vectors=True)  # needs compressed_search
    with pytest.raises(ValueError):
        ZGQConfig(rerank_budget=-1)


def test_extracted_links_are_valid(index):
    links = index._cmp["links"]
    assert links.shape == (N, 2 * 16)
    assert links.max() < N and links.min() >= -1
    assert (links >= 0).sum(1).mean() > 8
    # no self links
    assert not (links == np.arange(N)[:, None]).any()


def test_adc_equals_distance_to_decoded_vector(index, data):
    x, q, _ = data
    cmp = index._cmp
    ids, adc = cz.beam_search_batch(
        cmp["links"],
        index.pq_codes,
        cmp["zone_of"],
        *index._tables(q[:20])[:2],
        cmp["cross"],
        index._entries(q[:20]),
        64,
    )
    recon = (
        index.pq.decode(index.pq_codes)
        + index.zones.fine_centroids[index.zones.assignments]
    )
    for i in range(20):
        valid = ids[i] >= 0
        ref = ((q[i] - recon[ids[i][valid]]) ** 2).sum(1)
        np.testing.assert_allclose(adc[i][valid], ref, rtol=1e-3, atol=1e-3)


def test_nearest_zone_table_is_wrong_for_other_zones(index, data):
    x, q, _ = data
    pq, zones = index.pq, index.zones
    recon = pq.decode(index.pq_codes) + zones.fine_centroids[zones.assignments]
    err_wrong, err_right = [], []
    for i in range(30):
        z0 = zones.select_zones(q[i], 1)[0]
        table = pq.compute_distance_table(q[i], zones.fine_centroids[z0])
        wrong = pq.asymmetric_distance(index.pq_codes, table)
        truth = ((q[i] - recon) ** 2).sum(1)
        other = zones.assignments != z0
        err_wrong.append(np.abs(wrong[other] - truth[other]).mean())
        # same-zone candidates are scored correctly by the nearest-zone table
        same = ~other
        if same.any():
            err_right.append(np.abs(wrong[same] - truth[same]).mean())
    assert np.mean(err_right) < 1e-2
    assert np.mean(err_wrong) > 100 * max(np.mean(err_right), 1e-6)


def test_compressed_recall_with_rerank(index, data):
    _, q, gt = data
    ids, _ = index.batch_search(q, K, ef_search=64)
    assert compute_recall(ids, gt, K) > 90.0


def test_codes_only_recall_reasonable(data):
    x, q, gt = data
    idx = ZGQIndex(_cfg(pq_m=16)).build(x)
    ids, d = idx.batch_search(q, K, ef_search=64)
    assert ids.shape == (len(q), K)
    assert compute_recall(ids, gt, K) > 50.0


def test_non_residual_variant_runs(data):
    x, q, gt = data
    idx = ZGQIndex(_cfg(pq_m=16, use_residual_pq=False)).build(x)
    assert idx._cmp["cross"] is None
    ids, _ = idx.batch_search(q, K, ef_search=64)
    assert compute_recall(ids, gt, K) > 50.0


def test_drop_full_vectors_frees_storage_and_still_searches(data):
    x, q, gt = data
    idx = ZGQIndex(_cfg(pq_m=16, drop_full_vectors=True)).build(x)
    assert idx.vectors is None and idx.vector_norms is None and idx.graph is None
    ids, _ = idx.batch_search(q, K, ef_search=64)
    assert compute_recall(ids, gt, K) > 50.0
    with pytest.raises(RuntimeError):
        idx.save("/tmp/never.zgq")


def test_drop_keeps_vectors_only_when_rerank_requested(data):
    x, q, gt = data
    idx = ZGQIndex(_cfg(pq_m=16, drop_full_vectors=True, rerank_budget=40)).build(x)
    assert idx.vectors is not None and idx.graph is None
    ids, _ = idx.batch_search(q, K, ef_search=64)
    assert compute_recall(ids, gt, K) > 85.0


@pytest.mark.parametrize("residual", [True, False])
@pytest.mark.parametrize("rerank", [0, 40])
def test_actual_bytes_match_model(data, residual, rerank):
    x, _, _ = data
    idx = ZGQIndex(
        _cfg(
            pq_m=16,
            use_residual_pq=residual,
            drop_full_vectors=True,
            rerank_budget=rerank,
        )
    ).build(x)
    actual = idx.compressed_memory_bytes()
    model = mm.zgq_compressed_bytes(
        N,
        D,
        M=16,
        pq_m=16,
        n_zones=len(idx.zones.fine_centroids),
        residual=residual,
        keep_vectors=rerank > 0,
    )
    assert actual["total_bytes"] == model["total_bytes"]
    assert actual["components"] == model["components"]
