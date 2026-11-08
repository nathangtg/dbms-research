import numpy as np

from zgq import ZGQConfig, ZGQIndex


def _idx(data, **kw):
    vecs, _ = data
    return ZGQIndex(ZGQConfig(n_zones=16, use_hierarchy=False, ef_construction=100,
                              random_state=0, search_mode="zone_scan", **kw)).build(vecs)


def test_probe_count_varies_with_ratio(data):
    _, qs = data
    idx = _idx(data, adaptive_probe=True, min_probe=1, max_probe=12, probe_ratio=1.1)
    idx.batch_zone_search(qs, 10)
    p = idx.last_batch_stats["probes"]
    assert p.min() >= 1 and p.max() <= 12
    assert len(set(p.tolist())) > 1  # counts differ between queries
    # probing count is monotone in ratio threshold
    idx.config.probe_ratio = 1.5
    idx.batch_zone_search(qs, 10)
    assert idx.last_batch_stats["probes"].sum() > p.sum()


def test_adaptive_bounds_and_far_query(data):
    idx = _idx(data, adaptive_probe=True, min_probe=3, max_probe=5, probe_ratio=1.0)
    ids, _ = idx.search(data[1][0], k=5)
    assert len(ids) == 5
    idx.batch_zone_search(data[1], 5)
    assert np.all(idx.last_batch_stats["probes"] == 3)  # ratio 1.0 -> only nearest, clamped to min


def test_adaptive_recall_between_fixed_bounds(data):
    vecs, qs = data
    idx = _idx(data, adaptive_probe=True, min_probe=1, max_probe=16, probe_ratio=1.2)
    d = ((qs[:, None, :] - vecs[None]) ** 2).sum(-1)
    truth = np.argsort(d, axis=1)[:, :10]
    ids, _ = idx.batch_zone_search(qs, 10, use_pq_rerank=False)
    rec_ad = np.mean([len(set(a) & set(b)) / 10 for a, b in zip(ids, truth)])
    fixed1 = _idx(data)
    ids1, _ = fixed1.batch_zone_search(qs, 10, n_probe=1, use_pq_rerank=False)
    rec1 = np.mean([len(set(a) & set(b)) / 10 for a, b in zip(ids1, truth)])
    assert rec_ad >= rec1


def test_adaptive_save_load(data, tmp_path):
    idx = _idx(data, adaptive_probe=True, min_probe=2, max_probe=9, probe_ratio=1.25)
    p = str(tmp_path / "a.zgq")
    idx.save(p)
    c = ZGQIndex.load(p).config
    assert (c.adaptive_probe, c.min_probe, c.max_probe, c.probe_ratio) == (True, 2, 9, 1.25)
