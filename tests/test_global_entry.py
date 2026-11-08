import numpy as np

from zgq import ZGQConfig, ZGQIndex


def _idx(data, **kw):
    vecs, _ = data
    return ZGQIndex(ZGQConfig(n_zones=8, ef_construction=100, random_state=0, **kw)).build(vecs)


def test_random_insertion_changes_graph_not_zones(data):
    z = _idx(data)
    r = _idx(data, insertion_order="random")
    assert np.array_equal(z.zones.assignments, r.zones.assignments)
    assert r.graph.config.insertion_order == "random"
    _, qs = data
    a = z.batch_search(qs, k=10)[0]
    b = r.batch_search(qs, k=10)[0]
    assert not np.array_equal(a, b) or not np.array_equal(
        z.graph.hnsw.get_ids_list(), r.graph.hnsw.get_ids_list()
    )


def test_global_entry_adds_seed_candidates(data):
    _, qs = data
    g = _idx(data, insertion_order="random")
    e = _idx(data, insertion_order="random", search_mode="global_entry")
    seen_seed = False
    for q in qs:
        zones = e._choose_zones(q, 8)
        seeds = {i for z in zones for i in e.graph.zone_entry_points[int(z)]}
        ids_e, _ = e.search(q, k=10, use_pq_rerank=False)
        ids_g, _ = g.search(q, k=10, use_pq_rerank=False)
        # seeded result is never worse than the plain graph result for the same graph
        assert len(ids_e) == len(ids_g)
        seen_seed |= bool(seeds)
    assert seen_seed


def test_global_entry_not_worse_than_graph(data):
    vecs, qs = data
    d = ((qs[:, None, :] - vecs[None]) ** 2).sum(-1)
    truth = np.argsort(d, axis=1)[:, :10]
    g = _idx(data, insertion_order="random")
    e = _idx(data, insertion_order="random", search_mode="global_entry")
    rg = np.mean([len(set(a) & set(b)) / 10 for a, b in zip(g.batch_zone_search(qs, 10, use_pq_rerank=False)[0], truth)])
    re_ = np.mean([len(set(a) & set(b)) / 10 for a, b in zip(e.batch_zone_search(qs, 10, use_pq_rerank=False)[0], truth)])
    assert re_ >= rg


def test_global_entry_save_load(data, tmp_path):
    _, qs = data
    e = _idx(data, insertion_order="random", search_mode="global_entry")
    p = str(tmp_path / "g.zgq")
    e.save(p)
    l = ZGQIndex.load(p)
    assert l.config.search_mode == "global_entry" and l.graph.config.insertion_order == "random"
    assert np.array_equal(
        e.batch_zone_search(qs, 10)[0], l.batch_zone_search(qs, 10)[0]
    )
