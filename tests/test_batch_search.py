import numpy as np

from zgq import ZGQSearch
from zgq.search import SearchConfig


def test_index_batch_search_n_jobs_matches_sequential(built_index, data):
    _, queries = data
    kwargs = dict(k=5, fast_mode=False)
    ids1, d1 = built_index.batch_search(queries, n_jobs=1, **kwargs)
    ids4, d4 = built_index.batch_search(queries, n_jobs=4, **kwargs)
    assert ids1.shape == (len(queries), 5)
    assert np.array_equal(ids1, ids4)
    assert np.allclose(d1, d4)


def test_search_class_batch_search_n_jobs(built_index, data):
    _, queries = data
    s = ZGQSearch(
        built_index.zones,
        built_index.graph,
        built_index.vectors,
        pq=built_index.pq,
        pq_codes=built_index.pq_codes,
        vector_norms=built_index.vector_norms,
        config=SearchConfig(),
    )
    ids1, _ = s.batch_search(queries, k=5, n_jobs=1)
    ids4, _ = s.batch_search(queries, k=5, n_jobs=4)
    assert ids1.shape == (len(queries), 5)
    assert np.array_equal(ids1, ids4)
