import numpy as np
import pytest

from zgq import ZGQConfig, ZGQIndex


def test_build_sets_state(built_index, data):
    vectors, _ = data
    assert built_index.is_built
    assert built_index.n_vectors == len(vectors)
    assert built_index.dimension == vectors.shape[1]


def test_search_shapes(built_index, data):
    _, queries = data
    ids, dists = built_index.search(queries[0], k=10)
    assert len(ids) == 10 and len(dists) == 10
    assert len(set(ids.tolist())) == 10
    assert np.all(np.diff(dists) >= -1e-5)


def test_batch_search_shapes(built_index, data):
    _, queries = data
    ids, dists = built_index.batch_search(queries, k=5)
    assert ids.shape == (len(queries), 5)
    assert dists.shape == (len(queries), 5)


def test_search_before_build_raises(data):
    _, queries = data
    with pytest.raises(RuntimeError):
        ZGQIndex(ZGQConfig()).search(queries[0], k=5)


def test_dimension_mismatch_raises(built_index):
    with pytest.raises(ValueError):
        built_index.batch_search(np.zeros((2, 7), dtype=np.float32), k=3)
