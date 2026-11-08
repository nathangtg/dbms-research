import numpy as np

from zgq import ZGQIndex


def test_save_load_roundtrip(built_index, data, tmp_path):
    _, queries = data
    path = str(tmp_path / "idx.zgq")
    built_index.save(path)
    loaded = ZGQIndex.load(path)
    assert loaded.is_built
    assert loaded.n_vectors == built_index.n_vectors
    for q in queries[:5]:
        a, _ = built_index.search(q, k=10)
        b, _ = loaded.search(q, k=10)
        assert np.array_equal(a, b)
