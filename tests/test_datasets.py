"""Dataset loader specs. Offline: uses a tiny HDF5 fixture, no network."""
import hashlib
import json

import numpy as np
import pytest

h5py = pytest.importorskip("h5py")

from benchmarks import datasets as ds  # noqa: E402


def _write(path, distance="euclidean", n=300, nq=20, dim=8, k=12, seed=0, dtype=np.float64):
    rng = np.random.RandomState(seed)
    train = (rng.randn(n, dim) * 3 + 1).astype(dtype)
    test = (rng.randn(nq, dim) * 3 + 1).astype(dtype)
    if distance == "angular":
        tn = train / np.linalg.norm(train, axis=1, keepdims=True)
        qn = test / np.linalg.norm(test, axis=1, keepdims=True)
    else:
        tn, qn = train, test
    d2 = ((qn[:, None, :] - tn[None, :, :]) ** 2).sum(-1)
    nb = np.argsort(d2, axis=1)[:, :k]
    with h5py.File(path, "w") as f:
        f["train"] = train
        f["test"] = test
        f["neighbors"] = nb
        f["distances"] = np.sqrt(np.take_along_axis(d2, nb, axis=1))
        f.attrs["distance"] = distance
    return train, test


def test_euclidean_untouched_float32_contiguous(tmp_path):
    p = tmp_path / "a.hdf5"
    train, test = _write(p)
    d = ds.load_hdf5(p, name="tiny")
    assert d.vectors.dtype == np.float32 and d.vectors.flags["C_CONTIGUOUS"]
    assert d.queries.dtype == np.float32 and d.queries.flags["C_CONTIGUOUS"]
    assert d.metric == "l2"
    np.testing.assert_allclose(d.vectors, train.astype(np.float32))
    assert d.gt.shape == (20, 10)


def test_angular_is_normalized_and_l2(tmp_path):
    p = tmp_path / "g.hdf5"
    _write(p, distance="angular")
    d = ds.load_hdf5(p, name="tiny-angular")
    assert d.metric == "l2"
    np.testing.assert_allclose(np.linalg.norm(d.vectors, axis=1), 1.0, atol=1e-5)
    np.testing.assert_allclose(np.linalg.norm(d.queries, axis=1), 1.0, atol=1e-5)


def test_shape_validation_raises(tmp_path):
    p = tmp_path / "a.hdf5"
    _write(p)
    with pytest.raises(ValueError):
        ds.load_hdf5(p, name="x", expect_train=(301, 8))
    with pytest.raises(ValueError):
        ds.load_hdf5(p, name="x", expect_test=(20, 9))
    d = ds.load_hdf5(p, name="x", expect_train=(300, 8), expect_test=(20, 8))
    assert d.vectors.shape == (300, 8)


def test_full_size_reuses_file_neighbors(tmp_path):
    p = tmp_path / "a.hdf5"
    _write(p)
    with h5py.File(p) as f:
        nb = f["neighbors"][:, :10]
    d = ds.load_hdf5(p, name="x")
    np.testing.assert_array_equal(d.gt, nb)


def test_subset_recomputes_exact_gt(tmp_path):
    p = tmp_path / "a.hdf5"
    train, _ = _write(p)
    d = ds.load_hdf5(p, name="x", n_base=100, n_queries=7, seed=3)
    assert d.vectors.shape == (100, 8) and d.queries.shape == (7, 8) and d.gt.shape == (7, 10)
    d2 = ds.load_hdf5(p, name="x", n_base=100, n_queries=7, seed=3)
    np.testing.assert_array_equal(d.vectors, d2.vectors)
    d3 = ds.load_hdf5(p, name="x", n_base=100, n_queries=7, seed=4)
    assert not np.array_equal(d.vectors, d3.vectors)
    v = d.vectors.astype(np.float64)
    q = d.queries.astype(np.float64)
    naive = np.argsort(((q[:, None] - v[None]) ** 2).sum(-1), axis=1)[:, :10]
    np.testing.assert_array_equal(d.gt, naive)
    assert d.source["n_base"] == 100 and d.source["subset_seed"] == 3
    # subset rows are rows of the original train set
    f32 = train.astype(np.float32)
    assert all((f32 == row).all(axis=1).any() for row in d.vectors[:5])


def test_n_queries_without_subset_uses_file_gt_prefix(tmp_path):
    p = tmp_path / "a.hdf5"
    _write(p)
    d = ds.load_hdf5(p, name="x", n_queries=5)
    assert d.queries.shape == (5, 8) and d.gt.shape == (5, 10)


def test_cache_dir_env_then_home(tmp_path, monkeypatch):
    monkeypatch.setenv("ZGQ_DATA_DIR", str(tmp_path / "z"))
    assert ds.cache_dir() == tmp_path / "z"
    monkeypatch.delenv("ZGQ_DATA_DIR")
    assert ds.cache_dir().name == "zgq-datasets"
    assert ds.cache_dir().parent.name == ".cache"


def test_file_gt_check_passes_and_detects_corruption(tmp_path):
    p = tmp_path / "g.hdf5"
    _write(p, distance="angular")
    d = ds.load_hdf5(p, name="x")
    with h5py.File(p) as f:
        file_gt = f["neighbors"][:, :10]
    ok, worst = ds.verify_file_gt(d.vectors, d.queries, file_gt, n_check=200)
    assert ok and worst < 1e-4
    bad = file_gt[:, ::-1].copy()
    ok, _ = ds.verify_file_gt(d.vectors, d.queries, bad, n_check=200)
    assert not ok


def test_ensure_downloaded_skips_when_size_matches(tmp_path, monkeypatch):
    monkeypatch.setenv("ZGQ_DATA_DIR", str(tmp_path))
    payload = b"hello world"
    monkeypatch.setitem(
        ds.DATASETS, "fake", {"url": "http://x/fake.hdf5", "bytes": len(payload),
                              "train": (1, 1), "test": (1, 1)})
    calls = []

    def fake_dl(url, dest):
        calls.append(url)
        dest.write_bytes(payload)

    monkeypatch.setattr(ds, "_download", fake_dl)
    path = ds.ensure_downloaded("fake")
    assert calls == ["http://x/fake.hdf5"]
    side = json.loads((path.parent / (path.name + ".json")).read_text())
    assert side["sha256"] == hashlib.sha256(payload).hexdigest()
    assert side["bytes"] == len(payload)
    ds.ensure_downloaded("fake")
    assert len(calls) == 1


def test_ensure_downloaded_rejects_wrong_size(tmp_path, monkeypatch):
    monkeypatch.setenv("ZGQ_DATA_DIR", str(tmp_path))
    monkeypatch.setitem(
        ds.DATASETS, "fake2", {"url": "u", "bytes": 99, "train": (1, 1), "test": (1, 1)})
    monkeypatch.setattr(ds, "_download", lambda url, dest: dest.write_bytes(b"abc"))
    with pytest.raises(ValueError):
        ds.ensure_downloaded("fake2")


def test_registry_entries():
    s = ds.DATASETS["sift-128-euclidean"]
    assert s["bytes"] == 525128288 and s["train"] == (1000000, 128)
    g = ds.DATASETS["glove-100-angular"]
    assert g["bytes"] == 485413888 and g["train"] == (1183514, 100)


def test_synthetic_wraps_generator():
    d = ds.load_dataset("synthetic-500", n_queries=12, k=5, seed=7)
    assert d.vectors.shape == (500, 128) and d.queries.shape == (12, 128)
    assert d.gt.shape == (12, 5) and d.metric == "l2"
    assert d.vectors.dtype == np.float32
    d2 = ds.load_dataset("synthetic-500", n_queries=12, k=5, seed=7)
    np.testing.assert_array_equal(d.vectors, d2.vectors)


def test_unknown_name_raises():
    with pytest.raises(KeyError):
        ds.load_dataset("nope")
