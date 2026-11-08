import os

import faiss
import hnswlib
import numpy as np
import pytest
from zgq import memory_model as mm

N, D = 3000, 32
TOL = (
    0.02  # hnswlib file vs model: upper-level size is an expectation over random levels
)


@pytest.fixture(scope="module")
def x():
    return np.random.RandomState(0).randn(N, D).astype(np.float32)


@pytest.mark.parametrize("M", [8, 16])
def test_hnswlib_formula_matches_saved_file(x, M, tmp_path):
    idx = hnswlib.Index("l2", D)
    idx.init_index(N, M=M, ef_construction=100, random_seed=1)
    idx.add_items(x)
    path = str(tmp_path / "h.bin")
    idx.save_index(path)
    actual = os.path.getsize(path)
    model = mm.hnswlib_bytes(N, D, M)["total_bytes"]
    assert abs(model - actual) / actual < TOL, (model, actual)


def test_ivfpq_matches_faiss_serialization(x):
    nlist, m = 32, 8
    idx = faiss.IndexIVFPQ(faiss.IndexFlatL2(D), D, nlist, m, 8)
    idx.train(x)
    idx.add(x)
    actual = len(faiss.serialize_index(idx))
    model = mm.faiss_ivfpq_bytes(N, D, nlist, m)["total_bytes"]
    assert abs(model - actual) / actual < 0.05, (model, actual)


def test_ivfpqr_matches_faiss_serialization(x):
    nlist, m, mr = 32, 8, 8
    idx = faiss.IndexIVFPQR(faiss.IndexFlatL2(D), D, nlist, m, 8, mr, 8)
    idx.train(x)
    idx.add(x)
    actual = len(faiss.serialize_index(idx))
    model = mm.faiss_ivfpq_bytes(N, D, nlist, m, refine_m=mr)["total_bytes"]
    assert abs(model - actual) / actual < 0.05, (model, actual)


@pytest.mark.parametrize("M", [8, 16])
def test_hnswpq_matches_faiss_serialization(x, M):
    idx = faiss.IndexHNSWPQ(D, 8, M)
    idx.hnsw.efConstruction = 50
    idx.train(x)
    idx.add(x)
    actual = len(faiss.serialize_index(idx))
    model = mm.faiss_hnswpq_bytes(N, D, M, 8)["total_bytes"]
    assert abs(model - actual) / actual < 0.05, (model, actual)


def test_hnswsq8_matches_faiss_serialization(x):
    M = 8
    idx = faiss.IndexHNSWSQ(D, faiss.ScalarQuantizer.QT_8bit, M)
    idx.hnsw.efConstruction = 50
    idx.train(x)
    idx.add(x)
    actual = len(faiss.serialize_index(idx))
    model = mm.faiss_hnswsq8_bytes(N, D, M)["total_bytes"]
    assert abs(model - actual) / actual < 0.05, (model, actual)


def test_components_sum_to_total_and_per_vector():
    r = mm.zgq_compressed_bytes(1000, 128, M=8, pq_m=16, n_zones=50)
    assert sum(r["components"].values()) == r["total_bytes"]
    assert r["bytes_per_vector"] == pytest.approx(r["total_bytes"] / 1000)


def test_keeping_vectors_adds_exactly_raw_bytes():
    a = mm.zgq_compressed_bytes(1000, 128, n_zones=50)["total_bytes"]
    b = mm.zgq_compressed_bytes(1000, 128, n_zones=50, keep_vectors=True)["total_bytes"]
    assert b - a == 1000 * 128 * 4


def test_default_zgq_holds_two_vector_copies():
    d = mm.zgq_default_bytes(100000, 128, n_zones=300)
    c = mm.zgq_compressed_bytes(100000, 128, n_zones=300)
    assert d["total_bytes"] > 2 * 100000 * 128 * 4
    assert d["total_bytes"] > 3 * c["total_bytes"]
