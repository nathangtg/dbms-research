"""Dataset loaders for the build-order study.

One interface for ann-benchmarks HDF5 files (SIFT1M, GloVe-100) and the
synthetic generator used elsewhere in the harness. Downloads are cached outside
git in ``$ZGQ_DATA_DIR`` or ``~/.cache/zgq-datasets``.

Angular datasets are L2-normalized (train and test) and reported with metric
``"l2"``: on unit vectors squared L2 distance is ``2 - 2 cos``, so the ranking
equals the angular ranking and the existing L2 index and ground-truth code work
unchanged.
"""
from __future__ import annotations

import hashlib
import json
import os
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np

from benchmarks.run_benchmarks import exact_ground_truth, generate_test_data

DATASETS: Dict[str, dict] = {
    "sift-128-euclidean": {
        "url": "http://ann-benchmarks.com/sift-128-euclidean.hdf5",
        "bytes": 525128288,
        "train": (1000000, 128),
        "test": (10000, 128),
    },
    "glove-100-angular": {
        "url": "http://ann-benchmarks.com/glove-100-angular.hdf5",
        "bytes": 485413888,
        "train": (1183514, 100),
        "test": (10000, 100),
    },
}

SYNTHETIC_DIM = 128
SYNTHETIC_DEFAULT_QUERIES = 1000


@dataclass
class Dataset:
    name: str
    vectors: np.ndarray
    queries: np.ndarray
    gt: np.ndarray
    metric: str
    source: dict = field(default_factory=dict)


def cache_dir() -> Path:
    env = os.environ.get("ZGQ_DATA_DIR")
    if env:
        return Path(env)
    return Path.home() / ".cache" / "zgq-datasets"


def _download(url: str, dest: Path) -> None:
    req = urllib.request.Request(url, headers={"User-Agent": "zgq-datasets/1.0"})
    with urllib.request.urlopen(req) as r, open(dest, "wb") as f:  # noqa: S310
        while True:
            chunk = r.read(1 << 20)
            if not chunk:
                break
            f.write(chunk)


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            chunk = f.read(1 << 22)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def _sidecar(path: Path) -> Path:
    return path.parent / (path.name + ".json")


def ensure_downloaded(name: str) -> Path:
    spec = DATASETS[name]
    d = cache_dir()
    d.mkdir(parents=True, exist_ok=True)
    path = d / f"{name}.hdf5"
    if not (path.exists() and path.stat().st_size == spec["bytes"]):
        part = d / f"{name}.hdf5.part"
        print(f"[datasets] downloading {spec['url']}")
        _download(spec["url"], part)
        size = part.stat().st_size
        if size != spec["bytes"]:
            part.unlink()
            raise ValueError(f"{name}: downloaded {size} bytes, expected {spec['bytes']}")
        part.replace(path)
        _sidecar(path).unlink(missing_ok=True)
    side = _sidecar(path)
    if not side.exists():
        info = {"name": name, "url": spec["url"], "bytes": path.stat().st_size,
                "sha256": _sha256(path)}
        side.write_text(json.dumps(info, indent=2) + "\n")
    return path


def _as_f32(a: np.ndarray, normalize: bool) -> np.ndarray:
    out = np.empty(a.shape, dtype=np.float32)
    step = 100_000
    for s in range(0, len(a), step):
        c = np.asarray(a[s : s + step], dtype=np.float64)
        if normalize:
            n = np.linalg.norm(c, axis=1, keepdims=True)
            c = c / np.maximum(n, 1e-12)
        out[s : s + step] = c
    return np.ascontiguousarray(out)


def load_hdf5(
    path,
    name: str,
    n_base: Optional[int] = None,
    n_queries: Optional[int] = None,
    k: int = 10,
    seed: int = 0,
    expect_train: Optional[Tuple[int, int]] = None,
    expect_test: Optional[Tuple[int, int]] = None,
    source: Optional[dict] = None,
) -> Dataset:
    import h5py

    with h5py.File(path, "r") as f:
        train, test = f["train"], f["test"]
        if expect_train is not None and tuple(train.shape) != tuple(expect_train):
            raise ValueError(f"{name}: train shape {train.shape}, expected {expect_train}")
        if expect_test is not None and tuple(test.shape) != tuple(expect_test):
            raise ValueError(f"{name}: test shape {test.shape}, expected {expect_test}")
        if train.ndim != 2 or test.ndim != 2 or train.shape[1] != test.shape[1]:
            raise ValueError(f"{name}: inconsistent train/test shapes")
        distance = f.attrs.get("distance", "euclidean")
        if isinstance(distance, bytes):
            distance = distance.decode()
        angular = distance == "angular"
        n_total = train.shape[0]
        nq = test.shape[0] if n_queries is None else min(n_queries, test.shape[0])
        if n_base is not None and n_base > n_total:
            raise ValueError(f"{name}: n_base {n_base} exceeds {n_total}")
        subset = n_base is not None and n_base < n_total
        if subset:
            idx = np.sort(np.random.RandomState(seed).choice(n_total, n_base, replace=False))
            vectors = _as_f32(train[idx], angular)
        else:
            vectors = _as_f32(train[:], angular)
        queries = _as_f32(test[:nq], angular)
        file_gt = None if subset else np.asarray(f["neighbors"][:nq, :k], dtype=np.int64)
        if file_gt is not None and file_gt.shape[1] < k:
            raise ValueError(f"{name}: file has only {file_gt.shape[1]} neighbors, need {k}")

    gt = exact_ground_truth(vectors, queries, k) if subset else file_gt
    src = dict(source or {})
    src.update({"file_distance": str(distance), "n_total": int(n_total),
                "n_base": int(len(vectors)), "subset_seed": int(seed) if subset else None})
    return Dataset(name, vectors, queries, gt, "l2", src)


def verify_file_gt(
    vectors: np.ndarray,
    queries: np.ndarray,
    file_gt: np.ndarray,
    n_check: int = 200,
    tol: float = 1e-4,
) -> Tuple[bool, float]:
    """Check file-provided GT against brute force, tie tolerant.

    Compares L2 distances rank by rank (not ids). Returns (ok, worst relative
    distance gap). Vectors must already be normalized for angular datasets.
    """
    n = min(n_check, len(queries))
    k = file_gt.shape[1]
    q = queries[:n]
    exact = exact_ground_truth(vectors, q, k)

    def dist(ids):
        diff = vectors[ids].astype(np.float64) - q[:, None, :].astype(np.float64)
        return np.sqrt((diff * diff).sum(-1))

    de, df = dist(exact), dist(file_gt[:n])
    worst = float(np.max(np.abs(de - df) / np.maximum(de, 1.0)))
    return worst <= tol, worst


def load_dataset(
    name: str,
    n_base: Optional[int] = None,
    n_queries: Optional[int] = None,
    k: int = 10,
    seed: int = 0,
) -> Dataset:
    if name.startswith("synthetic-"):
        n = int(name.split("-", 1)[1])
        nq = SYNTHETIC_DEFAULT_QUERIES if n_queries is None else n_queries
        vectors, queries = generate_test_data(n, nq, SYNTHETIC_DIM, seed)
        vectors = np.ascontiguousarray(vectors, dtype=np.float32)
        queries = np.ascontiguousarray(queries, dtype=np.float32)
        gt = exact_ground_truth(vectors, queries, k)
        return Dataset(name, vectors, queries, gt, "l2",
                       {"generator": "generate_test_data", "data_seed": seed, "n_base": n})
    if name not in DATASETS:
        raise KeyError(f"unknown dataset {name!r}")
    spec = DATASETS[name]
    path = ensure_downloaded(name)
    side = json.loads(_sidecar(path).read_text())
    source = {"url": spec["url"], "bytes": side["bytes"], "sha256": side["sha256"]}
    return load_hdf5(path, name, n_base, n_queries, k, seed,
                     expect_train=spec["train"], expect_test=spec["test"], source=source)
