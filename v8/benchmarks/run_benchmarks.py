"""
ZGQ v8 reproducible benchmark harness
=====================================

Compares ZGQ (several M values), hnswlib HNSW (M=16) and FAISS IVF-PQ on
seeded synthetic clustered-Gaussian data.

Protocol (recorded in summary.json under ``protocol``):
    * every random source is seeded (data, queries, index builds)
    * HNSW M=16 vs ZGQ M=16 is the matched comparison; ZGQ M=24 and M=32 are
      labeled variants with a larger graph, not a like-for-like comparison
    * one warm-up pass over all queries before timing each search setting
    * each setting is timed ``--repeats`` times (default 5) over the whole
      query batch; median and IQR across repeats are reported
    * build time and index memory (RSS delta around build) are recorded once
      per index, not per search setting
    * single thread by default (``--threads 1``) for build and search

Usage:
    python -m benchmarks.run_benchmarks --scale 10k 100k --out-dir results/2025-01-01
    python -m benchmarks.run_benchmarks --scale tiny --n-queries 50 --repeats 2 --out-dir /tmp/smoke
"""

import argparse
import datetime
import gc
import json
import multiprocessing
import os
import platform
import subprocess
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace
from typing import Callable, Dict, List, Optional, Tuple

import hnswlib
import numpy as np
from zgq import ZGQConfig, ZGQIndex
from zgq.search import compute_recall

try:
    import faiss
except ImportError:  # optional baseline
    faiss = None

try:
    import psutil
except ImportError:  # pragma: no cover
    psutil = None

from benchmarks.check_results import SCHEMA_VERSION, validate_summary

SCALES = {"tiny": 2_000, "10k": 10_000, "100k": 100_000, "1m": 1_000_000}
DIMENSION = 128
DEFAULT_EF = (16, 32, 64, 128, 256)
DEFAULT_NPROBE = (1, 4, 16, 64, 256)
HNSW_M = 16
ZGQ_MS = (16, 24, 32)
EF_CONSTRUCTION = 200


# ----------------------------------------------------------------------------
# Data
# ----------------------------------------------------------------------------


def generate_test_data(
    n_vectors: int,
    n_queries: int,
    dimension: int = DIMENSION,
    seed: int = 42,
) -> Tuple[np.ndarray, np.ndarray]:
    """Seeded clustered Gaussian data. Queries: 70% near clusters, 30% random.

    Vectors and queries come from separate streams so changing ``n_queries``
    never changes the database.
    """
    rng = np.random.RandomState(seed)
    n_clusters = max(10, n_vectors // 1000)
    centers = rng.randn(n_clusters, dimension).astype(np.float32)
    assign = rng.randint(0, n_clusters, n_vectors)

    vectors = np.empty((n_vectors, dimension), dtype=np.float32)
    chunk = 50_000
    for s in range(0, n_vectors, chunk):
        e = min(s + chunk, n_vectors)
        noise = rng.randn(e - s, dimension).astype(np.float32) * 0.3
        vectors[s:e] = centers[assign[s:e]] + noise

    qrng = np.random.RandomState(seed + 1)
    from_cluster = qrng.random_sample(n_queries) < 0.7
    q_assign = qrng.randint(0, n_clusters, n_queries)
    q_noise = qrng.randn(n_queries, dimension).astype(np.float32) * 0.3
    q_random = qrng.randn(n_queries, dimension).astype(np.float32)
    queries = np.where(from_cluster[:, None], centers[q_assign] + q_noise, q_random)
    return vectors, np.ascontiguousarray(queries, dtype=np.float32)


def exact_ground_truth(vectors: np.ndarray, queries: np.ndarray, k: int) -> np.ndarray:
    """Exact top-k L2 neighbors by chunked brute force (numpy only)."""
    vnorm = np.einsum("ij,ij->i", vectors, vectors)
    out = np.empty((len(queries), k), dtype=np.int64)
    step = max(1, int(2e7 // len(vectors)))  # bound the distance matrix size
    for s in range(0, len(queries), step):
        q = queries[s : s + step]
        d2 = vnorm[None, :] - 2.0 * (q @ vectors.T)
        part = np.argpartition(d2, k - 1, axis=1)[:, :k]
        order = np.argsort(np.take_along_axis(d2, part, axis=1), axis=1)
        out[s : s + step] = np.take_along_axis(part, order, axis=1)
    return out


# ----------------------------------------------------------------------------
# Environment
# ----------------------------------------------------------------------------


def _cpu_name() -> str:
    try:
        if sys.platform == "darwin":
            return subprocess.check_output(
                ["sysctl", "-n", "machdep.cpu.brand_string"], text=True
            ).strip()
        with open("/proc/cpuinfo") as f:
            for line in f:
                if line.startswith("model name"):
                    return line.split(":", 1)[1].strip()
    except Exception:
        pass
    return platform.processor() or "unknown"


def _git(args: List[str]) -> Optional[str]:
    try:
        return subprocess.check_output(
            ["git"] + args,
            cwd=Path(__file__).parent,
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except Exception:
        return None


def collect_environment() -> Dict:
    import numba
    import scipy
    import sklearn
    import zgq

    libs = {
        "numpy": np.__version__,
        "scipy": scipy.__version__,
        "scikit-learn": sklearn.__version__,
        "numba": numba.__version__,
        "hnswlib": getattr(hnswlib, "__version__", "unknown"),
        "faiss": getattr(faiss, "__version__", None) if faiss else None,
        "psutil": psutil.__version__ if psutil else None,
        "zgq": zgq.__version__,
    }
    if libs["hnswlib"] == "unknown":
        try:
            from importlib.metadata import version

            libs["hnswlib"] = version("hnswlib")
        except Exception:
            pass
    status = _git(["status", "--porcelain", "--untracked-files=no"])
    return {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "machine": platform.machine(),
        "cpu": _cpu_name(),
        "cpu_count": os.cpu_count(),
        "ram_gb": round(psutil.virtual_memory().total / 2**30, 1) if psutil else None,
        "libraries": libs,
        "git_commit": _git(["rev-parse", "HEAD"]),
        "git_dirty": bool(status) if status is not None else None,
    }


def rss_mb() -> float:
    if psutil is None:
        return float("nan")
    gc.collect()
    return psutil.Process().memory_info().rss / 2**20


# ----------------------------------------------------------------------------
# Index adapters: each returns (index_obj, build_time_s, rss_delta_mb, search_fn)
# ----------------------------------------------------------------------------


@contextmanager
def _graph_threads(n_threads: int):
    """Make the hnswlib index created inside ZGQ use ``n_threads`` threads.

    ZGQ builds its own hnswlib index and does not expose a thread setting.
    Wrapping Index keeps the build deterministic at 1 thread.
    """
    import zgq.core.graph as graph_mod

    class _Index(hnswlib.Index):
        def init_index(self, *a, **kw):
            super().init_index(*a, **kw)
            self.set_num_threads(n_threads)

    original = graph_mod.hnswlib
    graph_mod.hnswlib = SimpleNamespace(Index=_Index)
    try:
        yield
    finally:
        graph_mod.hnswlib = original


def _measured_build(build: Callable[[], object]) -> Tuple[object, float, float]:
    before = rss_mb()
    t0 = time.perf_counter()
    obj = build()
    elapsed = time.perf_counter() - t0
    after = rss_mb()
    return obj, elapsed, after - before


def build_hnsw(vectors, M, seed, threads):
    def build():
        idx = hnswlib.Index(space="l2", dim=vectors.shape[1])
        idx.init_index(
            max_elements=len(vectors),
            ef_construction=EF_CONSTRUCTION,
            M=M,
            random_seed=seed,
        )
        idx.set_num_threads(threads)
        idx.add_items(vectors, np.arange(len(vectors)))
        return idx

    idx, t, mem = _measured_build(build)

    def search(queries, k, ef):
        idx.set_ef(ef)
        return idx.knn_query(queries, k=k)[0]

    return idx, t, mem, search


def build_zgq(vectors, M, seed, threads):
    def build():
        np.random.seed(seed)
        cfg = ZGQConfig(M=M, ef_construction=EF_CONSTRUCTION, random_state=seed)
        with _graph_threads(threads):
            return ZGQIndex(cfg).build(vectors)

    idx, t, mem = _measured_build(build)

    def search(queries, k, ef):
        return idx.batch_search(queries, k, ef_search=ef, fast_mode=True)[0]

    return idx, t, mem, search


def ivfpq_params(n: int) -> Dict:
    return {"nlist": int(2 * np.sqrt(n)), "pq_m": 32, "nbits": 8}


def build_ivfpq(vectors, seed, threads):
    p = ivfpq_params(len(vectors))
    d = vectors.shape[1]
    faiss.omp_set_num_threads(threads)

    def build():
        np.random.seed(seed)
        quantizer = faiss.IndexFlatL2(d)
        idx = faiss.IndexIVFPQ(quantizer, d, p["nlist"], p["pq_m"], p["nbits"])
        idx.cp.seed = seed
        idx.pq.cp.seed = seed
        idx.train(vectors)
        idx.add(vectors)
        return idx

    idx, t, mem = _measured_build(build)

    def search(queries, k, nprobe):
        idx.nprobe = nprobe
        return idx.search(queries, k)[1]

    return idx, t, mem, search


# ----------------------------------------------------------------------------
# Measurement
# ----------------------------------------------------------------------------


def stats(values: List[float]) -> Dict:
    a = np.asarray(values, dtype=np.float64)
    q1, med, q3 = np.percentile(a, [25, 50, 75])
    return {
        "median": float(med),
        "q1": float(q1),
        "q3": float(q3),
        "iqr": float(q3 - q1),
        "n": int(len(a)),
    }


def measure_point(
    search, queries, gt, k, param, repeats, warmup
) -> Tuple[Dict, List[Dict]]:
    for _ in range(warmup):
        search(queries, k, param)

    runs, recalls, lats, qpss = [], [], [], []
    for r in range(repeats):
        t0 = time.perf_counter()
        ids = search(queries, k, param)
        elapsed = time.perf_counter() - t0
        recall = float(compute_recall(np.asarray(ids), gt, k))
        lat = elapsed / len(queries) * 1000.0
        qps = len(queries) / elapsed
        recalls.append(recall)
        lats.append(lat)
        qpss.append(qps)
        runs.append(
            {
                "repeat": r,
                "elapsed_s": elapsed,
                "recall_pct": recall,
                "latency_ms_per_query": lat,
                "qps": qps,
            }
        )
    summary = {
        "recall_pct": stats(recalls),
        "latency_ms_per_query": stats(lats),
        "qps": stats(qpss),
    }
    return summary, runs


def run_index(spec, vectors, queries, gt, args, scale, raw_dir) -> Dict:
    label, algo, builder, build_params, param_name, params = spec
    raw_dir = Path(raw_dir)
    print(f"  [{scale}] {label}: building ...", flush=True)
    idx, build_t, mem, search = builder(vectors)
    print(f"  [{scale}] {label}: build {build_t:.2f}s, rss +{mem:.1f} MB", flush=True)

    points, raw_runs = [], []
    for p in params:
        pt, runs = measure_point(
            search, queries, gt, args.k, p, args.repeats, args.warmup
        )
        sp = {param_name: int(p)}
        points.append({"search_param": sp, **pt})
        for r in runs:
            raw_runs.append({"search_param": sp, **r})
        print(
            f"    {param_name}={p:<4} recall={pt['recall_pct']['median']:.1f}% "
            f"lat={pt['latency_ms_per_query']['median']:.4f}ms "
            f"(IQR {pt['latency_ms_per_query']['iqr']:.4f})",
            flush=True,
        )

    raw_name = f"raw/{scale}_{label}.json"
    raw_dir.mkdir(parents=True, exist_ok=True)
    with open(raw_dir.parent / raw_name, "w") as f:
        json.dump(
            {
                "label": label,
                "scale": scale,
                "build": {"time_s": build_t, "rss_delta_mb": mem},
                "runs": raw_runs,
            },
            f,
            indent=1,
        )

    del idx, search
    gc.collect()
    return {
        "label": label,
        "algo": algo,
        "comparison": "baseline"
        if algo != "zgq"
        else ("matched" if build_params["M"] == HNSW_M else "variant"),
        "build_params": build_params,
        "build": {"time_s": build_t, "rss_delta_mb": mem},
        "points": points,
        "raw_file": raw_name,
    }


def index_specs(n, args):
    ef = args.ef
    specs = [
        (
            f"HNSW-M{HNSW_M}",
            "hnsw",
            lambda v: build_hnsw(v, HNSW_M, args.seed, args.threads),
            {"M": HNSW_M, "ef_construction": EF_CONSTRUCTION},
            "ef",
            ef,
        )
    ]
    for M in args.zgq_m:
        specs.append(
            (
                f"ZGQ-M{M}",
                "zgq",
                (lambda v, M=M: build_zgq(v, M, args.seed, args.threads)),
                {
                    "M": M,
                    "ef_construction": EF_CONSTRUCTION,
                    "n_zones": "auto",
                    "use_pq": True,
                    "search_mode": "batch_search fast_mode rerank_factor=3",
                },
                "ef",
                ef,
            )
        )
    if faiss is not None and not args.no_ivfpq:
        specs.append(
            (
                "IVFPQ",
                "ivfpq",
                lambda v: build_ivfpq(v, args.seed, args.threads),
                ivfpq_params(n),
                "nprobe",
                args.nprobe,
            )
        )
    return specs


def _index_worker(scale: str, label: str, argv_dict: Dict) -> Dict:
    """Build and measure one index in a fresh process.

    A fresh process per index keeps the RSS delta meaningful: in a long-lived
    process the allocator reuses memory freed by earlier indexes, which makes
    later builds look free.
    """
    args = argparse.Namespace(**argv_dict)
    n = SCALES[scale]
    vectors, queries = generate_test_data(n, args.n_queries, DIMENSION, args.seed)
    gt = exact_ground_truth(vectors, queries, args.k)
    if args.threads > 0 and faiss is not None:
        faiss.omp_set_num_threads(args.threads)
    spec = next(s for s in index_specs(n, args) if s[0] == label)
    return run_index(spec, vectors, queries, gt, args, scale, Path(args.out_dir) / "raw")


def run(args) -> Dict:
    out_dir = Path(args.out_dir)
    raw_dir = out_dir / "raw"
    out_dir.mkdir(parents=True, exist_ok=True)

    summary = {
        "schema_version": SCHEMA_VERSION,
        "created_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(
            timespec="seconds"
        ),
        "environment": collect_environment(),
        "protocol": {
            "seed": args.seed,
            "k": args.k,
            "n_queries": args.n_queries,
            "repeats": args.repeats,
            "warmup_passes": args.warmup,
            "threads": args.threads,
            "latency_metric": "wall time of one batch call / n_queries, single process",
            "memory_metric": "RSS delta (MB) across index build in a fresh process, after gc.collect()",
            "data": "clustered Gaussian, dim 128, 70% cluster queries / 30% uniform",
            "ground_truth": "exact brute force L2",
        },
        "scales": {},
    }
    ctx = multiprocessing.get_context("spawn")
    for scale in args.scale:
        n = SCALES[scale]
        print(f"\n=== scale {scale}: {n:,} vectors, {args.n_queries} queries ===", flush=True)
        indexes = []
        for spec in index_specs(n, args):
            with ProcessPoolExecutor(max_workers=1, mp_context=ctx) as pool:
                indexes.append(pool.submit(
                    _index_worker, scale, spec[0], vars(args)).result())
        summary["scales"][scale] = {
            "n_vectors": n, "dimension": DIMENSION, "n_queries": args.n_queries,
            "indexes": indexes,
        }
        with open(out_dir / "summary.json", "w") as f:  # incremental save
            json.dump(summary, f, indent=2)

    errors = validate_summary(summary, base_dir=out_dir)
    if errors:
        raise RuntimeError("summary failed schema validation: " + "; ".join(errors))
    print(f"\nWrote {out_dir / 'summary.json'}")
    return summary


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Run the ZGQ v8 benchmark harness")
    p.add_argument(
        "--scale",
        nargs="+",
        default=["10k"],
        choices=list(SCALES),
        help="dataset scale(s): tiny (2K, smoke only), 10k, 100k, 1m",
    )
    p.add_argument("--out-dir", default="results", help="output directory")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--k", type=int, default=10)
    p.add_argument("--n-queries", type=int, default=1000)
    p.add_argument("--repeats", type=int, default=5)
    p.add_argument("--warmup", type=int, default=1, help="warm-up passes per setting")
    p.add_argument("--threads", type=int, default=1)
    p.add_argument("--ef", type=int, nargs="+", default=list(DEFAULT_EF))
    p.add_argument("--nprobe", type=int, nargs="+", default=list(DEFAULT_NPROBE))
    p.add_argument("--zgq-m", type=int, nargs="+", default=list(ZGQ_MS))
    p.add_argument(
        "--no-ivfpq", action="store_true", help="skip the FAISS IVF-PQ baseline"
    )
    return p


def main(argv=None) -> int:
    run(build_parser().parse_args(argv))
    return 0


if __name__ == "__main__":
    sys.exit(main())
