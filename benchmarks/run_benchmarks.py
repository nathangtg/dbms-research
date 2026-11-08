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
    * ``--improvements`` adds opt-in ZGQ variants (off by default, so the
      canonical specs and results are unchanged)

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
from zgq import memory_model as mm

SCALES = {"tiny": 2_000, "10k": 10_000, "100k": 100_000, "1m": 1_000_000}
DIMENSION = 128
DEFAULT_EF = (16, 32, 64, 128, 256)
DEFAULT_NPROBE = (1, 4, 16, 64, 256)
HNSW_M = 16
ZGQ_MS = (16, 24, 32)
EF_CONSTRUCTION = 200
IMPROVE_NPROBE = (1, 4, 16, 64)
IMPROVE_RATIO_PCT = (105, 115, 130, 150, 200)
IMPROVE_GLOBAL_PROBE = 8
IMPROVE_MAX_PROBE = 64


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


def run_index(spec, vectors, queries, gt, args, scale, raw_dir, built=None) -> Dict:
    """Measure one index. ``built`` = (idx, build_t, mem, search, analytic) lets
    several variants share one graph build (used by the improvement variants)."""
    label, algo, builder, build_params, param_name, params = spec
    raw_dir = Path(raw_dir)
    if built is None:
        print(f"  [{scale}] {label}: building ...", flush=True)
        idx, build_t, mem, search = builder(vectors)
        analytic = None
        if algo == "hnsw":
            analytic = _bytes_entry(
                mm.hnswlib_bytes(len(vectors), vectors.shape[1], build_params["M"]),
                "zgq.memory_model.hnswlib_bytes",
            )
    else:
        idx, build_t, mem, search, analytic = built
    print(f"  [{scale}] {label}: build {build_t:.2f}s, rss +{mem:.1f} MB", flush=True)

    points, raw_runs = [], []
    for p in params:
        pt, runs = measure_point(
            search, queries, gt, args.k, p, args.repeats, args.warmup
        )
        sp = {param_name: int(p)}
        extra = getattr(search, "point_info", None)
        points.append({"search_param": sp, **pt, **(extra() if extra else {})})
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
    entry = {
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
    if analytic is not None:
        entry["analytic_bytes"] = analytic
    return entry


def _bytes_entry(model: Dict, source: str) -> Dict:
    return {
        "bytes_per_vector": model["bytes_per_vector"],
        "total_bytes": model["total_bytes"],
        "source": source,
    }


# Opt-in recall-improvement variants (--improvements). Each group shares one
# graph build; variants differ only in search settings and zone replicas.

def _zgq_search_fn(idx, v, search_mode, adaptive, nopq, replicas, sweep):
    """Return a search closure for one variant. ``sweep`` is 'ef', 'nprobe'
    or 'ratio'. The closure configures the shared index before each call."""
    cfg = idx.config
    info = {}

    def search(queries, k, p):
        cfg.search_mode = search_mode
        cfg.adaptive_probe = adaptive
        cfg.min_probe, cfg.max_probe = 1, IMPROVE_MAX_PROBE
        ef = None
        n_probe = IMPROVE_GLOBAL_PROBE
        if sweep == "ef":
            ef = int(p)
        elif sweep == "nprobe":
            n_probe = int(p)
        else:
            cfg.probe_ratio = p / 100.0
            n_probe = IMPROVE_MAX_PROBE
        ids = idx.batch_zone_search(
            queries, k, n_probe=n_probe, ef_search=ef, use_pq_rerank=not nopq
        )[0]
        info["mean_probes"] = idx.last_batch_stats["mean_probes"]
        return ids

    search.point_info = lambda: {"mean_probes": info.get("mean_probes")}
    return search


def improvement_groups(n, args):
    """Groups of (build_fn, [variant, ...]). A variant is
    (label, params_description, sweep_name, sweep_values, search_kwargs, replicas)."""
    ef = tuple(args.ef)
    nprobe = tuple(args.improve_nprobe)
    ratios = tuple(args.improve_ratio_pct)

    def zone_variants(pq_tag):
        nopq = pq_tag == "-nopq"
        return [
            (f"ZGQ-M16+global-entry{pq_tag}", "ef", ef,
             dict(search_mode="global_entry", adaptive=False, nopq=nopq, sweep="ef"), 1),
            (f"ZGQ-M16+zonescan{pq_tag}", "nprobe", nprobe,
             dict(search_mode="zone_scan", adaptive=False, nopq=nopq, sweep="nprobe"), 1),
            (f"ZGQ-M16+overlap2{pq_tag}", "nprobe", nprobe,
             dict(search_mode="zone_scan", adaptive=False, nopq=nopq, sweep="nprobe"), 2),
            (f"ZGQ-M16+adaptive{pq_tag}", "ratio_pct", ratios,
             dict(search_mode="zone_scan", adaptive=True, nopq=nopq, sweep="ratio"), 1),
        ]

    zone = zone_variants("-nopq") + zone_variants("")
    rand = [
        (lab.replace("ZGQ-M16", "ZGQ-M16+random"), pn, vals, kw, rep)
        for lab, pn, vals, kw, rep in zone
        if "global-entry" in lab
    ]
    return [
        {"key": "zone-graph", "insertion_order": "zone", "variants": zone},
        {"key": "random-graph", "insertion_order": "random", "variants": rand},
    ]


def _improvement_worker(scale: str, group_key: str, argv_dict: Dict) -> List[Dict]:
    args = argparse.Namespace(**argv_dict)
    n = SCALES[scale]
    vectors, queries = generate_test_data(n, args.n_queries, DIMENSION, args.seed)
    gt = exact_ground_truth(vectors, queries, args.k)
    group = next(g for g in improvement_groups(n, args) if g["key"] == group_key)
    order = group["insertion_order"]
    out_dir = Path(args.out_dir) / "raw"

    print(f"  [{scale}] group {group_key}: building ...", flush=True)

    def build():
        np.random.seed(args.seed)
        cfg = ZGQConfig(
            M=HNSW_M, ef_construction=EF_CONSTRUCTION, random_state=args.seed,
            insertion_order=order,
        )
        with _graph_threads(args.threads):
            return ZGQIndex(cfg).build(vectors)

    idx, build_t, mem = _measured_build(build)
    print(f"  [{scale}] group {group_key}: build {build_t:.2f}s", flush=True)
    base_model = mm.zgq_default_bytes(
        n, vectors.shape[1], HNSW_M, idx.pq.config.m, 8, len(idx.zones.fine_centroids)
    )

    results = []
    if group_key == "random-graph":
        # random insertion order on the canonical fast path (batch_search)
        fast = lambda q, k, e: idx.batch_search(q, k, ef_search=e, fast_mode=True)[0]
        results.append(run_index(
            ("ZGQ-M16+random", "zgq", None,
             {"M": HNSW_M, "ef_construction": EF_CONSTRUCTION, "insertion_order": order,
              "n_zones": "auto", "use_pq": True,
              "search_mode": "batch_search fast_mode rerank_factor=3"},
             "ef", args.ef),
            vectors, queries, gt, args, scale, out_dir,
            built=(idx, build_t, mem, fast,
                   _bytes_entry(base_model, "zgq.memory_model.zgq_default_bytes")),
        ))
        results[-1]["comparison"] = "variant"

    for label, param_name, values, kw, replicas in group["variants"]:
        extra_t = 0.0
        if replicas > 1 or idx.zones.n_replicas > 1:
            t0 = time.perf_counter()
            idx.zones.build_replicas(vectors, replicas)
            extra_t = time.perf_counter() - t0 if replicas > 1 else 0.0
        search = _zgq_search_fn(idx, vectors, replicas=replicas, **kw)
        model = dict(base_model)
        extra_bytes = n * (replicas - 1) * 4
        model["total_bytes"] = base_model["total_bytes"] + extra_bytes
        model["bytes_per_vector"] = model["total_bytes"] / n
        params = {
            "M": HNSW_M, "ef_construction": EF_CONSTRUCTION,
            "insertion_order": order, "zone_replicas": replicas,
            "search_mode": kw["search_mode"], "adaptive_probe": kw["adaptive"],
            "use_pq_rerank": not kw["nopq"],
            "shared_build": group_key,
        }
        if kw["sweep"] == "ef":
            params["n_probe"] = IMPROVE_GLOBAL_PROBE
        if kw["adaptive"]:
            params["max_probe"] = IMPROVE_MAX_PROBE
        spec = (label, "zgq", None, params, param_name, values)
        # build time = shared graph build + this variant's replica build
        res = run_index(
            spec, vectors, queries, gt, args, scale, out_dir,
            built=(idx, build_t + extra_t, mem, search,
                   _bytes_entry(model, "zgq.memory_model.zgq_default_bytes + n*(r-1)*4 replica ids")),
        )
        res["build"]["shared_graph_time_s"] = build_t
        res["build"]["replica_time_s"] = extra_t
        res["comparison"] = "variant"
        results.append(res)
    return results


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
        if args.improvements:
            for group in improvement_groups(n, args):
                with ProcessPoolExecutor(max_workers=1, mp_context=ctx) as pool:
                    indexes.extend(pool.submit(
                        _improvement_worker, scale, group["key"], vars(args)).result())
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
        "--improvements",
        action="store_true",
        help="also run the opt-in ZGQ recall-improvement variants (overlap, "
        "adaptive probing, global entry, random insertion); default specs unchanged",
    )
    p.add_argument("--improve-nprobe", type=int, nargs="+", default=list(IMPROVE_NPROBE))
    p.add_argument(
        "--improve-ratio-pct", type=int, nargs="+", default=list(IMPROVE_RATIO_PCT),
        help="adaptive probe_ratio sweep in percent (130 = 1.3)",
    )
    p.add_argument(
        "--no-ivfpq", action="store_true", help="skip the FAISS IVF-PQ baseline"
    )
    return p


def main(argv=None) -> int:
    run(build_parser().parse_args(argv))
    return 0


if __name__ == "__main__":
    sys.exit(main())
