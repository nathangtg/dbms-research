import argparse
import json
import time
from pathlib import Path
from typing import Dict, List, Sequence

import hnswlib
import numpy as np
from zgq import ZGQConfig, ZGQIndex
from zgq.core.graph import GraphConfig, ZoneGuidedGraph
from zgq.index import _batch_rerank_fast
from zgq.search import compute_recall

from benchmarks.run_benchmarks import (
    DIMENSION,
    EF_CONSTRUCTION,
    HNSW_M,
    SCALES,
    _graph_threads,
    exact_ground_truth,
    generate_test_data,
)

K = 10
SEED = 42
N_PROBES = (1, 2, 4, 8, 16, 32, 64)


def fraction_outside_zones(
    gt: np.ndarray, assignments: np.ndarray, probed: Sequence[np.ndarray]
) -> float:
    outside = []
    for row, zones in zip(gt, probed):
        in_probed = np.isin(assignments[row], zones)
        outside.append(1.0 - float(in_probed.mean()))
    return float(np.mean(outside))


def oracle_zone_coverage(gt: np.ndarray, assignments: np.ndarray) -> float:
    cov = []
    for row in gt:
        z = assignments[row[0]]
        cov.append(float((assignments[row] == z).mean()))
    return float(np.mean(cov))


def nearest_centroid_zones(
    queries: np.ndarray, centroids: np.ndarray, n_probe: int
) -> np.ndarray:
    d2 = (
        np.einsum("ij,ij->i", queries, queries)[:, None]
        - 2.0 * queries @ centroids.T
        + np.einsum("ij,ij->i", centroids, centroids)[None, :]
    )
    n_probe = min(n_probe, centroids.shape[0])
    part = np.argpartition(d2, n_probe - 1, axis=1)[:, :n_probe]
    order = np.argsort(np.take_along_axis(d2, part, axis=1), axis=1)
    return np.take_along_axis(part, order, axis=1)


def load(scale: str, n_queries: int):
    n = SCALES[scale]
    vectors, queries = generate_test_data(n, n_queries, DIMENSION, SEED)
    gt = exact_ground_truth(vectors, queries, K)
    return vectors, queries, gt


def build_zgq_index(vectors: np.ndarray, n_zones="auto") -> ZGQIndex:
    np.random.seed(SEED)
    cfg = ZGQConfig(
        M=HNSW_M, ef_construction=EF_CONSTRUCTION, random_state=SEED, n_zones=n_zones
    )
    with _graph_threads(1):
        return ZGQIndex(cfg).build(vectors)


def build_plain_hnsw(vectors: np.ndarray, seed: int = SEED) -> hnswlib.Index:
    idx = hnswlib.Index(space="l2", dim=vectors.shape[1])
    idx.init_index(
        max_elements=len(vectors),
        ef_construction=EF_CONSTRUCTION,
        M=HNSW_M,
        random_seed=seed,
    )
    idx.set_num_threads(1)
    idx.add_items(vectors, np.arange(len(vectors)))
    return idx


def build_graph(vectors, zones, order: str, seed: int) -> ZoneGuidedGraph:
    cfg = GraphConfig(
        M=HNSW_M,
        ef_construction=EF_CONSTRUCTION,
        random_seed=seed,
        insertion_order=order,
    )
    with _graph_threads(1):
        return ZoneGuidedGraph(cfg).build(
            vectors, zones.assignments, zones.fine_centroids
        )


def _timed(fn, repeats: int = 3):
    fn()  # warm-up
    times, out = [], None
    for _ in range(repeats):
        t0 = time.perf_counter()
        out = fn()
        times.append(time.perf_counter() - t0)
    return out, float(np.median(times))


def _point(ids, elapsed, gt, n_q) -> Dict:
    return {
        "recall": float(compute_recall(np.asarray(ids), gt, K)),
        "latency_ms": elapsed / n_q * 1000.0,
    }


def _hnsw_points(idx_search, queries, vectors, gt, ef, rerank_factor):
    n_q = len(queries)

    def run():
        k_s = K * rerank_factor
        ids = idx_search(queries, k_s, max(ef, k_s))
        if rerank_factor > 1:
            ids = _batch_rerank_fast(queries, vectors, ids, K)[0]
        return ids

    ids, t = _timed(run)
    return _point(ids, t, gt, n_q)


def d0_path_audit(vectors, queries, gt, idx: ZGQIndex, efs=(16, 64, 256)) -> Dict:
    calls = {"select_zones": 0}
    original = idx.zones.select_zones

    def counting(*a, **kw):
        calls["select_zones"] += 1
        return original(*a, **kw)

    idx.zones.select_zones = counting
    try:
        idx.batch_search(queries[:50], K, ef_search=64, fast_mode=True)
        fast_calls = calls["select_zones"]
        idx.batch_search(queries[:5], K, ef_search=64, fast_mode=False)
        slow_calls = calls["select_zones"] - fast_calls
    finally:
        idx.zones.select_zones = original

    plain = build_plain_hnsw(vectors)

    def plain_search(q, k, ef):
        plain.set_ef(ef)
        return plain.knn_query(q, k=k)[0]

    def graph_search(q, k, ef):
        return idx.graph.search_batch(q, k, ef)[0]

    rows = []
    for ef in efs:
        row = {"ef": ef}
        for rf in (1, 3):
            row[f"plain_hnsw_rf{rf}"] = _hnsw_points(
                plain_search, queries, vectors, gt, ef, rf
            )
            row[f"zgq_graph_rf{rf}"] = _hnsw_points(
                graph_search, queries, vectors, gt, ef, rf
            )
        ids, t = _timed(
            lambda: idx.batch_search(queries, K, ef_search=ef, fast_mode=True)[0]
        )
        row["zgq_batch_search_fast"] = _point(ids, t, gt, len(queries))
        rows.append(row)
    return {
        "select_zones_calls_in_fast_mode_50_queries": fast_calls,
        "select_zones_calls_in_zone_mode_5_queries": slow_calls,
        "rows": rows,
    }


def d1_insertion_order(
    vectors, queries, gt, idx: ZGQIndex, seeds=(42, 43), efs=(64, 256)
) -> Dict:
    variants = {}
    for seed in seeds:
        variants[f"plain_hnsw_seed{seed}"] = (
            "plain",
            build_plain_hnsw(vectors, seed),
            None,
        )
        for order in ("zone", "random"):
            t0 = time.perf_counter()
            g = build_graph(vectors, idx.zones, order, seed)
            build_s = time.perf_counter() - t0
            variants[f"zgq_graph_{order}_seed{seed}"] = ("graph", g, build_s)

    rows = []
    for name, (kind, obj, build_s) in variants.items():
        if kind == "plain":

            def search(q, k, ef, obj=obj):
                obj.set_ef(ef)
                return obj.knn_query(q, k=k)[0]

        else:

            def search(q, k, ef, obj=obj):
                return obj.search_batch(q, k, ef)[0]

        for ef in efs:
            for rf in (1, 3):
                p = _hnsw_points(search, queries, vectors, gt, ef, rf)
                rows.append(
                    {
                        "variant": name,
                        "ef": ef,
                        "rerank_factor": rf,
                        "build_s": build_s,
                        **p,
                    }
                )
    return {"seeds": list(seeds), "rows": rows}


def d2_nprobe(queries, gt, idx: ZGQIndex, n_probes=N_PROBES) -> Dict:
    rows = []
    for npb in n_probes:
        row = {"n_probe": npb}
        for label, pq in (("with_pq_rerank", True), ("without_pq_rerank", False)):
            t0 = time.perf_counter()
            ids = np.stack(
                [idx.search(q, K, n_probe=npb, use_pq_rerank=pq)[0][:K] for q in queries]
            )
            row[label] = _point(ids, time.perf_counter() - t0, gt, len(queries))
        rows.append(row)
    return {"ef_search": idx.config.ef_search, "rows": rows}


def d3_oracle_zone(queries, gt, idx: ZGQIndex, n_probes=N_PROBES) -> Dict:
    a = idx.zones.assignments
    cents = idx.zones.fine_centroids
    sizes = np.bincount(a, minlength=len(cents))
    rows = []
    for npb in n_probes:
        sel = [idx.zones.select_zones(q, npb) for q in queries]
        flat = nearest_centroid_zones(queries, cents, npb)
        rows.append(
            {
                "n_probe": npb,
                "frac_true_neighbors_outside_select_zones": fraction_outside_zones(
                    gt, a, sel
                ),
                "frac_true_neighbors_outside_flat_nearest": fraction_outside_zones(
                    gt, a, list(flat)
                ),
                "frac_queries_nn1_zone_probed": float(
                    np.mean([a[g[0]] in s for g, s in zip(gt, sel)])
                ),
            }
        )
    return {
        "n_zones": int(len(cents)),
        "zone_size": {
            "min": int(sizes.min()),
            "median": float(np.median(sizes)),
            "max": int(sizes.max()),
        },
        "oracle_zone_recall": oracle_zone_coverage(gt, a),
        "rows": rows,
    }


def d4_zone_sweep(
    vectors, queries, gt, zone_counts: List, n_probes=(1, 4, 16, 64)
) -> Dict:
    out = []
    for nz in zone_counts:
        t0 = time.perf_counter()
        idx = build_zgq_index(vectors, nz)
        build_s = time.perf_counter() - t0
        fast_ids, t = _timed(
            lambda: idx.batch_search(queries, K, ef_search=64, fast_mode=True)[0]
        )
        out.append(
            {
                "n_zones_requested": nz if nz == "auto" else int(nz),
                "build_s": build_s,
                "fast_mode_ef64": _point(fast_ids, t, gt, len(queries)),
                "d2": d2_nprobe(queries, gt, idx, n_probes)["rows"],
                "d3": d3_oracle_zone(queries, gt, idx, n_probes),
            }
        )
        del idx
    return {"runs": out}


def _write(out_dir: Path, name: str, scale: str, payload: Dict, meta: Dict) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{name}_{scale}.json"
    with open(path, "w") as f:
        json.dump({**meta, **payload}, f, indent=2)
    print(f"wrote {path}", flush=True)


def main(argv=None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--scale", choices=["tiny", "10k", "100k", "1m"], required=True)
    p.add_argument(
        "--checks",
        nargs="+",
        default=["d0", "d1", "d2", "d3"],
        choices=["d0", "d1", "d2", "d3", "d4"],
    )
    p.add_argument("--n-queries", type=int, default=1000)
    p.add_argument(
        "--zone-counts", nargs="+", type=int, default=[50, 100, 316, 632, 1000]
    )
    p.add_argument("--out-dir", required=True)
    args = p.parse_args(argv)

    out = Path(args.out_dir)
    meta = {
        "scale": args.scale,
        "n_vectors": SCALES[args.scale],
        "dim": DIMENSION,
        "k": K,
        "seed": SEED,
        "n_queries": args.n_queries,
        "M": HNSW_M,
        "ef_construction": EF_CONSTRUCTION,
        "threads": 1,
    }
    vectors, queries, gt = load(args.scale, args.n_queries)

    idx = None
    if set(args.checks) & {"d0", "d1", "d2", "d3"}:
        t0 = time.perf_counter()
        idx = build_zgq_index(vectors)
        meta_build = {
            "zgq_build_s": time.perf_counter() - t0,
            "n_zones_default": int(len(idx.zones.fine_centroids)),
        }
        print(f"built ZGQ in {meta_build['zgq_build_s']:.1f}s", flush=True)
        meta.update(meta_build)

    t_all = time.perf_counter()
    if "d0" in args.checks:
        _write(
            out,
            "d0_path_audit",
            args.scale,
            d0_path_audit(vectors, queries, gt, idx),
            meta,
        )
    if "d1" in args.checks:
        _write(
            out,
            "d1_insertion_order",
            args.scale,
            d1_insertion_order(vectors, queries, gt, idx),
            meta,
        )
    if "d2" in args.checks:
        _write(out, "d2_nprobe", args.scale, d2_nprobe(queries, gt, idx), meta)
    if "d3" in args.checks:
        _write(
            out, "d3_oracle_zone", args.scale, d3_oracle_zone(queries, gt, idx), meta
        )
    if "d4" in args.checks:
        _write(
            out,
            "d4_zone_sweep",
            args.scale,
            d4_zone_sweep(vectors, queries, gt, args.zone_counts),
            meta,
        )
    print(f"checks done in {time.perf_counter() - t_all:.1f}s", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
