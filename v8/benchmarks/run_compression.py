"""
Compression-first comparison at equal memory
============================================

Compares the compressed ZGQ variants (graph + PQ/RPQ codes, optional exact
rerank) against FAISS HNSW+PQ, HNSW+SQ8, IVF-PQ and IVF-PQR, with plain hnswlib
HNSW-M16 as the full-vector memory reference.

Memory is the analytic size from ``zgq.memory_model`` (RSS is unreliable on
macOS). Every index also records a cross-check: the real serialized size
(``faiss.serialize_index`` length, hnswlib saved file size, or the actual
nbytes of the ZGQ compressed arrays).

Output (schema ``zgq-compression/1``, deliberately separate from the
``zgq-bench/1`` summary.json so the canonical results still validate):
    <out-dir>/compression_summary.json
    <out-dir>/raw/<scale>_<label>.json
    <out-dir>/equal_memory_<scale>.md      (via ``--tables-only``)

Usage:
    python -m benchmarks.run_compression --scale 100k --out-dir results/<date>-improvements/compression
    python -m benchmarks.run_compression --out-dir results/<date>-improvements/compression --tables-only
    python -m benchmarks.run_compression --scale 100k --adc-check --out-dir ...
"""

import argparse
import datetime
import gc
import json
import os
import sys
import tempfile
import time
from pathlib import Path
from typing import Callable, Dict, List

import faiss
import hnswlib
import numpy as np
from zgq import ZGQConfig, ZGQIndex
from zgq import memory_model as mm
from zgq.search import compute_recall

from benchmarks.run_benchmarks import (
    DIMENSION,
    EF_CONSTRUCTION,
    SCALES,
    _graph_threads,
    collect_environment,
    exact_ground_truth,
    generate_test_data,
    measure_point,
)

SCHEMA = "zgq-compression/1"
EF_SWEEP = (16, 32, 64, 128, 256, 512)
NPROBE_SWEEP = (1, 4, 16, 64, 256)
RERANK_BUDGET = 50
# Equal-memory bands, bytes per vector (analytic, everything counted)
BANDS = (
    ("A: 60-120 B/vec", 60, 120),
    ("B: 120-200 B/vec", 120, 200),
    ("C: 200-300 B/vec", 200, 300),
    ("ref: >=500 B/vec", 500, 1e9),
)


def _variant(label, family, params, model, search, param_name, values, crosscheck):
    return {
        "label": label,
        "family": family,
        "params": params,
        "model": model,
        "search": search,
        "param_name": param_name,
        "values": list(values),
        "crosscheck": crosscheck,
    }


# ----------------------------------------------------------------------------
# Builders: each returns (list of variants, build_time_s)
# ----------------------------------------------------------------------------


def build_zgq_compressed(
    vectors, M, pq_m, residual, rerank_variant, seed, threads, ef_values
):
    n, d = vectors.shape
    t0 = time.perf_counter()
    np.random.seed(seed)
    cfg = ZGQConfig(
        M=M,
        ef_construction=EF_CONSTRUCTION,
        random_state=seed,
        pq_m=pq_m,
        use_residual_pq=residual,
        compressed_search=True,
        insertion_order="random",
    )
    with _graph_threads(threads):
        idx = ZGQIndex(cfg).build(vectors)
    build_t = time.perf_counter() - t0
    n_zones = len(idx.zones.fine_centroids)
    tag = "RPQ" if residual else "PQ"
    variants = []
    # codes only: what a deployment without raw vectors would hold
    actual = idx.compressed_memory_bytes()
    model = mm.zgq_compressed_bytes(
        n, d, M=M, pq_m=pq_m, n_zones=n_zones, residual=residual
    )
    assert actual["total_bytes"] == model["total_bytes"]
    params = {
        "M": M,
        "pq_m": pq_m,
        "pq_bits": 8,
        "n_zones": n_zones,
        "residual": residual,
        "ef_construction": EF_CONSTRUCTION,
        "insertion_order": "random",
        "rerank": "none (codes only)",
    }

    def search_codes(queries, k, ef):
        idx.config.rerank_budget = 0
        return idx.batch_search(queries, k, ef_search=ef)[0]

    variants.append(
        _variant(
            f"ZGQ-C-{tag}-M{M}-pq{pq_m}",
            "zgq_compressed",
            params,
            model,
            search_codes,
            "ef",
            ef_values,
            {
                "method": "actual nbytes of stored arrays",
                "bytes": actual["total_bytes"],
            },
        )
    )
    if rerank_variant:
        model_r = mm.zgq_compressed_bytes(
            n, d, M=M, pq_m=pq_m, n_zones=n_zones, residual=residual, keep_vectors=True
        )

        def search_rerank(queries, k, ef):
            idx.config.rerank_budget = RERANK_BUDGET
            return idx.batch_search(queries, k, ef_search=ef)[0]

        p2 = dict(
            params,
            rerank=f"exact over top {RERANK_BUDGET} ADC candidates (raw vectors kept)",
        )
        variants.append(
            _variant(
                f"ZGQ-C-{tag}-M{M}-pq{pq_m}+rerank{RERANK_BUDGET}",
                "zgq_compressed_rerank",
                p2,
                model_r,
                search_rerank,
                "ef",
                [e for e in ef_values if e >= RERANK_BUDGET],
                {
                    "method": "actual nbytes of stored arrays + raw vectors",
                    "bytes": actual["total_bytes"] + vectors.nbytes,
                },
            )
        )
    return variants, build_t, idx


def build_hnswlib(vectors, M, seed, threads):
    n, d = vectors.shape
    t0 = time.perf_counter()
    idx = hnswlib.Index(space="l2", dim=d)
    idx.init_index(
        max_elements=n, ef_construction=EF_CONSTRUCTION, M=M, random_seed=seed
    )
    idx.set_num_threads(threads)
    idx.add_items(vectors, np.arange(n))
    build_t = time.perf_counter() - t0
    idx.set_num_threads(1)
    fd, path = tempfile.mkstemp()
    os.close(fd)
    idx.save_index(path)
    size = os.path.getsize(path)
    os.remove(path)

    def search(queries, k, ef):
        idx.set_ef(ef)
        return idx.knn_query(queries, k=k)[0]

    v = _variant(
        f"HNSW-M{M}",
        "hnswlib_full",
        {"M": M, "ef_construction": EF_CONSTRUCTION},
        mm.hnswlib_bytes(n, d, M),
        search,
        "ef",
        EF_SWEEP,
        {"method": "hnswlib save_index file size", "bytes": size},
    )
    return [v], build_t, idx


def build_faiss_hnswpq(vectors, M, pq_m, seed, threads):
    n, d = vectors.shape
    faiss.omp_set_num_threads(threads)
    t0 = time.perf_counter()
    idx = faiss.IndexHNSWPQ(d, pq_m, M)
    idx.hnsw.efConstruction = EF_CONSTRUCTION
    idx.train(vectors[:100000])
    idx.add(vectors)
    build_t = time.perf_counter() - t0
    faiss.omp_set_num_threads(1)

    def search(queries, k, ef):
        idx.hnsw.efSearch = ef
        return idx.search(queries, k)[1]

    v = _variant(
        f"HNSWPQ-M{M}-pq{pq_m}",
        "faiss_hnswpq",
        {"M": M, "pq_m": pq_m, "ef_construction": EF_CONSTRUCTION},
        mm.faiss_hnswpq_bytes(n, d, M, pq_m),
        search,
        "ef",
        EF_SWEEP,
        {
            "method": "len(faiss.serialize_index)",
            "bytes": len(faiss.serialize_index(idx)),
        },
    )
    return [v], build_t, idx


def build_faiss_hnswsq(vectors, M, seed, threads):
    n, d = vectors.shape
    faiss.omp_set_num_threads(threads)
    t0 = time.perf_counter()
    idx = faiss.IndexHNSWSQ(d, faiss.ScalarQuantizer.QT_8bit, M)
    idx.hnsw.efConstruction = EF_CONSTRUCTION
    idx.train(vectors[:100000])
    idx.add(vectors)
    build_t = time.perf_counter() - t0
    faiss.omp_set_num_threads(1)

    def search(queries, k, ef):
        idx.hnsw.efSearch = ef
        return idx.search(queries, k)[1]

    v = _variant(
        f"HNSWSQ8-M{M}",
        "faiss_hnswsq8",
        {"M": M, "ef_construction": EF_CONSTRUCTION},
        mm.faiss_hnswsq8_bytes(n, d, M),
        search,
        "ef",
        EF_SWEEP,
        {
            "method": "len(faiss.serialize_index)",
            "bytes": len(faiss.serialize_index(idx)),
        },
    )
    return [v], build_t, idx


def build_faiss_ivfpq(vectors, pq_m, refine_m, seed, threads):
    n, d = vectors.shape
    nlist = int(2 * np.sqrt(n))
    faiss.omp_set_num_threads(threads)
    t0 = time.perf_counter()
    quant = faiss.IndexFlatL2(d)
    if refine_m:
        idx = faiss.IndexIVFPQR(quant, d, nlist, pq_m, 8, refine_m, 8)
        label = f"IVFPQR-pq{pq_m}-r{refine_m}"
        family = "faiss_ivfpqr"
    else:
        idx = faiss.IndexIVFPQ(quant, d, nlist, pq_m, 8)
        label = f"IVFPQ-pq{pq_m}"
        family = "faiss_ivfpq"
    idx.cp.seed = seed
    idx.pq.cp.seed = seed
    idx.train(vectors[: max(100000, 40 * nlist)])
    idx.add(vectors)
    build_t = time.perf_counter() - t0
    faiss.omp_set_num_threads(1)

    def search(queries, k, nprobe):
        idx.nprobe = nprobe
        if refine_m:
            idx.k_factor = 4
        return idx.search(queries, k)[1]

    v = _variant(
        label,
        family,
        {
            "nlist": nlist,
            "pq_m": pq_m,
            "refine_m": refine_m,
            "k_factor": 4 if refine_m else None,
        },
        mm.faiss_ivfpq_bytes(n, d, nlist, pq_m, refine_m=refine_m or None),
        search,
        "nprobe",
        NPROBE_SWEEP,
        {
            "method": "len(faiss.serialize_index)",
            "bytes": len(faiss.serialize_index(idx)),
        },
    )
    return [v], build_t, idx


def build_plan(args) -> List[Dict]:
    """Ordered build plan: label of the first variant -> builder closure."""
    th = args.build_threads
    seed = args.seed
    plan = []
    plan.append(("HNSW-M16", lambda v: build_hnswlib(v, 16, seed, th)))
    for M, pq_m, rr in ((8, 16, True), (8, 32, False), (16, 16, False), (16, 32, True)):
        plan.append(
            (
                f"ZGQ-C-RPQ-M{M}-pq{pq_m}",
                lambda v, M=M, pq_m=pq_m, rr=rr: build_zgq_compressed(
                    v, M, pq_m, True, rr, seed, th, EF_SWEEP
                ),
            )
        )
    plan.append(
        (
            "ZGQ-C-PQ-M16-pq32",
            lambda v: build_zgq_compressed(v, 16, 32, False, False, seed, th, EF_SWEEP),
        )
    )
    plan.append(
        (
            "ZGQ-C-PQ-M8-pq16",
            lambda v: build_zgq_compressed(v, 8, 16, False, False, seed, th, EF_SWEEP),
        )
    )
    for M, pq_m in ((8, 16), (8, 32), (16, 16), (16, 32)):
        plan.append(
            (
                f"HNSWPQ-M{M}-pq{pq_m}",
                lambda v, M=M, pq_m=pq_m: build_faiss_hnswpq(v, M, pq_m, seed, th),
            )
        )
    for M in (8, 16):
        plan.append(
            (f"HNSWSQ8-M{M}", lambda v, M=M: build_faiss_hnswsq(v, M, seed, th))
        )
    for pq_m, rm in ((16, None), (32, None), (64, None), (16, 64), (32, 32), (32, 64)):
        plan.append(
            (
                f"IVFPQ-pq{pq_m}" if not rm else f"IVFPQR-pq{pq_m}-r{rm}",
                lambda v, pq_m=pq_m, rm=rm: build_faiss_ivfpq(v, pq_m, rm, seed, th),
            )
        )
    if args.only:
        plan = [p for p in plan if any(p[0].startswith(o) for o in args.only)]
    return plan


# ----------------------------------------------------------------------------
# Run
# ----------------------------------------------------------------------------


def run(args) -> Dict:
    out_dir = Path(args.out_dir)
    (out_dir / "raw").mkdir(parents=True, exist_ok=True)
    path = out_dir / "compression_summary.json"
    if path.exists():
        summary = json.loads(path.read_text())
    else:
        summary = {
            "schema_version": SCHEMA,
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
                "search_threads": 1,
                "build_threads": args.build_threads,
                "latency_metric": "wall time of one batch call / n_queries, single thread",
                "memory_metric": "analytic bytes per vector (zgq.memory_model), "
                "cross-checked against serialized/actual sizes",
                "data": "clustered Gaussian, dim 128, 70% cluster queries / 30% uniform",
                "ground_truth": "exact brute force L2",
            },
            "scales": {},
        }
    for scale in args.scale:
        n = SCALES[scale]
        print(f"\n=== scale {scale}: {n:,} vectors ===", flush=True)
        vectors, queries = generate_test_data(n, args.n_queries, DIMENSION, args.seed)
        gt = exact_ground_truth(vectors, queries, args.k)
        faiss.omp_set_num_threads(1)
        sc = summary["scales"].setdefault(
            scale,
            {
                "n_vectors": n,
                "dimension": DIMENSION,
                "n_queries": args.n_queries,
                "indexes": [],
            },
        )
        done = {i["_first"] for i in sc["indexes"]}
        for first_label, builder in build_plan(args):
            if first_label in done:
                continue
            print(f"  [{scale}] {first_label}: building ...", flush=True)
            variants, build_t, obj = builder(vectors)
            print(f"  [{scale}] {first_label}: build {build_t:.1f}s", flush=True)
            for var in variants:
                points, raw_runs = [], []
                for p in var["values"]:
                    pt, runs = measure_point(
                        var["search"], queries, gt, args.k, p, args.repeats, args.warmup
                    )
                    sp = {var["param_name"]: int(p)}
                    points.append({"search_param": sp, **pt})
                    raw_runs += [{"search_param": sp, **r} for r in runs]
                    print(
                        f"    {var['label']} {var['param_name']}={p:<4} "
                        f"recall={pt['recall_pct']['median']:.1f}% "
                        f"lat={pt['latency_ms_per_query']['median']:.4f}ms",
                        flush=True,
                    )
                raw_name = f"raw/{scale}_{var['label']}.json"
                (out_dir / raw_name).write_text(
                    json.dumps(
                        {
                            "label": var["label"],
                            "scale": scale,
                            "build": {"time_s": build_t},
                            "runs": raw_runs,
                        },
                        indent=1,
                    )
                )
                m = var["model"]
                sc["indexes"].append(
                    {
                        "label": var["label"],
                        "family": var["family"],
                        "params": var["params"],
                        "build": {"time_s": build_t, "shared_with": first_label},
                        "memory": {
                            "model_total_bytes": m["total_bytes"],
                            "bytes_per_vector": m["bytes_per_vector"],
                            "components": m["components"],
                            "crosscheck": var["crosscheck"],
                            "crosscheck_ratio": var["crosscheck"]["bytes"]
                            / m["total_bytes"],
                        },
                        "points": points,
                        "raw_file": raw_name,
                        "_first": first_label,
                    }
                )
            del obj, variants
            gc.collect()
            path.write_text(json.dumps(summary, indent=2))
    path.write_text(json.dumps(summary, indent=2))
    print(f"\nWrote {path}")
    return summary


# ----------------------------------------------------------------------------
# Equal-memory tables
# ----------------------------------------------------------------------------


def _best(idx: Dict, key: str):
    pts = idx["points"]
    return max(pts, key=lambda p: p["recall_pct"]["median"])


def _first_reaching(idx: Dict, target: float):
    for p in sorted(idx["points"], key=lambda p: p["latency_ms_per_query"]["median"]):
        if p["recall_pct"]["median"] >= target:
            return p
    return None


def equal_memory_table(sc: Dict, scale: str) -> str:
    lines = [
        f"# Equal-memory comparison, {scale} ({sc['n_vectors']:,} vectors, dim {sc['dimension']})",
        "",
        "Bytes/vector is the analytic model (`zgq.memory_model`), counting everything the index "
        "needs to answer queries. `max recall` is the best point in the sweep. "
        "`ms @ R>=x` is the lowest median latency among sweep points reaching recall x "
        "(`-` = never reached).",
        "",
    ]
    for name, lo, hi in BANDS:
        members = [
            i for i in sc["indexes"] if lo <= i["memory"]["bytes_per_vector"] < hi
        ]
        if not members:
            continue
        members.sort(key=lambda i: -_best(i, "")["recall_pct"]["median"])
        lines += [
            f"## Band {name}",
            "",
            "| index | B/vec | max recall % | ms at max | ms @ R>=70 | ms @ R>=90 | ms @ R>=95 |",
            "|---|---:|---:|---:|---:|---:|---:|",
        ]
        for i in members:
            b = _best(i, "")
            cells = []
            for t in (70, 90, 95):
                p = _first_reaching(i, t)
                cells.append(
                    "-" if p is None else f"{p['latency_ms_per_query']['median']:.3f}"
                )
            lines.append(
                f"| {i['label']} | {i['memory']['bytes_per_vector']:.1f} | "
                f"{b['recall_pct']['median']:.1f} | {b['latency_ms_per_query']['median']:.3f} | "
                + " | ".join(cells)
                + " |"
            )
        lines.append("")
    lines += [
        "## Model cross-check",
        "",
        "| index | model B/vec | measured/serialized B/vec | ratio |",
        "|---|---:|---:|---:|",
    ]
    for i in sc["indexes"]:
        mem = i["memory"]
        lines.append(
            f"| {i['label']} | {mem['bytes_per_vector']:.1f} | "
            f"{mem['crosscheck']['bytes'] / sc['n_vectors']:.1f} | {mem['crosscheck_ratio']:.3f} |"
        )
    return "\n".join(lines) + "\n"


# ----------------------------------------------------------------------------
# ADC check: nearest-zone table vs per-candidate zone centroid
# ----------------------------------------------------------------------------


def adc_check(args) -> Dict:
    """Measure the ZGQIndex.search PQ-shortlist issue (briefing hypothesis)."""
    scale = args.scale[0]
    n = SCALES[scale]
    vectors, queries = generate_test_data(n, args.n_queries, DIMENSION, args.seed)
    gt = exact_ground_truth(vectors, queries, args.k)
    nq = min(args.adc_queries, len(queries))
    k = args.k
    np.random.seed(args.seed)
    cfg = ZGQConfig(
        M=16,
        ef_construction=EF_CONSTRUCTION,
        random_state=args.seed,
        compressed_search=True,
        insertion_order="random",
    )
    with _graph_threads(args.build_threads):
        idx = ZGQIndex(cfg).build(vectors)
    zones, pq = idx.zones, idx.pq
    cent = zones.fine_centroids
    recon = pq.decode(idx.pq_codes) + cent[zones.assignments]  # c_z + decoded residual
    rows = []
    for n_probe in (1, 4, 16, 64):
        hit = {
            "no_pq_shortlist": 0,
            "nearest_zone_table": 0,
            "per_candidate_zone_adc": 0,
        }
        err = {"nearest_zone_table": [], "per_candidate_zone_adc": []}
        ncand = []
        for qi in range(nq):
            q = queries[qi]
            sel = idx._choose_zones(q, n_probe)
            cand, _ = idx.graph.search(q, min(k * len(sel), n), sel)
            cand = np.asarray(cand, dtype=np.int64)
            ncand.append(len(cand))
            exact = ((vectors[cand] - q) ** 2).sum(1)
            table = pq.compute_distance_table(q, cent[sel[0]])
            wrong = pq.asymmetric_distance(idx.pq_codes[cand], table)
            right = ((recon[cand] - q) ** 2).sum(1)
            err["nearest_zone_table"].append(
                float(np.mean(np.abs(wrong - exact) / exact))
            )
            err["per_candidate_zone_adc"].append(
                float(np.mean(np.abs(right - exact) / exact))
            )
            r = min(2 * k, len(cand))
            for name, score in (
                ("no_pq_shortlist", exact),
                ("nearest_zone_table", wrong),
                ("per_candidate_zone_adc", right),
            ):
                short = cand[np.argsort(score)[:r]]
                ex = ((vectors[short] - q) ** 2).sum(1)
                top = short[np.argsort(ex)[:k]]
                hit[name] += len(set(top.tolist()) & set(gt[qi].tolist()))
        rows.append(
            {
                "n_probe": n_probe,
                "mean_candidates": float(np.mean(ncand)),
                "recall_pct": {k_: 100.0 * v / (nq * k) for k_, v in hit.items()},
                "mean_relative_adc_error": {
                    k_: float(np.mean(v)) for k_, v in err.items()
                },
            }
        )
        print(rows[-1], flush=True)
    out = {
        "scale": scale,
        "n_queries": nq,
        "k": k,
        "rows": rows,
        "note": "graph.search candidates of ZGQIndex.search; shortlist 2k by score, exact rerank",
    }
    p = Path(args.out_dir) / f"adc_check_{scale}.json"
    p.write_text(json.dumps(out, indent=2))
    print(f"Wrote {p}")
    return out


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Equal-memory compression benchmark")
    p.add_argument("--scale", nargs="+", default=["100k"], choices=list(SCALES))
    p.add_argument("--out-dir", required=True)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--k", type=int, default=10)
    p.add_argument("--n-queries", type=int, default=1000)
    p.add_argument("--repeats", type=int, default=5)
    p.add_argument("--warmup", type=int, default=1)
    p.add_argument("--build-threads", type=int, default=8)
    p.add_argument(
        "--only",
        nargs="+",
        help="only plan entries whose label starts with one of these",
    )
    p.add_argument("--tables-only", action="store_true")
    p.add_argument("--adc-check", action="store_true")
    p.add_argument("--adc-queries", type=int, default=200)
    return p


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    if args.adc_check:
        adc_check(args)
        return 0
    if args.tables_only:
        summary = json.loads(
            (Path(args.out_dir) / "compression_summary.json").read_text()
        )
        for scale, sc in summary["scales"].items():
            (Path(args.out_dir) / f"equal_memory_{scale}.md").write_text(
                equal_memory_table(sc, scale)
            )
            print(f"wrote equal_memory_{scale}.md")
        return 0
    run(args)
    return 0


if __name__ == "__main__":
    sys.exit(main())
