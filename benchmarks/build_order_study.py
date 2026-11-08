import argparse
import itertools
import json
import multiprocessing
import os
import shutil
import tempfile
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Dict, List, Optional

import hnswlib
import numpy as np
from zgq.core.build_order import insertion_order
from zgq.core.zones import AdaptiveHierarchicalZones, ZoneConfig
from zgq.memory_model import hnswlib_bytes

from benchmarks.datasets import load_dataset
from benchmarks.graph_stats import analyze_graph, read_hnsw_index, search_hops
from benchmarks.run_benchmarks import (
    EF_CONSTRUCTION,
    HNSW_M,
    collect_environment,
    measure_point,
    stats,
)

SCHEMA = "zgq-build-order/1"
DATA_SEED = 0
K = 10
HOP_EFS = (64, 256)
TARGETS = (80.0, 90.0, 95.0)
BOOTSTRAP = 2000
EXPERIMENTS = ("e1", "e2", "e3")
BASELINE = {"e1": "random", "e2": "random", "e3": "e3_random"}


def make_config(
    experiment="e1",
    dataset="synthetic",
    n_base=10000,
    strategy="zone",
    seed=0,
    ef=(16, 32, 64, 128, 256, 512),
    threads=1,
    skeleton_fraction=0.05,
    n_queries=None,
    repeats=5,
    warmup=1,
    concurrent_jobs=0,
    build_repeats=3,
) -> Dict:
    if experiment not in EXPERIMENTS:
        raise ValueError(f"experiment must be one of {EXPERIMENTS}")
    if dataset == "synthetic":
        name = f"synthetic-{n_base}"
        n_queries = 1000 if n_queries is None else n_queries
    else:
        name = dataset
    return {
        "experiment": experiment,
        "dataset": name,
        "n_base": None if n_base in (None, "full") else int(n_base),
        "strategy": strategy,
        "skeleton_fraction": float(skeleton_fraction),
        "seed": int(seed),
        "ef": [int(e) for e in ef],
        "threads": int(threads),
        "n_queries": n_queries,
        "repeats": int(repeats),
        "warmup": int(warmup),
        "build_repeats": int(build_repeats),
        "concurrent_jobs": int(concurrent_jobs),
        "k": K,
        "M": HNSW_M,
        "ef_construction": EF_CONSTRUCTION,
    }


def strategy_key(cfg: Dict) -> str:
    if cfg["strategy"] == "skeleton":
        return f"skeleton-{cfg['skeleton_fraction']:g}"
    return cfg["strategy"]


def run_path(out_dir, cfg: Dict) -> Path:
    n = cfg["n_base"] if cfg["n_base"] is not None else "full"
    name = f"{cfg['dataset']}_n{n}_{strategy_key(cfg)}_s{cfg['seed']}_t{cfg['threads']}.json"
    return Path(out_dir) / cfg["experiment"] / name


def build_zones(vectors, seed):
    zc = ZoneConfig(
        auto_zones=True, use_hierarchy=True, random_state=seed, verbose=False
    )
    zones = AdaptiveHierarchicalZones(zc)
    t0 = time.perf_counter()
    zones.build(vectors)
    return zones, time.perf_counter() - t0


def compute_order(strategy, vectors, assignments, centroids, seed, skeleton_fraction):
    return insertion_order(
        strategy, vectors, assignments, centroids, seed, skeleton_fraction
    )


def _new_index(dim, n, seed, threads):
    idx = hnswlib.Index(space="l2", dim=dim)
    idx.init_index(
        max_elements=n, ef_construction=EF_CONSTRUCTION, M=HNSW_M, random_seed=seed
    )
    idx.set_num_threads(threads)
    return idx


def _hits(ids, gt, k):
    ids = np.asarray(ids)
    return np.array(
        [len(set(map(int, a[:k])) & set(map(int, b[:k]))) for a, b in zip(ids, gt)],
        dtype=np.int64,
    )


def _sweep(idx, queries, gt, cfg):
    def search(q, k, ef):
        idx.set_ef(ef)
        return idx.knn_query(q, k=k, num_threads=1)[0]

    idx.set_num_threads(1)
    points, per_query = [], {}
    for ef in cfg["ef"]:
        summary, _ = measure_point(
            search, queries, gt, cfg["k"], ef, cfg["repeats"], cfg["warmup"]
        )
        points.append({"ef": ef, **summary})
        per_query[str(ef)] = _hits(search(queries, cfg["k"], ef), gt, cfg["k"]).tolist()
    return points, per_query


def _quick_recall(idx, queries, gt, cfg):
    def search(q, k, ef):
        idx.set_ef(ef)
        return idx.knn_query(q, k=k, num_threads=1)[0]

    out = {}
    for ef in cfg["ef"]:
        s, _ = measure_point(search, queries, gt, cfg["k"], ef, 1, 0)
        out[str(ef)] = s["recall_pct"]["median"]
    return out


def run_one(cfg: Dict) -> Dict:
    seed, k = cfg["seed"], cfg["k"]
    ds = load_dataset(cfg["dataset"], cfg["n_base"], cfg["n_queries"], k, DATA_SEED)
    vectors, queries, gt = ds.vectors, ds.queries, ds.gt
    n, dim = vectors.shape
    exp = cfg["experiment"]

    zones, zone_time = build_zones(vectors, seed)
    assign, cents = zones.assignments, zones.fine_centroids
    n_zones = int(len(cents))

    run = {
        "schema": SCHEMA,
        "kind": "run",
        "environment": collect_environment(),
        "dataset": {
            "name": ds.name,
            "n": int(n),
            "dim": int(dim),
            "n_queries": int(len(queries)),
            "metric": ds.metric,
            "source": ds.source,
        },
        "config": cfg,
        "strategy_key": strategy_key(cfg),
        "concurrent_jobs": cfg["concurrent_jobs"],
        "zone_time_s": zone_time,
        "n_zones": n_zones,
        "analytic_bytes": hnswlib_bytes(n, dim, HNSW_M)["total_bytes"],
        "analytic_bytes_source": "zgq.memory_model.hnswlib_bytes",
    }

    tmp = tempfile.mkdtemp(prefix="zgq_study_")
    try:
        if exp == "e3":
            idx = _run_e3(run, cfg, vectors, assign, cents)
        else:
            idx = _run_build(run, cfg, vectors, assign, cents)

        path = os.path.join(tmp, "index.bin")
        idx.save_index(path)
        run["index_file_bytes"] = os.path.getsize(path)
        run["points"], run["per_query_hits"] = _sweep(idx, queries, gt, cfg)
        _finalize_extra(run, queries, gt, cfg)

        if exp == "e3":
            run["graph_stats"], run["hops"] = None, {}
        else:
            g = read_hnsw_index(path)
            run["graph_stats"] = analyze_graph(
                g, g.vectors, np.asarray(assign)[g.labels], seed=seed
            )
            run["hops"] = {
                str(ef): search_hops(g, None, queries, ef, k)["summary"]
                for ef in HOP_EFS
            }
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    return run


def _run_build(run, cfg, vectors, assign, cents):
    n, dim = vectors.shape
    t0 = time.perf_counter()
    order = compute_order(
        cfg["strategy"], vectors, assign, cents, cfg["seed"], cfg["skeleton_fraction"]
    )
    run["order_time_s"] = time.perf_counter() - t0
    assert len(order) == n and len(np.unique(order)) == n

    n_builds = cfg["build_repeats"] if cfg["experiment"] == "e2" else 1
    times, extra = [], []
    idx = None
    for r in range(n_builds):
        idx = _new_index(dim, n, cfg["seed"], cfg["threads"])
        t0 = time.perf_counter()
        idx.add_items(vectors[order], order)
        times.append(time.perf_counter() - t0)
        if r < n_builds - 1:
            extra.append(idx)
    run["build_time_s"] = float(np.median(times))
    if n_builds > 1:
        run["_extra_indexes"] = extra
        run["build_repeats"] = [{"build_time_s": t} for t in times]
    return idx


def _run_e3(run, cfg, vectors, assign, cents):
    n, dim = vectors.shape
    perm = np.random.RandomState(cfg["seed"]).permutation(n).astype(np.int64)
    n_base = int(0.8 * n)
    base, rest = perm[:n_base], perm[n_base:]
    if cfg["strategy"] == "e3_zone":
        zone = compute_order("zone", vectors, assign, cents, cfg["seed"], 0.0)
        in_rest = np.zeros(n, dtype=bool)
        in_rest[rest] = True
        rest = zone[in_rest[zone]]
    elif cfg["strategy"] != "e3_random":
        raise ValueError("e3 strategy must be e3_random or e3_zone")
    run["order_time_s"] = 0.0
    idx = _new_index(dim, n, cfg["seed"], cfg["threads"])
    t0 = time.perf_counter()
    idx.add_items(vectors[base], base)
    run["build_time_s"] = time.perf_counter() - t0
    t0 = time.perf_counter()
    idx.add_items(vectors[rest], rest)
    run["insertion_time_s"] = time.perf_counter() - t0
    run["n_inserted_incrementally"] = int(len(rest))
    return idx


def _finalize_extra(run, queries, gt, cfg):
    extra = run.pop("_extra_indexes", None)
    if extra:
        for rec, idx in zip(run["build_repeats"], extra):
            rec["recall_pct"] = _quick_recall(idx, queries, gt, cfg)


_RUN_KEYS = (
    "schema",
    "environment",
    "dataset",
    "config",
    "build_time_s",
    "zone_time_s",
    "order_time_s",
    "n_zones",
    "index_file_bytes",
    "analytic_bytes",
    "points",
    "graph_stats",
    "hops",
    "concurrent_jobs",
    "strategy_key",
)


def check_run(run: Dict) -> List[str]:
    errs = []
    if run.get("schema") != SCHEMA:
        errs.append(f"schema must be {SCHEMA!r}")
    for key in _RUN_KEYS:
        if key not in run:
            errs.append(f"missing key {key!r}")
    pts = run.get("points")
    if not isinstance(pts, list) or not pts:
        errs.append("points must be a non-empty list")
    else:
        for p in pts:
            for key in ("ef", "recall_pct", "latency_ms_per_query"):
                if key not in p:
                    errs.append(f"point missing {key!r}")
    if (
        run.get("config", {}).get("experiment") == "e3"
        and "insertion_time_s" not in run
    ):
        errs.append("e3 run missing insertion_time_s")
    return errs


def _atomic_write(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + f".{os.getpid()}.tmp")
    with open(tmp, "w") as f:
        json.dump(obj, f, indent=1)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


def execute(cfg: Dict, out_dir: str) -> Path:
    path = run_path(out_dir, cfg)
    if path.exists():
        return path
    run = run_one(cfg)
    errs = check_run(run)
    if errs:
        raise RuntimeError(f"invalid run for {path.name}: {errs}")
    _atomic_write(path, run)
    return path


def latency_at_recall(recalls, lats, target) -> Optional[float]:
    pts = sorted(zip(recalls, lats))
    if pts[-1][0] < target:
        return None
    if pts[0][0] >= target:
        return pts[0][1]
    for (r0, l0), (r1, l1) in zip(pts, pts[1:]):
        if r0 <= target <= r1:
            return l1 if r1 == r0 else l0 + (l1 - l0) * (target - r0) / (r1 - r0)
    return None


def _flatten(d, prefix=""):
    out = {}
    if isinstance(d, dict):
        for key, v in d.items():
            out.update(_flatten(v, f"{prefix}{key}."))
    elif isinstance(d, (int, float)) and not isinstance(d, bool):
        out[prefix[:-1]] = float(d)
    return out


def _stat_over(values):
    return stats(values) if values else None


def _boot_ci(samples: np.ndarray, rng) -> Dict:
    n = len(samples)
    idx = rng.randint(0, n, size=(BOOTSTRAP, n))
    means = samples[idx].mean(axis=1)
    lo, hi = np.percentile(means, [2.5, 97.5])
    return {"diff": float(samples.mean()), "ci_low": float(lo), "ci_high": float(hi)}


def _point_map(run, field):
    return {str(p["ef"]): p[field]["median"] for p in run["points"]}


def _paired(group_runs, base_runs, ef_list) -> Dict:
    rng = np.random.RandomState(0)
    by_seed = {r["config"]["seed"]: r for r in group_runs}
    base_seed = {r["config"]["seed"]: r for r in base_runs}
    common = sorted(set(by_seed) & set(base_seed))
    out = {"recall_pct": {}, "latency_ms": {}, "build_time_s": None, "seeds": common}
    if len(common) >= 2:
        for ef in ef_list:
            for fld, name in (
                ("recall_pct", "recall_pct"),
                ("latency_ms_per_query", "latency_ms"),
            ):
                d = np.array(
                    [
                        _point_map(by_seed[s], fld)[ef]
                        - _point_map(base_seed[s], fld)[ef]
                        for s in common
                    ]
                )
                out[name][ef] = {
                    **_boot_ci(d, rng),
                    "method": "bootstrap over seeds",
                    "n": len(common),
                }
        d = np.array(
            [by_seed[s]["build_time_s"] - base_seed[s]["build_time_s"] for s in common]
        )
        out["build_time_s"] = {
            **_boot_ci(d, rng),
            "method": "bootstrap over seeds",
            "n": len(common),
        }
    elif len(common) == 1:
        a, b = by_seed[common[0]], base_seed[common[0]]
        ha, hb = a.get("per_query_hits"), b.get("per_query_hits")
        if ha and hb:
            kk = a["config"].get("k", K)
            for ef in ef_list:
                d = (np.array(ha[ef]) - np.array(hb[ef])) / kk * 100.0
                out["recall_pct"][ef] = {
                    **_boot_ci(d, rng),
                    "method": "bootstrap over queries",
                    "n": int(len(d)),
                }
    return out


def aggregate(runs: List[Dict]) -> Dict:
    groups: Dict[tuple, List[Dict]] = {}
    for r in runs:
        c = r["config"]
        key = (
            c["experiment"],
            r["dataset"]["name"]
            if "dataset" in r and isinstance(r.get("dataset"), dict)
            else c["dataset"],
            c["n_base"],
            r["strategy_key"],
            c.get("threads", 1),
        )
        groups.setdefault(key, []).append(r)

    out_groups = []
    for key in sorted(groups, key=lambda k: tuple(str(x) for x in k)):
        exp, dset, n_base, skey, threads = key
        rs = groups[key]
        efs = [str(p["ef"]) for p in rs[0]["points"]]
        rec = {
            ef: _stat_over([_point_map(r, "recall_pct")[ef] for r in rs]) for ef in efs
        }
        lat = {
            ef: _stat_over([_point_map(r, "latency_ms_per_query")[ef] for r in rs])
            for ef in efs
        }
        flat = [_flatten(r.get("graph_stats") or {}) for r in rs]
        structure = (
            {
                m: float(np.median([f[m] for f in flat if m in f]))
                for m in sorted(set().union(*flat))
            }
            if flat
            else {}
        )
        hops = {}
        for ef in sorted(set().union(*[set((r.get("hops") or {}).keys()) for r in rs])):
            fl = [_flatten(r["hops"][ef]) for r in rs if ef in r.get("hops", {})]
            hops[ef] = {
                m: float(np.median([f[m] for f in fl if m in f]))
                for m in sorted(set().union(*fl))
            }
        recs = [rec[ef]["median"] for ef in efs]
        lats = [lat[ef]["median"] for ef in efs]
        grp = {
            "experiment": exp,
            "dataset": dset,
            "n_base": n_base,
            "strategy": skey,
            "threads": threads,
            "n_seeds": len({r["config"]["seed"] for r in rs}),
            "seeds": sorted({r["config"]["seed"] for r in rs}),
            "ef": [int(e) for e in efs],
            "recall_pct": rec,
            "latency_ms": lat,
            "build_time_s": _stat_over([r["build_time_s"] for r in rs]),
            "zone_time_s": _stat_over([r["zone_time_s"] for r in rs]),
            "order_time_s": _stat_over([r["order_time_s"] for r in rs]),
            "index_file_bytes": int(np.median([r["index_file_bytes"] for r in rs]))
            if all("index_file_bytes" in r for r in rs)
            else None,
            "concurrent_jobs_max": max(r.get("concurrent_jobs", 0) for r in rs),
            "structure": structure,
            "hops": hops,
            "equal_recall_latency_ms": {
                f"{t:g}": latency_at_recall(recs, lats, t) for t in TARGETS
            },
            "paired_vs_random": None,
        }
        if exp == "e3":
            grp["insertion_time_s"] = _stat_over(
                [r["insertion_time_s"] for r in rs if "insertion_time_s" in r]
            )
        if exp == "e2":
            grp["build_repeat_recall_pct"] = {
                ef: _stat_over(
                    [
                        b["recall_pct"][ef]
                        for r in rs
                        for b in r.get("build_repeats", [])
                        if "recall_pct" in b
                    ]
                )
                for ef in efs
            }
        grp["_runs"] = rs
        out_groups.append(grp)

    index = {
        (g["experiment"], g["dataset"], g["n_base"], g["strategy"], g["threads"]): g
        for g in out_groups
    }
    for g in out_groups:
        base = index.get(
            (
                g["experiment"],
                g["dataset"],
                g["n_base"],
                BASELINE[g["experiment"]],
                g["threads"],
            )
        )
        if base is not None and base is not g:
            g["paired_vs_random"] = _paired(
                g["_runs"], base["_runs"], [str(e) for e in g["ef"]]
            )
    for g in out_groups:
        del g["_runs"]
    return {
        "schema": SCHEMA,
        "kind": "summary",
        "n_runs": len(runs),
        "bootstrap_resamples": BOOTSTRAP,
        "note": "paired_vs_random is strategy minus baseline (random, or e3_random for e3); "
        "method is 'bootstrap over seeds' with >=2 shared seeds, otherwise "
        "'bootstrap over queries'.",
        "environment": runs[0].get("environment") if runs else None,
        "groups": out_groups,
    }


_GROUP_KEYS = (
    "experiment",
    "dataset",
    "n_base",
    "strategy",
    "n_seeds",
    "ef",
    "recall_pct",
    "latency_ms",
    "build_time_s",
    "zone_time_s",
    "order_time_s",
    "concurrent_jobs_max",
    "structure",
    "hops",
    "equal_recall_latency_ms",
    "paired_vs_random",
)


def check_summary(summary: Dict) -> List[str]:
    errs = []
    if summary.get("schema") != SCHEMA:
        errs.append(f"schema must be {SCHEMA!r}")
    groups = summary.get("groups")
    if not isinstance(groups, list) or not groups:
        errs.append("groups must be a non-empty list")
        return errs
    for g in groups:
        for key in _GROUP_KEYS:
            if key not in g:
                errs.append(f"group {g.get('strategy')!r} missing {key!r}")
        for ef, s in (g.get("recall_pct") or {}).items():
            if not s or not (s["q1"] <= s["median"] <= s["q3"]):
                errs.append(f"bad recall stats at ef={ef} in {g.get('strategy')!r}")
    return errs


def load_runs(raw_dir) -> List[Dict]:
    runs = []
    for p in sorted(Path(raw_dir).rglob("*.json")):
        r = json.loads(p.read_text())
        if r.get("kind") == "run":
            runs.append(r)
    return runs


def build_configs(args) -> List[Dict]:
    n_base = args.n_base
    configs = []
    for exp in args.experiment:
        for n in n_base:
            if exp == "e3":
                strategies = [("e3_random", None), ("e3_zone", None)]
            else:
                strategies = []
                for s in args.strategies:
                    if s == "skeleton":
                        strategies += [(s, f) for f in args.skeleton_fractions]
                    else:
                        strategies.append((s, None))
            for (s, f), seed in itertools.product(strategies, args.seeds):
                configs.append(
                    make_config(
                        experiment=exp,
                        dataset=args.dataset,
                        n_base=None if n == "full" else int(n),
                        strategy=s,
                        seed=seed,
                        ef=args.ef,
                        threads=args.threads if exp != "e2" else args.e2_threads,
                        skeleton_fraction=0.05 if f is None else f,
                        n_queries=args.n_queries,
                        repeats=args.repeats,
                        concurrent_jobs=max(args.concurrent_jobs, args.jobs - 1),
                    )
                )
    return configs


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser()
    p.add_argument(
        "--dataset",
        default="synthetic",
        help="sift-128-euclidean, glove-100-angular or synthetic",
    )
    p.add_argument(
        "--n-base", nargs="+", default=["10000"], help="10000, 100000, 1000000 or full"
    )
    p.add_argument("--strategies", nargs="+", default=["zone", "random", "skeleton"])
    p.add_argument(
        "--skeleton-fractions", nargs="+", type=float, default=[0.01, 0.05, 0.10]
    )
    p.add_argument("--seeds", nargs="+", type=int, default=[0])
    p.add_argument("--ef", nargs="+", type=int, default=[16, 32, 64, 128, 256, 512])
    p.add_argument("--threads", type=int, default=1, help="build threads (e1, e3)")
    p.add_argument("--e2-threads", type=int, default=8, help="build threads for e2")
    p.add_argument("--experiment", nargs="+", default=["e1"], choices=EXPERIMENTS)
    p.add_argument("--n-queries", type=int, default=None)
    p.add_argument("--repeats", type=int, default=5)
    p.add_argument(
        "--jobs", type=int, default=1, help="parallel runs (contaminates timings)"
    )
    p.add_argument(
        "--concurrent-jobs",
        type=int,
        default=0,
        help="other study processes running outside this launcher",
    )
    p.add_argument(
        "--inprocess", action="store_true", help="do not isolate runs in subprocesses"
    )
    p.add_argument("--out-dir", default="results/latest-novelty/raw")
    p.add_argument(
        "--aggregate", metavar="RAW_DIR", help="aggregate runs in RAW_DIR and exit"
    )
    p.add_argument("--out", help="summary path for --aggregate")
    p.add_argument(
        "--check", nargs="+", metavar="SUMMARY", help="validate summary files and exit"
    )
    return p


def _worker(cfg, out_dir):
    return str(execute(cfg, out_dir))


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    if args.check:
        bad = 0
        for f in args.check:
            errs = check_summary(json.loads(Path(f).read_text()))
            print(f"{f}: {'OK' if not errs else errs}")
            bad += bool(errs)
        return 1 if bad else 0
    if args.aggregate:
        summary = aggregate(load_runs(args.aggregate))
        errs = check_summary(summary)
        if errs:
            print(errs)
            return 1
        out = Path(args.out or Path(args.aggregate).parent / "study_summary.json")
        _atomic_write(out, summary)
        print(f"wrote {out} ({summary['n_runs']} runs)")
        return 0

    configs = build_configs(args)
    todo = [c for c in configs if not run_path(args.out_dir, c).exists()]
    print(f"{len(configs)} runs, {len(configs) - len(todo)} already done", flush=True)
    if args.inprocess:
        for c in todo:
            print("run", run_path(args.out_dir, c).name, flush=True)
            _worker(c, args.out_dir)
    else:
        ctx = multiprocessing.get_context("spawn")
        with ProcessPoolExecutor(
            max_workers=args.jobs, mp_context=ctx, max_tasks_per_child=1
        ) as ex:
            for c, path in zip(todo, ex.map(_worker, todo, [args.out_dir] * len(todo))):
                print("done", path, flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
