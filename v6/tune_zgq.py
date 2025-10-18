"""
ZGQ V6 Parameter Tuning Script

Sweeps a small grid of parameters (Z, n_probe, ef_search) on a medium-sized
dataset and reports recall/latency trade-offs to help choose defaults.

Usage (in venv):
  python tune_zgq.py --Z 20 30 --n-probe 3 5 --ef-search 40 50 \
    --N 50000 --Q 300 --d 128 --trials 1
"""

from __future__ import annotations

import argparse
import itertools
from typing import List, Dict, Any

import numpy as np

from benchmark_framework import ANNSBenchmark


def generate_clustered_dataset(N: int, d: int, n_clusters: int = 20):
    print(f"\nGenerating clustered dataset...")
    print(f"  N={N:,} vectors, d={d} dimensions, {n_clusters} clusters")
    vectors = []
    for _ in range(n_clusters):
        center = np.random.randn(d) * 3
        cluster_size = N // n_clusters
        cluster_points = center + np.random.randn(cluster_size, d) * 0.8
        vectors.append(cluster_points)
    vectors = np.vstack(vectors).astype(np.float32)
    indices = np.random.permutation(N)
    vectors = vectors[indices]
    print("  ✓ Dataset generated")
    return vectors


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--Z", nargs="+", type=int, default=[20, 30])
    parser.add_argument("--n-probe", dest="n_probe", nargs="+", type=int, default=[3, 5])
    parser.add_argument("--ef-search", dest="ef_search", nargs="+", type=int, default=[40, 50])
    parser.add_argument("--M", type=int, default=16)
    parser.add_argument("--ef-construction", type=int, default=200)
    parser.add_argument("--pq-m", type=int, default=16)
    parser.add_argument("--use-pq", action="store_true", default=True)
    parser.add_argument("--N", type=int, default=50000)
    parser.add_argument("--Q", type=int, default=300)
    parser.add_argument("--d", type=int, default=128)
    parser.add_argument("--trials", type=int, default=1)
    args = parser.parse_args()

    np.random.seed(42)
    N, Q, d = args.N, args.Q, args.d
    vectors = generate_clustered_dataset(N, d, n_clusters=20)
    # Queries: sample from dataset with slight noise
    q_idx = np.random.choice(N, Q, replace=False)
    queries = vectors[q_idx].copy()
    queries += np.random.randn(Q, d).astype(np.float32) * 0.1

    print("\nSetting up benchmark (computing ground truth)...")
    bench = ANNSBenchmark(
        dataset=vectors,
        queries=queries,
        k_values=[1, 5, 10],
        n_trials=args.trials,
        verbose=True,
    )

    grid = list(itertools.product(args.Z, args.n_probe, args.ef_search))
    results: List[Dict[str, Any]] = []

    print(f"\nSweeping {len(grid)} configs...")
    for i, (Z, n_probe, ef_search) in enumerate(grid, 1):
        print(f"\n[{i}/{len(grid)}] Z={Z}, n_probe={n_probe}, ef_search={ef_search}")
        r = bench.benchmark_zgq(
            n_zones=Z,
            M=args.M,
            ef_construction=args.ef_construction,
            ef_search=ef_search,
            n_probe=n_probe,
            pq_m=args.pq_m,
            use_pq=args.use_pq,
            name_suffix=f"Z{Z}-P{n_probe}-EF{ef_search}"
        )
        results.append({
            "Z": Z,
            "n_probe": n_probe,
            "ef_search": ef_search,
            "recall@10": r.recall_at_k.get(10, 0.0),
            "latency_ms": r.mean_latency,
            "qps": r.queries_per_second,
            "build_s": r.build_time,
            "mem_mb": r.memory_mb,
        })

    # Sort by recall desc then latency asc
    results.sort(key=lambda x: (-x["recall@10"], x["latency_ms"]))

    print("\n======================== TUNING RESULTS ========================")
    print("Z  n_probe  ef  |  R@10    Lat(ms)   QPS    Build(s)  Mem(MB)")
    for row in results:
        print(f"{row['Z']:>2}  {row['n_probe']:>7}  {row['ef_search']:>3}  |  "
              f"{row['recall@10']:>6.4f}  {row['latency_ms']:>7.3f}  "
              f"{row['qps']:>5.0f}  {row['build_s']:>8.2f}  {row['mem_mb']:>7.2f}")

    # Save JSON summary
    import json
    with open("tuning_results_medium.json", "w") as f:
        json.dump(results, f, indent=2)
    print("\nSaved: tuning_results_medium.json")


if __name__ == "__main__":
    main()
