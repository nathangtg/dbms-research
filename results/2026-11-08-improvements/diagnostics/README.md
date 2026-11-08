# ZGQ recall-gap diagnostics

Measurement-first diagnostics for why ZGQ-M16 trails HNSW-M16 (recorded on
`results/2026-11-08`). Nothing here is a canonical benchmark `summary.json`.
All files are named diagnostic JSON. All `recall` values are **percent**
(recall@10 x 100, as returned by `zgq.search.compute_recall`).

## Setup

Same protocol as `benchmarks/run_benchmarks.py`: `generate_test_data` seed 42,
1000 queries, k=10, dim 128, M=16, ef_construction=200, single thread (ZGQ's
internal hnswlib forced to 1 thread), exact ground truth by brute force.
Latency is the median of 3 timed passes after 1 warm-up, whole-batch time divided
by query count (zone-path checks time one query at a time). Machine: Apple M4, 16 GB.
Scales: 10K and 100K. 1M was not run (the checks that matter were decisive at 100K).

## Commands

```
cd v8
python -m benchmarks.diagnostics --scale 10k  --checks d0 d1 d2 d3 --out-dir results/2026-11-08-improvements/diagnostics   # 19.8 s
python -m benchmarks.diagnostics --scale 100k --checks d0 d1 d2 d3 --out-dir results/2026-11-08-improvements/diagnostics   # 97.8 s
python -m benchmarks.diagnostics --scale 100k --checks d4        --out-dir results/2026-11-08-improvements/diagnostics   # 116.5 s
# D2/D3 were re-run after D2 gained the PQ-off column (final files come from these):
python -m benchmarks.diagnostics --scale 10k  --checks d2 d3 --out-dir results/2026-11-08-improvements/diagnostics   # 4.6 s
python -m benchmarks.diagnostics --scale 100k --checks d2 d3 --out-dir results/2026-11-08-improvements/diagnostics   # 6.4 s
```

Files: `d0_path_audit_*`, `d1_insertion_order_*`, `d2_nprobe_*`, `d3_oracle_zone_*`
(10k, 100k), `d4_zone_sweep_100k`, plus `run_*.log`. Zone count at default is 92
(10K) and 306 (100K).

## D0: path audit

`batch_search(fast_mode=True)`, the path the benchmark uses, called
`zones.select_zones` **0 times** over 50 queries (5 queries in zone mode: 5 calls).
It only queries the single global hnswlib graph and exact-reranks. Zones play no
role at search time on this path.

Recall at ef 64 / 256 (rf = rerank_factor):

| Scale | Index | ef 64 rf1 | ef 64 rf3 | ef 256 rf1 | ef 256 rf3 |
|---|---|---|---|---|---|
| 10K | plain hnswlib M16 | 92.41 | 92.40 | 98.73 | 98.72 |
| 10K | ZGQ graph | 92.30 | 92.29 | 98.21 | 98.20 |
| 100K | plain hnswlib M16 | 86.05 | 86.05 | 94.19 | 94.19 |
| 100K | ZGQ graph | 78.92 | 78.92 | 84.77 | 84.77 |

`batch_search(fast_mode=True)` returns exactly the ZGQ graph rf3 numbers. The
rerank factor changes recall by at most 0.01 points at ef >= 64 because hnswlib
already returns exact distances; it only helps at ef 16 where k*rf raises the
effective ef (100K: 59.75 rf1 to 70.52 rf3, plain 63.77 to 75.71).

## D1: insertion-order ablation (same graph class, same ids, seeds 42 and 43)

Recall (percent), rf1, zone-ordered (current) vs random-order insertion vs plain hnswlib:

| Scale | ef | zone-ordered (s42 / s43) | random-order (s42 / s43) | plain HNSW (s42 / s43) |
|---|---|---|---|---|
| 10K | 64 | 92.30 / 92.91 | 92.64 / 92.00 | 92.41 / 92.60 |
| 10K | 256 | 98.21 / 98.73 | 98.77 / 98.71 | 98.73 / 98.77 |
| 100K | 64 | 78.92 / 79.60 | 85.06 / 84.29 | 86.05 / 85.00 |
| 100K | 256 | 84.77 / 85.89 | 93.41 / 93.12 | 94.19 / 94.19 |

Latency at 100K ef 64: zone-ordered 0.061 ms, random-order 0.084 ms, plain 0.085 ms
(the zone-ordered graph is faster because it is a worse graph that terminates sooner).
Build time at 100K: 9.6 s zone-ordered vs 11.7 s random-order.

At 10K the seed-to-seed spread (up to 0.7 points) is as large as the gap,
so there is no measurable gap. At 100K zone-ordered insertion costs 6 to 7 points
at ef 64 and about 9 points at ef 256; random-order insertion recovers to within
about 1 point of plain HNSW (the plain and random-order seed spread is about 1 point
at ef 64). **Zone-ordered insertion is the main cause of the recall gap on the
benchmark path.** The ef 256 ceiling of ~85 means the zone-ordered graph cannot
be fixed by a larger ef.

## D2: zone path recall vs n_probe (`ZGQIndex.search`, ef_search 64)

Recall (percent) with the default PQ shortlist stage and with `use_pq_rerank=False`:

| n_probe | 10K PQ on | 10K PQ off | 100K PQ on | 100K PQ off |
|---|---|---|---|---|
| 1 | 59.56 | 59.56 | 63.09 | 63.09 |
| 4 | 55.10 | 70.23 | 57.30 | 80.19 |
| 16 | 50.88 | 98.12 | 41.73 | 85.56 |
| 64 | 45.45 | 99.99 | 40.23 | 93.87 |

With the PQ stage on, recall **falls** as n_probe grows (more candidates, but only
2k survive a PQ-distance shortlist computed with one residual table from the
nearest zone's centroid, which is wrong for candidates from other zones). With it
off, recall rises with n_probe as expected. The zone path still queries the global
graph for `k * n_probe` candidates (it does not restrict to zones), so n_probe
acts only as a candidate budget. The zone path is 40 to 60 points below the fast
path at its default n_probe. See `d2_nprobe_*.json` for every point and latency.

## D3: oracle zone and zone-boundary loss

Recall of exact search restricted to the zone of the true nearest neighbor
(`oracle_zone_recall`): 25.6% at 10K (92 zones), 69.5% at 100K (306 zones).
Zones are small relative to neighbor sets, so one zone is far from sufficient.

Mean fraction of the 10 true neighbors outside the top-n_probe zones (`select_zones`):

| n_probe | 10K | 100K |
|---|---|---|
| 1 | 0.772 | 0.406 |
| 4 | 0.327 | 0.183 |
| 8 | 0.127 | 0.171 |
| 16 | 0.015 | 0.090 |
| 64 | 0.000 | 0.000 |

At 100K, n_probe 16 still misses 9% of true neighbors, so any real zone-restricted
search needs about 32+ probes for under 3% loss. Boundary loss is real, but it does
not affect the benchmark path (D0).

## D4: zone-count sweep at 100K (fresh index per count)

Requested counts produce slightly fewer zones (50 gave 48, 316 gave 304, 1000 gave 982).

| zones (actual) | fast path ef64 recall | oracle-zone recall | outside @16 | zone path PQ-on @16 | PQ-off @16 |
|---|---|---|---|---|---|
| 48 | 85.49 | 0.922 | 0.070 | 24.0 | 92.5 |
| 96 | 83.58 | 0.871 | 0.070 | 27.5 | 91.1 |
| 304 | 80.14 | 0.705 | 0.070 | 43.8 | 88.2 |
| 624 | 80.13 | 0.608 | 0.070 | 43.1 | 89.6 |
| 982 | 81.60 | 0.555 | 0.071 | 43.7 | 89.4 |

Fewer zones give higher fast-path recall (fewer long zone-ordered runs) and a
higher oracle-zone recall. Zone count does not rescue the zone path with PQ on.

## Hypotheses

* **Zone boundaries: refuted for the benchmark path, real but secondary for the zone
  path.** The fast path never reads zones (D0). In the zone path, boundary loss is
  small at n_probe >= 32 (D3) and is dwarfed by the PQ shortlist loss (D2).
* **Small graphs losing long-range links: not supported.** ZGQ already uses one
  global graph, and rebuilding that same graph with random insertion order
  recovers recall to within about 1 point of plain HNSW (D1).
* **Fixed rerank budget (rf=3): refuted.** rf1 and rf3 differ by <= 0.01 points at
  ef >= 64 (D0), because the returned candidates already carry exact distances.
* **New cause found: non-random zone-ordered insertion.** It costs 6 to 9 points of
  recall at 100K and nothing measurable at 10K (D1), and explains why the gap grows
  with scale. A second finding: the PQ shortlist stage in `ZGQIndex.search` lowers
  recall as n_probe grows (D2).

Caveats: single dataset family (clustered Gaussian, dim 128), two seeds for D1,
recall differences under about 1 point are within seed noise.
