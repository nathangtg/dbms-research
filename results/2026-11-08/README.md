# Canonical benchmark results, 2026-11-08

Produced by:

```bash
cd v8
python -m benchmarks.run_benchmarks --scale 10k 100k 1m --out-dir results/2026-11-08
python -m benchmarks.check_results results/2026-11-08/summary.json --canonical
```

Path convention for consumers (for example `visualization/generate_ieee_figures.py`):
`v8/results/<YYYY-MM-DD>/summary.json` (schema `zgq-bench/1`, validated by
`benchmarks/check_results.py`) plus per-repeat raw records in
`v8/results/<YYYY-MM-DD>/raw/<scale>_<index label>.json`.

## Machine

| Item | Value |
|---|---|
| CPU | Apple M4, 10 cores (run single-threaded, `--threads 1`) |
| RAM | 16 GB |
| OS | macOS (arm64) |
| Python | 3.13.7 |
| numpy / scipy / scikit-learn / numba | 2.5.4 / 1.18.1 / 1.9.1 / 0.68.1 |
| hnswlib / faiss-cpu | 0.8.0 / 1.15.1 |
| zgq | 8.0.0 |
| Git commit | `8444b40` plus uncommitted harness changes (`git_dirty: true` in summary.json). The harness was committed unchanged in `3882848`; see "Provenance" |

## Protocol

Seed 42; synthetic clustered Gaussian data, dim 128, k=10; 1000 queries (70% near
cluster centers, 30% uniform); exact brute-force ground truth; one warm-up pass and
5 timed repeats per search setting, reported as median and IQR over repeats.
Latency is wall time of one batch call divided by the number of queries.
Build time and RSS delta are recorded once per index. Each index is built in a fresh
process. HNSW (M=16) vs ZGQ (M=16) is the matched comparison. ZGQ M=24 and M=32 have
larger graphs and are labeled variants. ZGQ is searched with
`batch_search(fast_mode=True)`, which uses `rerank_factor=3`: it raises the effective
`ef` to at least 30 and reranks 30 candidates exactly. This is why ZGQ shows higher
recall than HNSW at `ef=16`; it is not a graph-quality effect.
IVF-PQ: `nlist = 2*sqrt(N)`, 32 sub-quantizers, 8 bits, no refinement.

## Selected results (median over 5 repeats; latency in ms per query)

| Scale | Index | Build (s) | ef=64 recall % / latency | ef=256 recall % / latency |
|---|---|---|---|---|
| 10K | HNSW-M16 | 0.9 | 92.4 / 0.046 | 98.7 / 0.099 |
| 10K | ZGQ-M16 | 2.4 | 92.3 / 0.049 | 98.2 / 0.101 |
| 10K | ZGQ-M32 | 2.4 | 94.5 / 0.052 | 99.0 / 0.104 |
| 100K | HNSW-M16 | 11.8 | 86.1 / 0.084 | 94.2 / 0.158 |
| 100K | ZGQ-M16 | 19.0 | 78.9 / 0.065 | 84.8 / 0.122 |
| 100K | ZGQ-M32 | 18.6 | 80.1 / 0.070 | 86.2 / 0.128 |
| 1M | HNSW-M16 | 172.8 | 77.6 / 0.124 | 87.0 / 0.226 |
| 1M | ZGQ-M16 | 203.7 | 68.1 / 0.091 | 76.8 / 0.174 |
| 1M | ZGQ-M32 | 254.5 | 72.9 / 0.101 | 80.4 / 0.191 |

IVF-PQ recall saturates at 64.4% (10K), 61.5% (100K) and 56.2% (1M) regardless of
`nprobe`, because PQ without refinement caps accuracy on this data.
Full sweeps are in `summary.json`.

## Honest reading

* At 10K, ZGQ-M16 matches HNSW-M16 recall at equal `ef` and is about 5% slower per query.
* At 100K and 1M, ZGQ is less accurate than HNSW at every matched `ef`, by 7 to 10 recall
  points at `ef=64` (100K, 1M) and 9 to 10 points at `ef=256`. ZGQ is faster per query at the same `ef`,
  but HNSW at a smaller `ef` reaches higher recall at similar or lower latency
  (for example 1M: HNSW `ef=64` is 77.6% at 0.124 ms, while ZGQ-M16 `ef=256` is 76.8% at 0.174 ms).
  No ZGQ setting in this sweep dominates HNSW. Larger ZGQ M narrows but does not close the gap.
* ZGQ builds are slower than HNSW at every scale (1.2x to 2.7x).
* ZGQ index RSS delta is far larger than HNSW (100K: about 820 MB vs 16 MB; 1M: about
  2.3 to 2.7 GB vs 129 MB), since ZGQ keeps the raw vectors, the PQ codes and zone
  structures in addition to the graph.

## Caveats

* RSS deltas on macOS are unreliable in absolute terms. macOS compresses idle pages,
  so the HNSW delta (129 MB at 1M) is far below the roughly 640 MB the data structure
  must occupy, and the IVF-PQ delta at 1M is negative (-488 MB). Only the ZGQ vs HNSW
  ordering and rough magnitude are usable. Do not quote absolute index sizes from
  these numbers; a Linux re-run or an analytic size would be needed for a memory claim.
* All 1M numbers come from a single 16 GB machine, with 3 to 4 minute builds per index.
* ZGQ was run at library defaults (`n_zones='auto'`, PQ on); no per-scale tuning was done.
* Latencies are single-threaded batch calls, not per-request latencies.

## Provenance

The run was made from a dirty tree: `summary.json` records base commit `8444b40`, but the harness
did not exist there. The run was not repeated (about 20 minutes) because the code that produced it
is identified exactly instead. All of the following files were last modified before the run finished
(15:49) and are unchanged between commit `3882848` and the commit that added this section, apart from
a docstring edit in `quantization.py`. Git blob ids of the code that ran:

Blob ids at `3882848` (check with `git ls-tree -r 3882848 -- v8/benchmarks v8/zgq`):

```
benchmarks/__init__.py        e25a2c3285b33a8442336d34dde17442995707e1
benchmarks/check_results.py   81be5c9faf85ff022e9728c78272ab17f12560fe
benchmarks/run_benchmarks.py  15ec74bdd70c2a6db0ebd38078c6807c2c7af1b4
zgq/core/distances.py         58319ee34859d009aac47cd54ef712e8055e2ac7
zgq/core/graph.py             51b2b4c175b6730e699e9cc47364e49fbc971e9b
zgq/core/quantization.py      11e448b49a89c2b78c0db874c750bf2e31fb2730
zgq/core/zones.py             38a0f122bf2d2b93e61c33ee29ea658bdcfb42a9
```

To reproduce the code state, check out `3882848`.
