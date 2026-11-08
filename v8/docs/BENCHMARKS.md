# Benchmark Methodology and Results

This document describes how ZGQ v8 is benchmarked against HNSW (hnswlib) and FAISS IVF-PQ,
and reports the canonical results. Every number below is generated from
`results/2026-11-08/summary.json` by `visualization/make_tables.py` and
`visualization/generate_ieee_figures.py`. Nothing here is typed in by hand.
Each reported figure is tracked in [CLAIMS.md](CLAIMS.md).

## Protocol (as implemented in `benchmarks/run_benchmarks.py`)

| Item | Value |
|---|---|
| Data | Synthetic clustered Gaussian, dimension 128, seed 42 |
| Scales | 10K, 100K, 1M vectors |
| Queries | 1000 (70% near cluster centers, 30% uniform) |
| Ground truth | Exact brute-force L2, k=10 |
| Timing | 1 warm-up pass, 5 timed repeats, median and IQR reported |
| Latency | Wall time of one batch call divided by the number of queries |
| Threads | 1 (ZGQ's internal hnswlib threads are also forced to 1) |
| Build | Each index is built in a fresh process; build time and RSS delta recorded once |
| Machine | Apple M4, 10 cores, 16 GB, macOS, Python 3.13.7 (see `results/2026-11-08/README.md`) |
| Libraries | hnswlib 0.8.0, faiss-cpu 1.15.1, numba 0.68.1, numpy 2.5.4 |

### Comparison rules

* Only matched settings are compared. HNSW-M16 vs ZGQ-M16 at the same `ef` is the matched
  comparison. ZGQ-M24 and ZGQ-M32 have larger graphs and are labeled variants.
* ZGQ is searched with `batch_search(fast_mode=True)`, which uses `rerank_factor=3`. This raises
  the effective `ef` to at least 30 and reranks 30 candidates exactly. It explains why ZGQ shows
  higher recall than HNSW at `ef=16`. It is not a graph-quality effect, so `ef=16` is not
  used for conclusions.
* IVF-PQ uses `nlist = 2*sqrt(N)`, 32 sub-quantizers, 8 bits, no refinement, and is swept over `nprobe`.
* ZGQ runs at library defaults (`n_zones='auto'`, PQ on). No per-scale tuning was done.

## Results

Recall@10 (%) / latency (ms per query), median over 5 repeats. For IVF-PQ the column shows the
highest-recall `nprobe` instead of `ef`.

| Scale | Index | Role | Build (s) | ef=64: recall % / ms | ef=256: recall % / ms |
|---|---|---|---|---|---|
| 10K | HNSW-M16 | baseline | 0.9 | 92.4 / 0.046 | 98.7 / 0.099 |
| 10K | ZGQ-M16 | matched | 2.4 | 92.3 / 0.049 | 98.2 / 0.101 |
| 10K | ZGQ-M24 | variant | 2.4 | 94.7 / 0.051 | 98.8 / 0.102 |
| 10K | ZGQ-M32 | variant | 2.4 | 94.5 / 0.052 | 99.0 / 0.104 |
| 10K | IVFPQ | baseline | 2.8 | 64.4 / 0.084 (nprobe=64) | - |
| 100K | HNSW-M16 | baseline | 11.8 | 86.1 / 0.084 | 94.2 / 0.158 |
| 100K | ZGQ-M16 | matched | 19.0 | 78.9 / 0.065 | 84.8 / 0.122 |
| 100K | ZGQ-M24 | variant | 18.7 | 78.6 / 0.072 | 85.3 / 0.131 |
| 100K | ZGQ-M32 | variant | 18.6 | 80.1 / 0.070 | 86.2 / 0.128 |
| 100K | IVFPQ | baseline | 5.0 | 61.5 / 0.129 (nprobe=64) | - |
| 1M | HNSW-M16 | baseline | 172.8 | 77.6 / 0.124 | 87.0 / 0.226 |
| 1M | ZGQ-M16 | matched | 203.7 | 68.1 / 0.091 | 76.8 / 0.174 |
| 1M | ZGQ-M24 | variant | 226.1 | 70.2 / 0.098 | 76.8 / 0.183 |
| 1M | ZGQ-M32 | variant | 254.5 | 72.9 / 0.101 | 80.4 / 0.191 |
| 1M | IVFPQ | baseline | 30.3 | 56.2 / 0.421 (nprobe=64) | - |
The full sweep (`ef` 16 to 256, `nprobe` 1 to 256) is in `summary.json`. Tables are also
written to `tables/results_table.md` and `tables/results_table.tex`.

### Figures

Generated into `figures/` by `python -m visualization.generate_ieee_figures`:
`fig_recall_vs_qps`, `fig_latency_comparison`, `fig_pareto_frontier`, `fig_build_time`,
`fig_scaling_analysis`, `fig_ivf_pq_latency`, `fig_ivf_pq_tradeoff`.

### Reading the results

* At 10K, ZGQ-M16 matches HNSW-M16 recall at equal `ef` (92.3% vs 92.4% at `ef=64`) and is about
  2 to 7% slower per query (6.5% at `ef=64`: 0.049 ms vs 0.046 ms; about 2% at `ef=256`).
* At 100K and 1M, ZGQ-M16 is less accurate than HNSW-M16 at every equal `ef`
  (`ef=64`: 78.9% vs 86.1% at 100K, 68.1% vs 77.6% at 1M; `ef=256`: 84.8% vs 94.2% and 76.8% vs 87.0%).
  ZGQ is faster per query at the same `ef`, but HNSW at a smaller `ef` reaches higher recall at
  similar or lower latency (1M: HNSW `ef=64` is 77.6% at 0.124 ms; ZGQ-M16 `ef=256` is 76.8% at 0.174 ms).
  No ZGQ setting in this sweep dominates HNSW. ZGQ-M24 and M32 narrow the gap but do not close it.
* ZGQ builds 1.2x to 2.7x slower than HNSW-M16 at every scale (1M: 203.7 s vs 172.8 s for M16).
* ZGQ's index RSS delta is far larger than HNSW's (100K: about 820 MB vs 16 MB; 1M: about 2.3 to 2.7 GB
  vs 129 MB), because it keeps the raw vectors, PQ codes and zone structures next to the graph.
  macOS RSS deltas are unreliable in absolute terms (see below), so only this ordering is claimed.
* IVF-PQ recall saturates at 64.4% (10K), 61.5% (100K) and 56.2% (1M) regardless of `nprobe`. It
  is the fastest index at low recall and cannot reach the recall of the graph indexes without refinement.
  A latency ratio against IVF-PQ at 90% recall cannot be computed because IVF-PQ never reaches 90%.

### Assumptions behind the theory (Lemma 6.1 and Theorem 6.2)

The retired proof draft (`draft_papers/ZGQ_MATHEMATICAL_PROOF_DOCUMENT.md` at tag `pre-v8-cleanup`) predicts a query-time advantage
for ZGQ. The prediction holds only under two conditions, and the results above should be read with them in mind.

* **Lemma 6.1 (path reduction)** assumes spatial locality in insertion order: vectors inserted
  consecutively into a zone graph are spatially close, which shortens edges and reduces hops
  (`E[h_ZGQ] <= alpha * E[h_HNSW]`, alpha < 1). The assumption is not enforced or measured by the
  library. The benchmark data are synthetic with no particular insertion order, and no hop counts are
  recorded in `results/`, so alpha is unmeasured here. The hop counts quoted in the proof document
  (13.2 vs 9.8) come from an earlier run and are not reproducible.
* **Theorem 6.2 (query time)** requires `N > 10^4` and `alpha < 1 - sqrt(N) / (log N * ef)`. The 10K
  results sit at this boundary, not inside the regime, so they cannot test the theorem. The 100K and 1M
  results are inside the regime, and there ZGQ does not beat HNSW on the recall/latency frontier, so the
  measurements do not confirm the theorem's conclusion. Either alpha is not below the required bound
  on this data, or the cost model omits terms that matter in the implementation (the model is unvalidated).

## Compression-first comparison at equal memory

Analytic bytes/vector (`zgq/memory_model.py`, checked against saved/serialized sizes to within 0.2%)
at 100K and 1M, compressed ZGQ (graph over RPQ/PQ codes) against FAISS HNSW+PQ, HNSW+SQ8, IVF-PQ and
IVF-PQR. Full tables, commands, runtimes and limitations: `results/2026-11-08-improvements/compression/README.md`.

* In the 60 to 120 B/vector band IVFPQR-pq32-r64 reaches about 93% (100K) and 85.7% (1M) at 110 and 105 B/vector;
  the ZGQ compressed variants in that band reach 5.5% (100K) and 12.2% (1M).
* In the 200 to 300 B/vector band HNSWSQ8-M8 reaches 90.0% (100K) and 80.3% (1M) at 208.6 B/vector; ZGQ
  compressed reaches 60.1% at 201 B/vector (100K).
* ZGQ compressed beats FAISS HNSW+PQ at similar bytes only through residual coding (the non-residual ZGQ
  variant ties it: 19.0% vs 19.1% at 100K).
* ZGQ compressed with exact rerank on kept raw vectors: 95.7% at 777 B/vector (100K) vs HNSW-M16 96.7% at 661.
* Conclusion: at equal memory ZGQ compressed loses. Synthetic data, one machine, untuned, builds with 8 threads.

## Parameter guidance

| Goal | Suggestion |
|---|---|
| Lowest latency, recall secondary | `batch_search(fast_mode=True)` with `ef_search` 32 to 64 |
| Higher recall | Raise `ef_search` and `M`; at 100K+ recall still trails HNSW-M16 in these runs |
| Smallest memory | Not demonstrated. ZGQ currently needs far more memory than HNSW. |

## Running benchmarks

```bash
cd v8
make benchmark-smoke   # tiny-scale run plus schema check
make benchmark         # 10K, 1000 queries, 5 repeats, into results/latest
python -m benchmarks.run_benchmarks --scale 10k 100k 1m --out-dir results/$(date +%F)
python -m benchmarks.check_results results/<date>/summary.json --canonical
python -m visualization.generate_ieee_figures --results results/<date>/summary.json
python -m visualization.make_tables --results results/<date>/summary.json
```

Output layout: `results/<date>/summary.json` (schema `zgq-bench/1`) and
`results/<date>/raw/<scale>_<index>.json` with every per-repeat measurement.
`--scale` accepts `tiny`, `10k`, `100k`, `1m`. Data and ground truth are regenerated from
the seed on every run, so no data files are stored. The 1M scale took about 20 minutes in total.

## Limitations and threats to validity

1. **Synthetic data.** Clustered Gaussian data with 128 dimensions may not reflect SIFT, GIST or
   embedding distributions. No real dataset has been run. Rankings could change on real data.
2. **Query count.** 1000 queries per setting. Small recall differences (a point or less)
   should not be read as a winner without checking the IQR in `summary.json`.
3. **Single machine.** One Apple M4 laptop, 16 GB, macOS, one OS and one compiler toolchain.
   No repeat on Linux or x86.
4. **Python vs C++.** HNSW is hnswlib (C++). ZGQ is Python with numba and uses hnswlib internally
   for graph search. Build time and per-query overhead include interpreter costs that a C++
   implementation would not pay. Per-query latency is therefore not an algorithmic comparison.
5. **Batch timing.** Latency is a single-threaded batch call divided by the query count. It is not
   a per-request latency distribution, and p95/p99 were not measured.
6. **M variants.** ZGQ-M24/M32 use larger graphs than HNSW-M16 and are not matched comparisons.
   HNSW was not run at M24/M32, so no claim is made about whether larger M favors either.
7. **Memory.** RSS deltas on macOS are unreliable (compressed memory): HNSW at 1M shows 129 MB where
   about 640 MB is expected, and IVF-PQ at 1M shows a negative delta. No absolute index size is
   claimed. A Linux run or an analytic size computation is needed.
8. **Fast-mode artifact.** ZGQ's `fast_mode` reranking raises its effective `ef` at small `ef`,
   so low-`ef` comparisons favor ZGQ for reasons unrelated to graph quality.
9. **No tuning.** ZGQ ran at library defaults and no hyperparameter search was done for either side.
10. **IVF-PQ configuration.** The canonical run used one configuration (m=32, no refinement). The compression
    comparison above adds IVF-PQ with pq_m 16/32/64 and IVF-PQR; with refinement or a larger `m` IVF-PQ reaches higher recall.
