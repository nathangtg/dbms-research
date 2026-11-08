# Compression-first comparison at equal memory

Question: does ZGQ with PQ/RPQ codes (graph over codes, optional exact rerank) beat HNSW+PQ and IVF-PQ(R)
when every index is held to the same size?

**Answer: no. ZGQ compressed loses at equal memory in every size band at both 100K and 1M.**
It beats FAISS HNSW+PQ at equal bytes only because of residual coding, and IVF-PQ(R) and HNSW+SQ8
beat it by a wide margin at the same or lower size.

## Method

* Memory is the analytic bytes/vector model in `zgq/memory_model.py` (RSS is unreliable on macOS).
  Each index also has a cross-check column: real `faiss.serialize_index` length, hnswlib `save_index`
  file size, or the actual nbytes of the ZGQ compressed arrays. Ratios are 1.000 to 1.002 for all 21
  index variants at both scales (see "Model cross-check" in the `equal_memory_*.md` files).
* "Everything counted": graph links, codes, codebooks, ids, centroids, ZGQ's zone/codebook cross table.
  ZGQ-C codes-only keeps no raw vectors (`drop_full_vectors=True` semantics). `+rerank50` variants
  keep the raw vectors (`n*d*4` bytes), so they sit in the full-vector band.
* ZGQ-C search: level-0 links of the hnswlib graph, ADC scoring from RPQ codes against each candidate's own
  zone centroid (`zgq/core/compressed.py`), entry points of the 4 nearest zones, graph built with random
  insertion order. Parameters were not tuned.
* Data, queries and ground truth are the harness defaults (clustered Gaussian, dim 128, seed 42, 1000
  queries, k=10, exact L2 ground truth). Search is single threaded, 1 warm-up and 5 repeats per setting,
  median latency. Builds use 8 threads (`--build-threads 8`), so build times are not comparable with the
  single-threaded canonical results and graphs are not bit-reproducible.
* Bands (bytes/vector): A 60-120, B 120-200, C 200-300, ref >= 500. Index sizes cannot be matched exactly,
  so each band lists every index that falls inside it with its own byte count.
* Baselines: hnswlib HNSW-M16 (memory reference), FAISS HNSWPQ (M 8/16, pq_m 16/32), HNSWSQ8 (M 8/16),
  IVFPQ (pq_m 16/32/64), IVFPQR (pq_m 16 refine 64, 32/32, 32/64, `k_factor` 4). The ZGQ graph degree
  alone (2M links of 4 bytes) is 64 or 128 B/vector, so 20 to 40 B/vector budgets are only reachable by IVF-PQ.

## Headline rows (max recall in the sweep, median latency at that point)

100K (full tables: `equal_memory_100k.md`)

| band | index | B/vec | max recall % | ms |
|---|---|---:|---:|---:|
| A | IVFPQR-pq32-r64 | 109.9 | 92.9 | 0.148 |
| A | IVFPQ-pq64 | 76.5 | 87.3 | 0.302 |
| A | ZGQ-C-PQ-M8-pq16 | 82.9 | 5.5 | 0.022 |
| B | ZGQ-C-RPQ-M8-pq16 | 135.0 | 32.1 | 0.144 |
| B | HNSWPQ-M16-pq32 | 177.6 | 19.1 | 0.298 |
| C | HNSWSQ8-M8 | 208.6 | 90.0 | 0.317 |
| C | ZGQ-C-RPQ-M8-pq32 | 201.2 | 60.1 | 0.180 |
| ref | HNSW-M16 | 660.5 | 96.7 | 0.209 |
| ref | ZGQ-C-RPQ-M16-pq32+rerank50 | 777.2 | 95.7 | 0.216 |

1M (full tables: `equal_memory_1m.md`)

| band | index | B/vec | max recall % | ms |
|---|---|---:|---:|---:|
| A | IVFPQR-pq32-r64 | 105.3 | 85.7 | 0.479 |
| A | IVFPQ-pq64 | 73.2 | 82.7 | 0.971 |
| A | ZGQ-C-RPQ-M8-pq16 | 99.4 | 12.2 | 0.287 |
| B | ZGQ-C-RPQ-M16-pq32 | 196.2 | 33.7 | 0.411 |
| B | HNSWPQ-M16-pq32 | 176.4 | 14.7 | 0.452 |
| C | HNSWSQ8-M8 | 208.6 | 80.3 | 0.478 |
| ref | HNSW-M16 | 660.5 | 91.3 | 0.314 |
| ref | ZGQ-C-RPQ-M16-pq32+rerank50 | 708.2 | 66.8 | 0.420 |

## What the results say

1. **Equal memory: ZGQ compressed loses.** In band A IVFPQR/IVFPQ reach 83 to 93% where ZGQ-C reaches 3 to 12%.
   In band C HNSWSQ8 reaches 80 to 90% where ZGQ-C reaches 60% (100K), and ZGQ-C has no 1M point in that band.
   With raw vectors kept for rerank (full-vector band) ZGQ-C is 1 point below HNSW-M16 at 100K using more
   bytes (777 vs 661 B/vec) and 24 points below at 1M.
2. **Where ZGQ-C does beat a baseline**: against FAISS HNSW+PQ with a global (non-residual) PQ at similar bytes
   (100K band B: 32.1 to 32.2% vs 19.1%). The non-residual ZGQ-C-PQ variant matches HNSWPQ (19.0 vs 19.1%),
   so the gain comes from the zone-residual coding alone, not from the zone-guided graph.
3. **The code quantizer sets the recall ceiling.** At 100K the ZGQ-C codes-only curve flattens at the same
   recall as IVF-PQ with the same pq_m (pq16: 32.2% vs 33.7%; pq32: 60.7% vs 61.5%). At 1M the graph over codes
   falls below that ceiling (pq16 12.2% vs IVFPQ 27.5%; pq32 33.7% vs 54.1%): beam search on noisy ADC
   distances loses accuracy as n grows. No tuning of entry points or `ef` beyond 512 was tried.
4. **Zone cross table cost.** Per-candidate-zone ADC needs `Z*pq_m*256*4` bytes (306 zones at 100K: about
   50 B/vector for pq16 and 100 for pq32; 1023 zones at 1M: about 17 and 34). It dominates
   ZGQ-C size at 100K. The model lists it under `zone_codebook_cross_table`.
5. **Current default `ZGQIndex` memory**: the hnswlib graph holds a vector copy and `self.vectors` holds another,
   so a default index is `zgq_default_bytes` (see `zgq/memory_model.py`), above HNSW-M16 by the second copy
   plus codes and norms. This is analytic only; no default-index serialized size was measured.

## ADC table check (`adc_check_100k.json`)

Hypothesis from the diagnostics: `ZGQIndex.search` builds its PQ distance table from the nearest zone's
centroid although codes are residuals against each vector's own zone centroid.
Measured on 200 queries at 100K (shortlist = top 2k by score, then exact rerank; recall@10 %):

| n_probe | candidates | exact scoring | nearest-zone table | per-candidate-zone ADC |
|---:|---:|---:|---:|---:|
| 1 | 10 | 66.5 | 66.5 | 66.5 |
| 4 | 40 | 84.0 | 60.1 | 60.8 |
| 16 | 160 | 93.3 | 45.6 | 48.8 |
| 64 | 640 | 99.9 | 43.4 | 48.1 |

The table is wrong for candidates outside the nearest zone, and correcting it gains 0.6 to 4.7 points at
n_probe >= 4. It does not explain the loss: both PQ shortlists (pq_m 16) collapse as the candidate set grows
because the quantizer cannot rank 20 survivors out of hundreds. Mean relative ADC error is 7 to 10% for the
wrong table and 6 to 9.7% for the correct one. `ZGQIndex.search` was left unchanged (changing it would change
default behavior); the correct scoring is used only in the compressed mode.

## Commands and runtimes (Apple M4, 10 cores, 16 GB, macOS)

Run from `v8/` with `PATH=/tmp/zgqvenv/bin:$PATH`.

```
python -m benchmarks.run_compression --scale 100k --out-dir results/2026-11-08-improvements/compression   # about 3.5 min, log run_100k.log
python -m benchmarks.run_compression --scale 100k --adc-check --out-dir results/2026-11-08-improvements/compression   # about 1 min
nohup python -m benchmarks.run_compression --scale 1m --out-dir results/2026-11-08-improvements/compression > .../run_1m.log &   # about 22 min
python -m benchmarks.run_compression --out-dir results/2026-11-08-improvements/compression --tables-only
```

Defaults: seed 42, k 10, 1000 queries, 5 repeats, 1 warm-up, 8 build threads, single-threaded search.
ZGQ-C 1M builds took 109 to 164 s each; 100K builds 11 to 21 s (see `build.time_s` in the summary).

## Files

* `compression_summary.json`: schema `zgq-compression/1` (not `zgq-bench/1`, so `results/*/summary.json`
  validation is untouched). Per index: analytic components, `crosscheck`, per-point recall/latency/QPS statistics.
* `raw/`: every repeat. `equal_memory_100k.md`, `equal_memory_1m.md`: band tables and the model cross-check.
* `adc_check_100k.json`, `adc_check_100k.log`: ADC check. `run_*.log`: run logs.

## Limitations

Synthetic data only; one machine; graph builds used 8 threads; untuned ZGQ-C search (fixed 4 entry zones,
no `ef` above 512); latency includes Python table construction for ZGQ-C (numpy, 256-query chunks) but IVF/HNSW
baselines are pure C++, so latency columns are not an algorithmic comparison. The `+rerank50` variants use
exact distances on raw vectors, so they need `n*d*4` extra bytes. Bands are coarse: indexes inside one band
differ by up to 2x in bytes.
