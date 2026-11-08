# ZGQ recall-improvement options vs matched HNSW-M16 (10K, 100K, 1M)

All data is **synthetic clustered Gaussian, dimension 128** (70% cluster queries, 30% uniform), with exact
brute-force L2 ground truth. **No real dataset (for example SIFT1M) was run.** One machine, one seed.

## Machine and protocol

| Item | Value |
|---|---|
| CPU / RAM | Apple M4 (10 cores), 16 GB, macOS |
| Seed | 42 |
| Queries | 1000, k = 10 |
| Repeats | 5 timed passes after 1 warm-up per setting, median reported |
| Threads | 1 (search and build) |
| Graph | hnswlib M = 16, ef_construction = 200 for HNSW-M16 and every ZGQ variant |
| Scales | 10K, 100K, 1M vectors |
| ZGQ degrees | only M = 16 (ZGQ-M24/M32 skipped to save time) |

Command (run from the repository root, about 80 minutes wall time in total on this machine, dominated by 1M builds and the
nprobe sweeps):

```
python -m benchmarks.run_benchmarks --scale 10k 100k 1m --improvements --zgq-m 16 \
    --out-dir results/2026-11-08-improvements/final > results/2026-11-08-improvements/final/run.log
python -m benchmarks.check_results results/2026-11-08-improvements/final/summary.json --canonical
python -m benchmarks.compare_improvements results/2026-11-08-improvements/final/summary.json \
    > results/2026-11-08-improvements/final/comparison.md
```

Files: `summary.json` (schema `zgq-bench/1`, validated), `raw/` (every repeat), `run.log`,
`comparison.md` (all tables below, generated from `summary.json` only; read it for every number).
The canonical `results/2026-11-08/` is untouched. An earlier `--improvements` run was interrupted and deleted;
this folder is a complete re-run.

## How the numbers are compared

* **Equal ef**: recall difference in percentage points and latency ratio at the same `ef` as HNSW-M16 (only
  for variants swept over `ef`; the `nprobe` and adaptive variants have no `ef` axis).
* **Equal recall**: latency of each curve at a target recall (80/90/95%) by linear interpolation between the
  neighbouring measured points. "not reached" means the curve tops out below the target. HNSW-M16 itself
  tops out at 94.2% (100K) and 87.0% (1M), so the 95% target (and 90% at 1M) has no HNSW reference.
* **Build ratio**: build seconds of the variant over HNSW-M16 (single process, 1 thread). Variants in one group
  share one graph build, so they report the same build time.
* **Memory**: analytic index bytes (`zgq.memory_model`, `analytic_bytes` in `summary.json`), ratio to HNSW-M16.
  RSS deltas are in `summary.json` but are unreliable on macOS (HNSW-M16 shows +129 MB at 1M against 660 MB
  analytic, IVFPQ shows a negative delta); they are used only for ordering. The default `ZGQ-M16` entry has no
  `analytic_bytes` field; its value is taken from `ZGQ-M16+random` (same stored components).

## Results vs matched HNSW-M16

Common facts (`comparison.md`): every ZGQ variant keeps the raw vectors twice (hnswlib copy plus
`self.vectors`) plus PQ codes, so analytic bytes are **1.81 to 1.84x HNSW-M16** at all scales
(1197 MB vs 660 MB at 1M). Memory is therefore **worse than HNSW-M16 for all four ideas except compression,
which is reported separately below**.

### Baseline: default ZGQ-M16 (zone-ordered insertion), the gap to close

| Scale | HNSW-M16 recall at ef 256 | ZGQ-M16 recall at ef 256 | Gap | ZGQ latency ratio | Build ratio |
|---|---:|---:|---:|---:|---:|
| 10K | 98.7% | 98.2% | -0.5 pt | 1.0x | 2.97x |
| 100K | 94.2% | 84.8% | -9.4 pt | 0.8x | 1.64x |
| 1M | 87.0% | 76.8% | -10.2 pt | 0.8x | 1.04x |

ZGQ is faster per query at the same `ef` only because its graph is worse and terminates sooner.

### Idea 1: overlapping zones (`+overlap2`, each vector stored in its two nearest zones)

Zone path (`zone_scan`), swept over `n_probe`; compared with the same search without overlap (`+zonescan`)
and with HNSW-M16.

* **Recall**: with exact rerank (`-nopq`), `n_probe = 1` recall rises from 22.8% to 38.5% at 10K with overlap.
  At 1M, overlap reaches 99.0% at its largest `n_probe` against 97.7% without overlap.
* **Latency**: much worse than HNSW-M16 at equal recall. At 90% recall: 3.5x (10K) and 18.2x (100K) HNSW-M16
  latency; at 1M 80% recall costs 23.4 ms against 0.140 ms for HNSW-M16 (166.8x), and overlap is slower than
  no overlap at 100K and 1M at every target recall (82.3x without at 1M; at 10K it is faster at 80% recall, 3.6x vs 4.4x).
* **Memory**: +4 B/vector for the replica ids, ratio 1.82 vs 1.81 at 1M. Negligible next to the 1.8x base.
* **Build time**: 1.05x HNSW-M16 at 1M (shared graph build plus replica assignment, 1.5 s over the base).
* **Verdict**: negative. It raises recall at a given `n_probe` but not at a given latency, and does not close
  the gap to HNSW-M16. The with-PQ-rerank variants top out at 16.6% (1M), 34.4% (100K) and 44.3% (10K).

### Idea 2: adaptive probing (`+adaptive`, probe zones within a ratio of the nearest zone distance)

* **Recall/latency**: with exact rerank it reaches 97.3% at 1M (26.9 ms) and 100% at 100K (1.63 ms), against
  HNSW-M16 maximum 87.0% and 94.2%. It is better than fixed `n_probe` (`+zonescan-nopq`: 97.7% at 43.1 ms at
  1M), roughly halving latency at 80% recall (5.18 ms vs 11.5 ms at 1M), but is still 37.0x HNSW-M16 latency at
  80% recall at 1M and 4.3x at 90% recall at 100K.
* **Memory**: same as the base ZGQ (1.81x at 1M). **Build time**: 1.04x at 1M.
* **Verdict**: the best of the zone-path options and the only one that reaches the 95% recall HNSW-M16 cannot,
  but at 4 to 37x HNSW-M16 latency at 80 to 90% recall. It does not beat HNSW-M16 on a latency-recall trade-off. The default
  PQ-rerank version caps at 17.9% (1M), 38.6% (100K), 43.5% (10K).

### Idea 3: single global graph with zone-aware entry (`+global-entry`)

Search the global graph, entering at the nearest `n_probe = 8` zone centroids' members.

* **Recall at equal ef (exact rerank, `-nopq`)**: +20.4 pt at ef 16 (100K), turning to -9.9 pt at ef 256, with
  latency 6.3x down to 1.4x HNSW-M16. At 1M: +20.5 pt at ef 16, -10.8 pt at ef 256 (5.5x to 1.3x).
  The recall ceiling matches the zone-ordered graph (76.2% at 1M vs 76.8% for default ZGQ), so entry points
  do not repair the bad graph. With random insertion (below) the ceiling matches HNSW-M16.
* **With PQ rerank (default)**: much worse than HNSW-M16: -50.2 pt at ef 256 at 100K, 31.7% maximum at 1M.
* **Equal recall**: slower than HNSW-M16 (3.5x at 80% recall at 100K), and 1M 80% not reached.
* **Memory**: 1.81x. **Build time**: 1.04x at 1M.
* **Verdict**: negative on its own. It helps only at very low `ef` and costs 1.3 to 6x latency.

### Idea 4: compression-first at equal memory

Separate harness, results in `../compression/` (see `../compression/README.md`). **Negative:** ZGQ compressed
loses at equal memory in every size band at 100K and 1M. Example at 1M: band A (60 to 120 B/vector) IVFPQR-pq32-r64
reaches 85.7% recall (105.3 B/vector, 0.479 ms) while the best ZGQ-C reaches 12.2% (99.4 B/vector). At 100K
band C, HNSWSQ8-M8 reaches 90.0% (208.6 B/vector) against ZGQ-C 60.1%. With raw vectors for rerank, ZGQ-C is
1 point below HNSW-M16 at 100K using 777 vs 661 B/vector, and 24 points below at 1M. Build times in that
harness used 8 threads and are not comparable with this folder.

### Extra finding: random insertion order closes the gap (`+random`)

Not one of the four ideas, but the cause found by `../diagnostics/` (zone-ordered insertion degrades
hnswlib's graph). Building the same graph with random insertion order:

| Scale | HNSW-M16 recall at ef 256 | ZGQ-M16+random | Difference | Latency ratio at ef 256 | Build ratio | Bytes ratio |
|---|---:|---:|---:|---:|---:|---:|
| 10K | 98.7% | 98.7% | +0.0 pt | 1.1x | 2.95x | 1.84 |
| 100K | 94.2% | 93.4% | -0.8 pt | 1.1x | 1.82x | 1.82 |
| 1M | 87.0% | 86.9% | -0.1 pt | 1.0x | 1.43x | 1.81 |

At equal recall (80%) it costs 1.2x (10K), 1.1x (100K) and 1.1x (1M) HNSW-M16 latency. So the matched-HNSW
recall gap is closed, but ZGQ with random insertion is a slower HNSW with 1.8x the memory, not a better one.
Random insertion is opt-in (`insertion_order="random"`); the default stays zone-ordered, which keeps
the canonical results reproducible. `+random+global-entry-nopq` (exact rerank) reaches +30.9 pt at ef 16 at 1M
but costs 6.6x latency there and is 1.6x slower even at ef 256 with a -1.0 pt recall.

## Limitations

Synthetic data only; one machine; one seed; single-threaded; ZGQ-M24/M32 not run; RSS not used for memory
claims; the zone-path variants use Python-level zone scans, so their latency is not an algorithmic bound; the
interpolated equal-recall latencies assume linear behaviour between sweep points; shared builds in a group
mean per-variant build times are not independent measurements.
