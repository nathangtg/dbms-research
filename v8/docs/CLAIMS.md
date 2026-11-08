# Claims ledger

Every performance number or comparative statement that appeared in the docs, with its source and
status. Sources are paths inside `results/2026-11-08/summary.json` (the canonical run) unless noted.
Improvement-option rows cite `results/2026-11-08-improvements/final/comparison.md`, a table generated from that folder's `summary.json` by `python -m benchmarks.compare_improvements`.

Statuses: **verified** (traced to the committed results), **corrected** (replaced by a measured value),
**removed** (deleted because it is unsupported or contradicted), **flagged** (not edited, or only
analytic; needs the authors' attention).

`make claims-check` fails if any retired number below reappears in a Markdown or LaTeX file
(this file is exempt). Retired patterns are defined in `scripts/check_claims.py`. The guard matches fixed patterns only, so it cannot catch reworded or new untraceable figures; review new numbers by hand.

## Claims retained, with sources

| Claim | Source | Status |
|---|---|---|
| 10K ef=64: HNSW-M16 92.4% / 0.046 ms; ZGQ-M16 92.3% / 0.049 ms | `10k.indexes[HNSW-M16, ZGQ-M16].points[ef=64]` | verified |
| 10K ef=256: HNSW-M16 98.7% / 0.099 ms; ZGQ-M16 98.2% / 0.101 ms | `10k.indexes.points[ef=256]` | verified |
| 100K ef=64: HNSW-M16 86.1% / 0.084 ms; ZGQ-M16 78.9% / 0.065 ms | `100k.indexes.points[ef=64]` | verified |
| 100K ef=256: HNSW-M16 94.2% / 0.158 ms; ZGQ-M16 84.8% / 0.122 ms | `100k.indexes.points[ef=256]` | verified |
| 1M ef=64: HNSW-M16 77.6% / 0.124 ms; ZGQ-M16 68.1% / 0.091 ms | `1m.indexes.points[ef=64]` | verified |
| 1M ef=256: HNSW-M16 87.0% / 0.226 ms; ZGQ-M16 76.8% / 0.174 ms | `1m.indexes.points[ef=256]` | verified |
| ZGQ is 7 to 10 recall points below HNSW-M16 at ef=64 and ef=256 for 100K and 1M | `differences of the values above (7.2, 9.4, 9.5, 10.2)` | verified |
| ZGQ is 20 to 30% faster per query at equal ef at 100K and 1M | `ratios 0.65/0.84, 0.122/0.158, 0.091/0.124, 0.174/0.226` | verified |
| Build times HNSW-M16 / ZGQ-M16: 0.9/2.4 s (10K), 11.8/19.0 s (100K), 172.8/203.7 s (1M) | `indexes[].build.time_s` | verified |
| ZGQ builds 1.2x to 2.7x slower than HNSW-M16 across all ZGQ variants | `build.time_s ratios (1M M16 1.18x; 10K 2.7x)` | verified |
| ZGQ RSS delta about 820 MB vs HNSW 16 MB at 100K; about 2.3 to 2.7 GB vs 129 MB at 1M | `indexes[].build.rss_delta_mb (ordering only, macOS)` | verified |
| IVF-PQ recall saturates at 64.4 / 61.5 / 56.2% | `IVFPQ points nprobe 64 and 256` | verified |
| IVF-PQ 10K nprobe=1: 16.0% recall at 0.013 ms | `10k.indexes[IVFPQ].points[nprobe=1]` | verified |
| Iso-recall 100K: ZGQ-M32 ef=256 86.2% / 0.128 ms / 7,837 QPS vs HNSW-M16 ef=64 86.1% / 0.084 ms / 11,954 QPS (ratio about 0.66) | `100k.indexes points` | verified |
| 100K ef=128 matched row: HNSW-M16 90.3% / 0.115 ms / 8,671 QPS; ZGQ-M16 81.9% / 0.087 ms / 11,553 QPS | `100k.indexes points[ef=128]` | verified |
| 100K ef=64 QPS: HNSW-M16 11,954; ZGQ-M16 15,403 | `100k.indexes points[ef=64].qps` | verified |
| 100K ef=256 QPS: HNSW-M16 6,326; ZGQ-M16 8,210 | `100k.indexes points[ef=256].qps` | verified |
| 100K ef=64 ZGQ-M24 78.6%, ZGQ-M32 80.1% | `100k.indexes points[ef=64]` | verified |
| ZGQ recall higher than HNSW at ef=16 is a fast_mode rerank_factor=3 artifact | `results/2026-11-08/README.md protocol; zgq/index.py batch_search` | verified (mechanism from code) |
| Compression at equal memory (100K/1M): IVFPQR-pq32-r64 92.9%/85.7% at 109.9/105.3 B/vec; ZGQ-C-PQ-M8-pq16 5.5% (100K), ZGQ-C-RPQ-M8-pq16 12.2% (1M) in the same band; HNSWSQ8-M8 90.0%/80.3% at 208.6 B/vec; ZGQ-C-RPQ-M8-pq32 60.1% at 201.2 B/vec (100K); ZGQ-C+rerank50 95.7% at 777.2 B/vec vs HNSW-M16 96.7% at 660.5 (100K) | `results/2026-11-08-improvements/compression/compression_summary.json` (non-canonical schema `zgq-compression/1`; analytic bytes/vector, 8 build threads) | verified |
| ADC shortlist check: nearest-zone table vs per-candidate-zone ADC, recall 43.4% vs 48.1% at n_probe 64, exact scoring 99.9% (100K, 200 queries) | `results/2026-11-08-improvements/compression/adc_check_100k.json` | verified |
| Machine: Apple M4, 10 cores, 16 GB, macOS, Python 3.13.7 | `summary.json environment` | verified |
| All improvement data is synthetic clustered Gaussian, dim 128, seed 42, 1000 queries, 5 repeats, 1 thread, Apple M4 16 GB; no real dataset (for example SIFT1M) was run; ZGQ-M24/M32 skipped | `results/2026-11-08-improvements/final/summary.json` (`environment`, `protocol`), `results/2026-11-08-improvements/final/README.md` | verified |
| Zone-ordered ZGQ-M16 recall gap to HNSW-M16 at ef=256: -0.5 pt (10K), -9.4 pt (100K), -10.2 pt (1M); ZGQ latency 1.0x / 0.8x / 0.8x of HNSW; build 2.97x / 1.64x / 1.04x | `results/2026-11-08-improvements/final/comparison.md` (equal-ef and build tables), from `results/2026-11-08-improvements/final/summary.json` | verified |
| Analytic index bytes of every ZGQ variant are 1.81 to 1.84x HNSW-M16 (1197 MB vs 660 MB at 1M); RSS used only as ordering (macOS) | `results/2026-11-08-improvements/final/summary.json` `indexes[].analytic_bytes`; ZGQ-M16 default entry uses the +random value (same components) | verified |
| Random insertion (`insertion_order="random"`) at ef=256: recall 98.7% / 93.4% / 86.9% (10K/100K/1M) vs HNSW-M16 98.7 / 94.2 / 87.0, i.e. within 0.8 pt; latency 1.1x / 1.1x / 1.0x; build 2.95x / 1.82x / 1.43x of HNSW-M16; at 80% recall 1.2x / 1.1x / 1.1x HNSW latency | `results/2026-11-08-improvements/final/comparison.md` (rows `ZGQ-M16+random`) | verified |
| Diagnostic: zone-ordered insertion is the main cause of the gap: 100K ef 256 recall 84.77 (zone order) vs 93.41 (random order) vs 94.19 (plain hnswlib); the benchmarked fast path calls `select_zones` 0 times | `results/2026-11-08-improvements/diagnostics/d0_path_audit_100k.json`, `d1_insertion_order_100k.json`, `diagnostics/README.md` | verified |
| Overlapping zones (`zone_replicas=2`, zone_scan, exact rerank): n_probe=1 recall 38.5% vs 22.8% without overlap (10K); at 1M 80% recall costs 23.4 ms (166.8x HNSW-M16 0.140 ms) vs 11.5 ms (82.3x) without overlap; max recall 99.0% at 125 ms; +4 B/vector; build 1.05x. Negative | `results/2026-11-08-improvements/final/comparison.md` (rows `ZGQ-M16+overlap2-nopq`, `+zonescan-nopq`) | verified |
| Overlap with the default PQ shortlist tops out at 44.3% / 34.4% / 16.6% recall (10K/100K/1M) | `results/2026-11-08-improvements/final/comparison.md` (rows `ZGQ-M16+overlap2`) | verified |
| Adaptive probing (exact rerank): 97.3% max recall at 26.9 ms (1M), 100% at 1.63 ms (100K); 80% recall at 5.18 ms vs 11.5 ms fixed n_probe (1M); 37.0x HNSW-M16 latency at 80% (1M), 4.3x at 90% (100K); build 1.04x. With PQ shortlist it caps at 17.9% / 38.6% / 43.5% (1M/100K/10K) | `results/2026-11-08-improvements/final/comparison.md` (rows `ZGQ-M16+adaptive-nopq`, `+adaptive`) | verified |
| Global graph with zone-aware entry (exact rerank): +20.4 pt at ef 16, -9.9 pt at ef 256 (100K) at 6.3x to 1.4x HNSW-M16 latency; +20.5 pt / -10.8 pt (1M) at 5.5x to 1.3x; with PQ shortlist -50.2 pt at ef 256 (100K) and 31.7% max recall (1M); with random insertion and exact rerank +30.9 pt at ef 16 and -1.0 pt at ef 256 (1M) at 6.6x / 1.6x latency | `results/2026-11-08-improvements/final/comparison.md` (rows `ZGQ-M16+global-entry*`, `ZGQ-M16+random+global-entry*`) | verified |
| Equal-recall (80/90/95%) latency of every variant vs HNSW-M16 by linear interpolation; HNSW-M16 max recall 94.2% (100K), 87.0% (1M) so 95% (and 90% at 1M) has no HNSW reference | `results/2026-11-08-improvements/final/comparison.md` (equal-recall tables) | verified |
| Compression-first idea: ZGQ compressed loses to IVF-PQ(R) and HNSW+SQ8 at equal memory in every band (restated in the final README) | see the compression rows above; `results/2026-11-08-improvements/final/README.md` idea 4 | verified |

## Claims removed, corrected or flagged

| Claim | Location | Reason or source of replacement | Status |
|---|---|---|---|
| 94.8% Recall@10 (vs 92.9% HNSW) | README.md (root), proofing.tex, paper.tex | Unreproducible 2024 run, not in results/ | removed |
| 22% latency reduction, 11.4 ms vs 14.6 ms | README.md (root), proofing.tex, paper.tex, draft_papers/ZGQ_MATHEMATICAL_PROOF_DOCUMENT.md | Unreproducible run; contradicted by 2026-11-08 (HNSW at lower ef is faster at equal recall) | removed |
| 28% QPS increase, 43,794 vs 34,190 QPS | README.md (root), proofing.tex, paper.tex | Unreproducible run; iso-recall ratio at 100K measured about 0.66 | removed |
| 13.6% throughput improvement | README.md (root), paper.tex | Unreproducible run | removed |
| 0.048 ms latency, 32% faster than HNSW | v8/README.md | Unreproducible; 10K measured 0.049 ms (ZGQ) vs 0.046 ms (HNSW) at ef=64 | corrected |
| 68.2% recall vs HNSW 64.6% | v8/README.md | Unreproducible; 1M measured 68.1% (ZGQ) vs 77.6% (HNSW) at ef=64 | corrected |
| Memory 1M ~580 MB (-5% vs HNSW ~610 MB) | v8/README.md | Contradicted: ZGQ RSS delta about 2.3 to 2.7 GB vs HNSW 129 MB at 1M; macOS deltas unreliable so no absolute figure is claimed | removed |
| 13.3x speedup over IVF-PQ (151.6 ms vs 11.4 ms at >90% recall) | paper.tex, draft_papers/ZGQ_MATHEMATICAL_PROOF_DOCUMENT.md | IVF-PQ never reaches 90% recall (max 64.4/61.5/56.2%); the ratio cannot be computed | removed |
| 3x speedup over IVF-PQ (34 ms vs 11.42 ms at 93% recall) | proofing.tex | Came from a hardcoded dataset and a PYTHON_FACTOR=350 fudge in the deleted compare_ivf_pq.py | removed |
| Memory: IVF-PQ 25 MB, ZGQ 49 MB, HNSW 61 MB | proofing.tex | Not in results; contradicted by RSS ordering (ZGQ >> HNSW) | removed |
| ZGQ wins 3 of 4 metrics (recall, QPS, latency) | proofing.tex, paper.tex | Contradicted at 100K and 1M | removed |
| ZGQ 93.2% vs HNSW 92.9% at 100K iso-recall | proofing.tex, paper.tex | Unreproducible; 100K ef=64 measured 78.9% vs 86.1% | removed |
| Build time 3.59 s vs 2.14 s (1.67x / 68% slower) | proofing.tex, paper.tex | Replaced by measured build times (19.0 s vs 11.8 s at 100K) | corrected |
| Scaling advantages grow with dataset size | proofing.tex, paper.tex | Contradicted: recall at ef=64 falls faster for ZGQ (fig_scaling_analysis) | removed |
| ZGQ uses less memory than HNSW; 19% less memory | paper.tex abstract/conclusion, draft_papers/pdf/current_formula_v1.tex | Contradicted by RSS deltas | removed |
| ZGQ reduces index construction time | paper.tex conclusion | Contradicted: 1.2x to 2.7x slower | removed |
| Expected improvements over v7 (recall 75%+, memory -20-30%, build -15%) | CHANGELOG.md | Targets, never measured | removed |
| Fixed: zone pruning, graph connectivity at zone boundaries, memory leaks | CHANGELOG.md | No test or measurement committed | removed |
| Fixed: v7 recall collapse 55% to 21% at 100K | CHANGELOG.md | v7 not in tree; not reproducible | removed |
| v7 35% faster than HNSW at 10K | CHANGELOG.md | v7 not in tree; not reproducible | flagged |
| FINAL_RESULTS.md tables (95.10% recall at 10K, 78.70% vs 77.60% at 100K, 100 queries) | v8/FINAL_RESULTS.md (deleted) | Unreproducible 100-query run; replaced by docs/BENCHMARKS.md | removed |
| Tuning table: ~33K / 30K / 20K QPS expected | v8/FINAL_RESULTS.md (deleted) | Unmeasured | removed |
| 4x speedup from SIMD, 2x from early termination | v8/README.md | No measurement committed | removed |
| RPQ reduces quantization error by 15-20% | v8/README.md, docs/THEORY.md (H3 and section 3), zgq/core/quantization.py docstring | No measurement committed. Removed from all four places; THEORY.md now labels it a hypothesis | removed |
| O(log N) zone selection | v8/README.md | Analytic only; not benchmarked. Pointed to THEORY.md | flagged |
| Empirical overhead <1% for N=1e5 | paper.tex Theorem 3 | Contradicted by RSS deltas; kept only as analytic estimate | corrected |
| gamma = 1.65, alpha = 0.65, beta in [0.2, 0.4] 'empirically measured' | paper.tex, proofing.tex | No measurement in results/; relabeled as assumed model parameters | corrected |
| Theory-vs-empirical match: 1.30 vs 1.28 | proofing.tex | Based on retired numbers; measured iso-recall ratio about 0.66, model unvalidated | corrected |
| Reduced M=8 for ZGQ graphs in the methodology | paper.tex local graph quantization and zone-guided graph sections | Did not match the benchmarked configuration (M=16/24/32). Text now states M=16 matched plus M=24/32 variants | corrected |
| Hardware: Intel i5-12500H, WSL2 | paper.tex | Canonical run is Apple M4, macOS; text updated | corrected |
| Evaluation capped at 100,000 vectors | paper.tex limitations | 1M was run | corrected |
| paper.pdf contents | latex/paper.pdf | Still shows retired numbers; not modified (policy). Rebuild from paper.tex outside this change | flagged |
| Drafts: 94.8/11.4/43,794/13.3x, 19% less memory, 0.048, 68.2, 0.071 ms, 64.6%, 35% faster queries, 20% memory reduction, 610 MB, alpha 0.74 etc. | draft_papers/ (ZGQ_MATHEMATICAL_PROOF_DOCUMENT.md, MATHEMATICAL_RESULTS_QUICK_REF.md, current_formula_v1 to v4.tex and PDFs) | Deleted. The drafts carried dozens of untraceable empirical tables; the theory that matters is in docs/THEORY.md. History is at git tag `pre-v8-cleanup`. Extra patterns for these numbers were added to `scripts/check_claims.py` | removed |
| Expected Results table (ZGQ v8 90-95% recall, 'Lower' latency, IVF-PQ 60-80%) | docs/THEORY.md 6.3 | No source; contradicted by the canonical run. Replaced by a pointer to BENCHMARKS.md | removed |
| alpha = 0.74 'path reduction factor'; 'empirical' alpha = 0.65 | paper.tex Theorem 4 and conclusion | Not measured; now labeled assumed | corrected |

## Stale artifacts not regenerated

* `latex/paper.pdf` still contains retired numbers because the PDF is not modified by this change (user constraint). Rebuild it from `latex/paper.tex` (`make paper` builds into `latex/build/`).
* `v8/docs/proofing.pdf` was rebuilt from `proofing.tex` and contains no retired number.
* `docs/theory/` (including its v4 PDF) was deleted; see the drafts row above.

## Harness notes

* `zgq/core/quantization.py` holds `_PARALLEL_KERNEL_LOCK` around the numba ADC kernel launch to avoid a numba workqueue abort when several threads launch the parallel kernel. It is a library change, listed in CHANGELOG. `batch_search` on 20K x 64-d, 2000 queries: best of 3 is 17.6 ms (n_jobs=1) and 17.7 ms (n_jobs=4) with the lock, 17.5 ms and 17.3 ms without it, so the lock has no measurable effect.

## Known caveats on retained numbers

* Memory figures are RSS deltas on macOS and are not absolute index sizes (HNSW 1M shows 129 MB where about 640 MB is expected, IVF-PQ 1M is negative).
* ZGQ low-`ef` recall is inflated by `fast_mode` reranking; conclusions use `ef` of 64 and above.
* HNSW is hnswlib (C++), ZGQ is Python with numba.
