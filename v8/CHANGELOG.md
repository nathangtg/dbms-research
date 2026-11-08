# Changelog

All notable changes to the ZGQ project are documented in this file. Entries carry no dates. They are
grouped into phases in the order they appear in the git history, and each one cites the commits (short
hashes) or repository files that back it. For the narrative and the status of every claim see
[docs/RESEARCH_LOG.md](docs/RESEARCH_LOG.md) and [docs/CLAIMS.md](docs/CLAIMS.md).

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

Phases 4 to 8 below are not part of a tagged release. Phase 3 is the 8.0.0 code base and phases 1 and 2
are the pre-8.0.0 history.

### Added

Phase 7, build-order study (`results/2026-11-09-novelty/README.md`):

- Opt-in insertion orders (`build_order` module, `insertion_order` option): `042b443`, tests `0fe6315`.
- SIFT1M and GloVe-100 loaders with cached download and checks: `743d41f`, tests `c2a4c41`.
- hnswlib file parser, graph statistics and instrumented search: `22811d8`, tests `e066d0d`.
- Study runner, report generator and make targets (`study-smoke`, `study-check`, `study-report`):
  `fbcf497`, `b090313`, tests `d116e6c`.
- Partial raw output of the interrupted study, committed as is: `8da48cf`, `e6ae65a`. The folder has no
  `summary.json`, was not aggregated or verified, and nothing in it is quoted in the docs.

Phase 6, v1 preprint archive:

- `docs/preprint/v1.md`, the published v1 preprint archived verbatim: `fdf2a72`. The claims guard exempts
  only that exact path: `bfa2ecd`.
- `h5py` declared for the real-dataset loaders: `ec2b954`.

Phase 5, recall-gap diagnostics and improvement options (`results/2026-11-08-improvements/`):

- `benchmarks/` diagnostics script and opt-in random insertion order: `2120e38`. Measured diagnostics D0 to
  D4 at 10K and 100K: `205b297` (`diagnostics/README.md`).
- Opt-in `ZGQConfig` options, defaults unchanged: `zone_replicas`, `adaptive_probe` (`min_probe`,
  `max_probe`, `probe_ratio`), `search_mode` (`graph`, `zone_scan`, `global_entry`), and
  `ZGQIndex.batch_zone_search`, with a golden test for the default path: `ff80de7`, documented in `5d7686c`.
- Opt-in compressed ZGQ search (`compressed_search`, `rerank_budget`, `drop_full_vectors`,
  `compressed_entry_zones`), `ZGQIndex.compressed_memory_bytes()` and the analytic size model
  `zgq.memory_model`: `ba6c3a5`.
- `benchmarks/run_compression.py`, an equal-memory comparison against FAISS HNSW-PQ/SQ8 and IVF-PQ/PQR:
  `bee9583`. Results: `81b6065`, `2adccc3` (`results/2026-11-08-improvements/compression/README.md`).
- `--improvements` flag in `benchmarks/run_benchmarks.py` (variants off by default, canonical results
  unchanged): `24fec5a`. `benchmarks/compare_improvements.py` builds tables from a `summary.json`: `78d759c`.
- Full 10K, 100K and 1M sweep against matched HNSW-M16: `36364e4` (`results/2026-11-08-improvements/final/`).
  Outcomes written up with traceable claims: `7a68875`.

Measured outcomes, all on synthetic data and none with a real dataset (see
`results/2026-11-08-improvements/final/README.md`):

- `insertion_order='random'` brings recall to within 0.8 point of HNSW-M16 (86.9% vs 87.0% at 1M, ef 256) at
  1.0x to 1.1x latency, 1.43x build time and 1.81x analytic bytes.
- Overlapping zones, adaptive probing and a global graph with zone-aware entry points do not beat HNSW-M16
  on recall at equal latency.
- Compressed ZGQ loses to IVF-PQ(R) and HNSW+SQ8 at equal memory.

### Changed

Phase 5:

- `ResidualProductQuantizer.encode` processes rows in chunks of 50,000 (identical codes, bounded temporary
  memory at 1M vectors): `bee9583`.
- The PQ shortlist in `ZGQIndex.search` uses the nearest zone's centroid for all candidates. A per-candidate-zone
  correction gains 0.6 to 4.7 recall points at `n_probe >= 4` and does not remove the shortlist loss
  (`results/2026-11-08-improvements/compression/README.md`), so the default path is unchanged.

Phase 4, v8 consolidation:

- v1 to v7 removed from the working tree and kept at git tag `pre-v8-cleanup`: `8444b40`.
- Single dependency source in `setup.py`, `sys.path` hacks removed: `ef67973`.
- Makefile targets, README quick start and contributing steps corrected: `284eb5f`.
- Reproducible benchmark harness with schema check, tests, smoke target and committed canonical results:
  `3882848` (`results/2026-11-08/`).
- Figures and tables are generated only from `results/<run>/summary.json`; a claims ledger was added and
  docs reconciled to measured results: `8fe518d` ([docs/CLAIMS.md](docs/CLAIMS.md)).
- Drafts moved to `docs/theory`, theory assumptions stated, M=8 methodology text fixed: `1a48ae1`.
  Untraceable theory numbers and the drafts were then dropped (kept at `pre-v8-cleanup`): `03b8609`.
  `alpha = 0.65` is labelled as assumed and a guard pattern was added: `bbdb449`.
- `THEORY.md` no longer predicts results or states a 15 to 20 percent RPQ error gain
  ([docs/THEORY.md](docs/THEORY.md)).
- Retired numbers are listed in [docs/CLAIMS.md](docs/CLAIMS.md) and blocked by `make claims-check`.

Phase 8, code cleanup (no behavior change intended):

- Narrating docstrings and comments dropped, dead helpers removed, control flow simplified in core, index,
  benchmarks and visualization scripts: `bc18a46`, `a7fe734`, `6626d12`, `ee022d0`.
- `batch_search` modes, save limits, units and build-time adjustments documented in the API reference:
  `e9f946c`.

### Fixed

- `ZGQIndex.batch_search(n_jobs>1, fast_mode=False)` aborted the process (nested numba parallel launch in the
  asymmetric distance kernel). A lock around the kernel launch in `zgq/core/quantization.py` prevents it; the
  v8 test suite covers it: `46bc14b`. The lock has no measurable effect on `batch_search` time
  (`docs/CLAIMS.md`, harness notes).

## [8.0.0]

Phase 3, v8 implementation. Version 8 is a rewrite of the earlier ZGQ code as a package under `v8/`.

### Added

- Package skeleton, versioning, requirements and initial implementation: `c8c2dfd`, `39584ed`, `782e398`.
- `ZGQIndex` and `ZGQConfig` with batch search and configuration options: `c5e1d66`.
- Adaptive Hierarchical Zones (multi-level zone partitioning): `87df652`.
- Zone-guided graph navigation (HNSW structure with zone-aware features): `d9fc363`.
- Residual Product Quantization: `5be6dfd`.
- Search algorithms module and core components: `079af4a`, `985b074`.
- Numba-optimized distance computations: `a2e3858`.
- Evaluation metrics (recall, precision, MRR, NDCG) and I/O utilities: `5062d8d`, `2a378bb`, `a506a67`.
- Zone-ordered insertion for the HNSW index: `d513d37`.
- Benchmark suite against HNSW: `6d91a31`. Quick start example: `a6afd86`.
- Documentation: API reference `da57dc6`, theory `b6809e3`, benchmarking methodology `30b0e7c`,
  README `4ecfd12`, CONTRIBUTING `e4ece76`, Makefile `441bb61`, LICENSE `52748df`.
- First version of this changelog: `7faad25`.

### Changed

- Configuration through a dataclass (`ZGQConfig`) instead of a dictionary: `c5e1d66`.

### Removed

- `FINAL_RESULTS.md` (`6e74c42`) was later deleted because its tables came from an unreproducible 100-query
  run. It was replaced by [docs/BENCHMARKS.md](docs/BENCHMARKS.md) (`8fe518d`).

### Performance

Measured values (canonical run `results/2026-11-08`, synthetic data, single thread; see
[docs/BENCHMARKS.md](docs/BENCHMARKS.md)):

- 10K vectors: ZGQ-M16 recall equals HNSW-M16 at equal `ef` (92.3% vs 92.4% at `ef=64`), about 5% slower per query.
- 100K and 1M vectors: ZGQ-M16 recall is below HNSW-M16 at every equal `ef` (100K `ef=64`: 78.9% vs 86.1%;
  1M: 68.1% vs 77.6%).
- Build time is 1.2x to 2.7x slower than HNSW-M16.
- Memory use is much higher than HNSW (RSS delta ordering only; macOS measurements).
- No recall, latency, memory or build-time improvement over earlier versions is claimed. Earlier versions are
  not part of the canonical run.

Earlier versions of this file cited a v7 recall collapse at 100K, a v7 speed advantage over HNSW at 10K and a
list of projected gains. They came from scripts no longer in the tree or were targets that were never
measured, so they are not restated. Earlier "Fixed" entries about zone pruning, graph connectivity at zone
boundaries and memory leaks had no committed test or measurement and were removed
(see the removed rows in [docs/CLAIMS.md](docs/CLAIMS.md)).

## Pre-8.0.0 history

Source for these phases is git history only. The code is available at tag `pre-v8-cleanup`.

### Phase 2: V7 and the optimization plans

- Adaptive probing and HNSW optimizations in the ZGQ index: `56d966e`. V7 requirements: `60bb4e0`.
  V7 implementation details and validation results: `92fb566`.
- Optimization suite and plans: `0dda932`, `6d1190e`, `ddef848`, `64f947b`, `0354f0c` (unified graph approach).
- Performance regression analysis and critical findings at scale: `6dac7e2`, `89a88b0`, `ce86880`.
- Quick start guide, automated visualization and benchmark documentation: `3a63144`, `feaff58`, `62fe220`,
  README `3513677`, requirements `e285957`.
- Ultra-fast search and visualization tools: `56dccaf`, `8ae0e12`.
- Benchmark summaries and comparisons of ZGQ, HNSW, IVF and IVF+PQ: `b6b67f6`, `c563e43`, `cb1f0c7`,
  `8c37f9a`, `563a1e2`, `7767c40`.
- Research methodology, mathematical foundations and proof-of-concept documents: `7135ce4`, `59a65c0`, `e9b8775`.
- The numbers in these documents were not carried into v8. Where they cannot be reproduced from committed
  results they are not restated.

### Phase 1: V5 and V6

- Initial ZGQ V5 implementation and benchmark framework: `b837268`.
- V6 modules: zonal partitioning (`fe01324`), per-zone HNSW graphs (`86a73cd`), product quantization
  (`dc232d5`), distance metrics (`5fec763`), aggregation and re-ranking (`d4b32c7`), baseline algorithms
  (`41d8a1b`), benchmarking framework (`bbfcdca`), visualization (`4867f62`), main `ZGQIndex` (`9cda852`).
- V6 documentation: architecture `7307a43`, offline indexing `cced271`, online search `f73e8d3`, graph
  construction `9bc31c1`, distances `d9349fd`, product quantization `15fd39e`, tuning guide `e42a7dc`,
  project summary `0dbde8b`, README `95d268c`.
- Demonstration workflow `2a0baa2` and version comparison tool `782e08b`.

### Paper and preprint sources

- LaTeX sources for the paper and a proofing document, figures and drafts were committed in parallel with the
  code (for example `cdcfbb7`, `2b7da9b`, `12bddbf`, `0dba08a`).
  The paper's retired numbers are listed in [docs/CLAIMS.md](docs/CLAIMS.md); `latex/paper.pdf` was not
  regenerated.
