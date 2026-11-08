# Changelog

All notable changes to the ZGQ project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- Opt-in `ZGQConfig` options, defaults unchanged: `insertion_order`, `zone_replicas`, `adaptive_probe` (`min_probe`, `max_probe`, `probe_ratio`), `search_mode` ('graph', 'zone_scan', 'global_entry'); `ZGQIndex.batch_zone_search`. Persisted by save/load. No recall results claimed yet.

### Changed

- Removed v1 to v7 from the working tree (kept at git tag `pre-v8-cleanup`).
- Reproducible benchmark harness with schema check, tests, smoke target and committed canonical results.
- Figures and tables are generated only from `results/<date>/summary.json`.
- Added a lock around the numba asymmetric-distance kernel launch in `zgq/core/quantization.py` (avoids a numba workqueue abort when called from several threads; no measurable change in `batch_search` time for `n_jobs` 1 vs 4).
- Deleted `docs/theory/` drafts (untraceable empirical tables; kept at tag `pre-v8-cleanup`). THEORY.md no longer predicts results or states a 15-20% RPQ error gain.
- Docs reconciled to measured results; retired numbers are listed in `docs/CLAIMS.md` and blocked by `make claims-check`.

## [8.0.0] - 2024-12-04

### Major Release: Complete Algorithm Redesign

This release represents a major overhaul of the ZGQ algorithm, addressing the scaling limitations discovered in v7 and introducing several new architectural innovations.

### Added

- **Adaptive Hierarchical Zones (AHZ)**: Multi-level zone partitioning that scales adaptively with dataset size
  - Automatic zone count calculation: `n_zones = sqrt(n) * zone_factor`
  - 3-level hierarchy for improved search pruning
  - Zone boundary caching for faster membership tests
  
- **Zone-Guided Graph Navigation**: HNSW-style graph with zone-aware search
  - Priority queue considers zone information for edge selection
  - Entry point caching per zone
  - Improved connectivity through zone-aware link selection
  
- **Residual Product Quantization (RPQ)**: Enhanced quantization encoding residuals from zone centroids
  - Better compression ratios for clustered data
  - Reduced quantization error within zones
  - Configurable subquantizer parameters
  
- **Multi-probe Search Strategy**: Search multiple promising zones in parallel
  - Configurable probe count (n_probe parameter)
  - Zone prefetching for reduced latency
  - Dynamic candidate aggregation
  
- **SIMD-Optimized Distance Computations**: Numba JIT-compiled distance functions
  - Vectorized L2, inner product, and cosine distances
  - Batch distance computation for efficiency
  - Cache-friendly memory access patterns
  
- **Comprehensive Benchmark Suite**: Full evaluation framework
  - Comparison against HNSW and FAISS baselines
  - Recall-latency curve generation
  - Scaling analysis across dataset sizes
  
- **IEEE Publication-Ready Documentation**
  - Complete theoretical foundation
  - Algorithm complexity analysis
  - API reference documentation

### Changed

- Complete rewrite of index structure for better modularity
- Configuration via dataclass instead of dictionary
- Improved memory layout for better cache utilization
- Enhanced graph construction algorithm

### Fixed

- `ZGQIndex.batch_search(n_jobs>1, fast_mode=False)` aborted the process (nested numba parallel
  launch in the asymmetric distance kernel). A lock in `quantization.py` prevents it; covered by tests.

The v7 recall collapse at 100K (55.1% to 21.2%) and the v7 "35% faster than HNSW at 10K" figures that
earlier versions of this file cited came from v7 scripts that are no longer in the tree. They are not
reproducible from committed results and are not restated here. Earlier "Fixed" entries about zone pruning,
graph connectivity at zone boundaries and memory leaks had no committed test or measurement and were removed.

### Removed

- Deprecated single-level zone partitioning
- Legacy HNSW wrapper (replaced with zone-guided graph)
- Unused visualization dependencies from core package

### Performance

Measured values (canonical run `results/2026-11-08`, synthetic data, single thread; see
[docs/BENCHMARKS.md](docs/BENCHMARKS.md)):

- 10K vectors: ZGQ-M16 recall equals HNSW-M16 at equal `ef` (92.3% vs 92.4% at `ef=64`), about 5% slower per query.
- 100K and 1M vectors: ZGQ-M16 recall is below HNSW-M16 at every equal `ef` (100K `ef=64`: 78.9% vs 86.1%; 1M: 68.1% vs 77.6%).
- Build time is 1.2x to 2.7x slower than HNSW-M16.
- Memory use is much higher than HNSW (RSS delta ordering only; macOS measurements).
- No recall, latency, memory or build-time improvement over v7 is claimed. v7 is not part of the canonical run.

## [7.0.0] - 2024-11-XX

### Previous Version Summary

V7 introduced the unified HNSW approach with zone metadata but suffered from scaling issues:

- Reported (not reproducible from committed results) faster search than HNSW at 10K and a recall drop at 100K
- Fixed zone count (100 zones) didn't adapt to dataset size

### Known Issues

- Zone partitioning used fixed count regardless of data size
- Unified HNSW approach didn't leverage zone structure during search
- No hierarchical zone organization

## [6.0.0] - 2024-XX-XX

### Earlier Versions

Previous versions (v1-v6) explored various approaches:

- v1-v3: Initial zone partitioning concepts
- v4: Product quantization integration
- v5: HNSW graph experiments
- v6: Aggregation and reranking strategies

These versions established the theoretical foundation that v8 builds upon.

---

## Migration Guide: v7 → v8

### Breaking Changes

1. **Configuration API changed**:
   ```python
   # v7
   index = ZGQIndex(n_zones=100, M=16)
   
   # v8
   from zgq import ZGQIndex, ZGQConfig
   config = ZGQConfig(zone_factor=1.5, M=16)
   index = ZGQIndex(config)
   ```

2. **Search method signature**:
   ```python
   # v7
   results = index.search(query, k=10, ef=64)
   
   # v8
   results = index.search(query, k=10, n_probe=8, ef_search=64)
   ```

3. **Index persistence format changed** - v7 indices must be rebuilt

### New Features to Leverage

1. Enable multi-probe search for better recall:
   ```python
   config = ZGQConfig(n_probe=8)  # Search 8 zones
   ```

2. Use RPQ for memory efficiency:
   ```python
   config = ZGQConfig(use_pq=True, pq_n_subquantizers=8)
   ```

3. Batch search for throughput:
   ```python
   indices, distances = index.batch_search(queries, k=10)
   ```

## Roadmap

### Planned for v8.1

- [ ] GPU acceleration via CUDA
- [ ] Incremental index updates
- [ ] Distributed search support
- [ ] Additional distance metrics

### Future Considerations

- Learned index tuning
- Automatic parameter selection
- Integration with vector databases
