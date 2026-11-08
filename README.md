# ZGQ v8: Zone-Guided Quantization for High-Performance Approximate Nearest Neighbor Search

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## Abstract

Zone-Guided Quantization (ZGQ) is a research Approximate Nearest Neighbor Search (ANNS) algorithm that combines hierarchical zonal partitioning with graph navigation. In the canonical benchmarks it matches HNSW recall at 10K vectors but trails HNSW at 100K and 1M (see below). ZGQ introduces:

1. **Adaptive Hierarchical Zones (AHZ)**: Multi-level zone structure that scales efficiently with dataset size
2. **Zone-Guided Navigation (ZGN)**: Leverages zone boundaries to prune search space during graph traversal  
3. **Optimized Distance Computation**: numba-compiled Euclidean distance kernels
4. **Residual Product Quantization (RPQ)**: Improved vector compression maintaining high recall

## Key Results

Canonical run `results/2026-11-08` (synthetic clustered data, dim 128, 1000 queries, single thread,
median of 5 repeats, Apple M4). Matched comparison HNSW-M16 vs ZGQ-M16, recall@10 % / ms per query:

| Scale | `ef` | HNSW-M16 | ZGQ-M16 |
|-------|------|----------|---------|
| 10K   | 64   | 92.4 / 0.046 | 92.3 / 0.049 |
| 10K   | 256  | 98.7 / 0.099 | 98.2 / 0.101 |
| 100K  | 64   | 86.1 / 0.084 | 78.9 / 0.065 |
| 100K  | 256  | 94.2 / 0.158 | 84.8 / 0.122 |
| 1M    | 64   | 77.6 / 0.124 | 68.1 / 0.091 |
| 1M    | 256  | 87.0 / 0.226 | 76.8 / 0.174 |

* ZGQ is less accurate than HNSW at 100K and 1M for every equal `ef`. Per-query time is lower at equal
  `ef`, but HNSW at lower `ef` gives higher recall at similar or lower latency.
* ZGQ builds 1.2x to 2.7x slower than HNSW and its index uses far more memory (macOS RSS deltas, so
  only the ordering is claimed).
* IVF-PQ without refinement saturates at 64%, 61% and 56% recall (10K, 100K, 1M).
* HNSW is hnswlib (C++); ZGQ is Python with numba, so timing is not a pure algorithmic comparison.

### Recall-improvement options (opt-in)

All on synthetic clustered Gaussian data, dim 128. **No real dataset (for example SIFT1M) has been run.**
Details in [docs/BENCHMARKS.md](docs/BENCHMARKS.md) and `results/2026-11-08-improvements/final/README.md`.

* The recall gap to HNSW-M16 at 100K and 1M is caused by zone-ordered graph insertion. With
  `ZGQConfig(insertion_order="random")` ZGQ-M16 reaches 86.9% vs 87.0% (1M) and 93.4% vs 94.2% (100K) at ef 256, at
  1.0x to 1.1x HNSW-M16 latency, 1.4x to 1.8x build time and 1.8x analytic index bytes. The gap closes, but this is
  HNSW with extra stored data, not an improvement over HNSW. The default is unchanged.
* Overlapping zones, adaptive probing and a global graph with zone-aware entry do not beat HNSW-M16 on the
  recall/latency trade-off (3.5x to 167x latency at equal recall for the zone-path variants).
* Compression-first ZGQ loses to IVF-PQ(R) and HNSW+SQ8 at equal memory.
* The benchmarked `batch_search(fast_mode=True)` path does not read zones at search time, so the zone-navigation
  description in the abstract does not describe the measured path.

Full tables, figures, limitations and threats to validity: [docs/BENCHMARKS.md](docs/BENCHMARKS.md).
Claim-by-claim traceability: [docs/CLAIMS.md](docs/CLAIMS.md).
Phase-by-phase account of the study: [docs/RESEARCH_LOG.md](docs/RESEARCH_LOG.md). Change history: [CHANGELOG.md](CHANGELOG.md).

## Installation

```bash
uv sync              # dev environment (default group), creates .venv from uv.lock
uv sync --no-dev     # runtime dependencies only
uv run pytest        # run the tests
```

## Quick Start

```python
from zgq import ZGQConfig, ZGQIndex

# Create and build index
config = ZGQConfig(
    n_zones='auto',        # Automatically determine optimal zones
    use_hierarchy=True,    # Multi-level zone hierarchy
    use_simd=True,         # Enable SIMD acceleration
    use_pq=True,           # Enable (residual) product quantization
)
index = ZGQIndex(config)
index.build(vectors)

# Search
ids, distances = index.search(query, k=10)

# Batch search
ids, distances = index.batch_search(queries, k=10, n_jobs=4)
```

## Project Structure

```
├── README.md
├── CHANGELOG.md
├── CONTRIBUTING.md
├── Makefile                     # test, verify, benchmark, figures, tables, claims and link checks
├── setup.py                     # Package setup and dependency source
├── pyproject.toml               # package metadata and dependencies (uv)
├── uv.lock                      # locked dependency versions
├── .python-version              # default interpreter for uv
├── pytest.ini
├── zgq/                         # Main package
│   ├── __init__.py
│   ├── index.py                 # ZGQIndex and ZGQConfig
│   ├── search.py                # Search algorithms
│   ├── core/
│   │   ├── zones.py             # Adaptive Hierarchical Zones
│   │   ├── graph.py             # Zone-Guided Graph
│   │   ├── distances.py         # Distance kernels
│   │   └── quantization.py      # Residual Product Quantization
│   └── utils/
│       ├── io.py                # Save/load
│       └── metrics.py           # Evaluation metrics
├── benchmarks/
│   ├── run_benchmarks.py        # Reproducible harness (ZGQ, HNSW, IVF-PQ)
│   ├── check_results.py         # summary.json schema validator
│   └── compare_improvements.py  # Variant vs HNSW-M16 tables from summary.json
├── scripts/
│   ├── check_claims.py          # Fails if a retired number reappears in the docs
│   └── check_links.py           # Markdown and LaTeX link and path checker
├── visualization/
│   ├── generate_ieee_figures.py # Figures from results/summary.json only
│   └── make_tables.py           # Markdown and LaTeX tables from summary.json
├── results/2026-11-08/          # Canonical results (summary.json, raw/, README.md)
├── figures/                     # Generated figures (PDF)
├── tables/                      # Generated tables (Markdown, LaTeX)
├── tests/                       # pytest suite
├── examples/quickstart.py
├── docs/                        # API.md, THEORY.md, BENCHMARKS.md, CLAIMS.md, RESEARCH_LOG.md, proofing.tex
├── latex/                       # Research paper LaTeX source
└── References/                  # Cited papers
```

## History

Earlier iterations (v1 to v7) were removed from the working tree. They remain available in git history at the tag `pre-v8-cleanup`. The v8 code previously lived in a `v8/` subdirectory and was moved to the repository root; file history is preserved. The study narrative is in [docs/RESEARCH_LOG.md](docs/RESEARCH_LOG.md) and the release notes are in [CHANGELOG.md](CHANGELOG.md).

## Algorithm Overview

### 1. Adaptive Hierarchical Zones (AHZ)

Unlike flat zone partitioning, AHZ creates a multi-level hierarchy:

```
Level 0: 1 zone (entire dataset)
Level 1: √N zones (coarse partitioning)
Level 2: N^(2/3) zones (fine partitioning)
```

Zone selection cost is analyzed in [docs/THEORY.md](docs/THEORY.md); it has not been benchmarked separately.

### 2. Zone-Guided Navigation (ZGN)

During graph traversal, ZGN uses zone boundaries to:
- **Prune**: Skip candidates in far zones early
- **Prioritize**: Explore candidates in query's zone first  
- **Bridge**: Use inter-zone connections efficiently

### 3. Distance Computation

Euclidean distance kernels are compiled with numba (`zgq/core/distances.py`). No speedup factor over a
baseline kernel has been measured, so none is claimed.

### 4. Residual Product Quantization (RPQ)

Encodes residuals from zone centroids instead of raw vectors:
```
residual = vector - zone_centroid
pq_code = PQ.encode(residual)
```

No measurement of quantization error against standard PQ has been committed, so no improvement is claimed.

## Limitations

The benchmarks use synthetic data, 1000 queries, a single machine and a Python implementation
compared against C++ hnswlib. See "Limitations and threats to validity" in
[docs/BENCHMARKS.md](docs/BENCHMARKS.md).

## Citation

If you use ZGQ in your research, please cite:

```bibtex
@article{zgq2024,
  title={Zone-Guided Quantization: High-Performance Approximate Nearest Neighbor Search},
  author={[Authors]},
  journal={[Journal/Conference]},
  year={2024}
}
```

## License

MIT License - see [LICENSE](LICENSE) for details.
