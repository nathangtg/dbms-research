# DBMS Research - ZGQ Algorithm

**Research on Approximate Nearest Neighbor Search (ANNS) algorithms**

## Overview

This repository documents the development of **ZGQ (Zonal Graph Quantization)**, a novel ANNS algorithm designed to optimize the memory-performance trade-off in high-dimensional vector search. ZGQ combines hierarchical zonal partitioning with optimized graph navigation. It is evaluated against HNSW and IVF-PQ; the measured results (below) do not show an advantage over HNSW.

**Status of the evidence (canonical run 2026-11-08, synthetic data, single thread, Apple M4):**
- At 10K vectors ZGQ-M16 matches HNSW-M16 recall at equal `ef` (92.3% vs 92.4% at `ef=64`) and is 2 to 7% slower per query.
- At 100K and 1M vectors ZGQ is less accurate than HNSW-M16 at every equal `ef` (for example 78.9% vs 86.1% at 100K, `ef=64`), and HNSW reaches the same recall at lower latency.
- ZGQ builds 1.2x to 2.7x slower and uses much more memory than HNSW. HNSW is hnswlib (C++); ZGQ is Python, so timing is not an algorithmic comparison.
- IVF-PQ recall saturates at 64%, 61% and 56% (10K, 100K, 1M) without refinement.

Full tables, caveats and threats to validity: [v8/docs/BENCHMARKS.md](v8/docs/BENCHMARKS.md). Every number in the docs is traced in [v8/docs/CLAIMS.md](v8/docs/CLAIMS.md). The study history is in [v8/docs/RESEARCH_LOG.md](v8/docs/RESEARCH_LOG.md) and [v8/CHANGELOG.md](v8/CHANGELOG.md).

## Repository Structure

The current implementation lives in `v8/`.

```
dbms-research/
├── v8/           # ZGQ implementation (package, tests, benchmarks, docs)
│   ├── zgq/
│   ├── tests/
│   ├── benchmarks/
│   └── docs/
├── latex/        # Research paper LaTeX source
└── References/   # Cited papers
```

## History

Earlier iterations (v1 to v7) were removed from the working tree. They remain available in git history at the tag `pre-v8-cleanup`.

## What is ZGQ?

ZGQ introduces a hybrid indexing framework that integrates four key technologies:

1.  **Adaptive Hierarchical Zones (AHZ):** A multi-level partitioning strategy that divides the vector space into semantically coherent zones using K-Means clustering. This reduces the search space and improves cache locality.
2.  **Zone-Guided Graph (ZGG):** Instead of a monolithic graph, ZGQ constructs independent, lightweight HNSW graphs within each zone. This enables parallel search execution and reduces global graph complexity.
3.  **Residual Product Quantization (RPQ):** Compresses vectors to reduce memory footprint while maintaining high recall by quantizing the residuals from zone centroids.
4.  **Smart Aggregation & Re-ranking:** Merges results from multiple zones and re-ranks top candidates to ensure high accuracy.

## Quick Start

To run the latest version of ZGQ (v8):

**Requirements:** Python 3.8+, 8GB+ RAM

1.  **Install Dependencies:**
    ```bash
    cd v8
    pip install -e '.[dev]'
    ```

2.  **Run the quick start:**
    ```bash
    cd v8
    python examples/quickstart.py
    ```

3.  **Run Tests and Benchmarks:**
    ```bash
    cd v8
    make test
    make verify            # imports, tiny search, benchmark smoke, claims and link checks
    make benchmark-smoke   # tiny-scale benchmark plus schema check
    ```

## Performance Results

Canonical results (128-d synthetic clustered data, 1000 queries, single thread, median of 5 repeats) are in
`v8/results/2026-11-08/`. Matched comparison, HNSW-M16 vs ZGQ-M16 at `ef=64`, recall@10 % / ms per query:

| Scale | HNSW-M16 | ZGQ-M16 |
|-------|----------|---------|
| 10K   | 92.4 / 0.046 | 92.3 / 0.049 |
| 100K  | 86.1 / 0.084 | 78.9 / 0.065 |
| 1M    | 77.6 / 0.124 | 68.1 / 0.091 |

ZGQ is faster per query at equal `ef` at 100K and 1M but loses 7 to 10 recall points. At matched recall HNSW is
at least as fast. See [v8/docs/BENCHMARKS.md](v8/docs/BENCHMARKS.md) for the full sweep and limitations.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

