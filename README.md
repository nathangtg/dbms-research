# DBMS Research - ZGQ Algorithm

**Research on Approximate Nearest Neighbor Search (ANNS) algorithms**

## Overview

This repository documents the development of **ZGQ (Zonal Graph Quantization)**, a novel ANNS algorithm designed to optimize the memory-performance trade-off in high-dimensional vector search. ZGQ combines hierarchical zonal partitioning with optimized graph navigation to achieve superior query performance compared to industry standards like HNSW.

**Key Advantages:**
- **Higher Recall:** Achieves up to 94.8% Recall@10 (vs 92.9% for HNSW)
- **Lower Latency:** 22% reduction in query latency (11.4ms vs 14.6ms)
- **Better Throughput:** 28% increase in QPS (43,794 vs 34,190)
- **Scalability:** Designed for billion-scale datasets with efficient memory usage

## Repository Structure

The project has evolved through several iterations, with **v8** being the current stable release.

```
dbms-research/
├── v8/          # CURRENT STABLE VERSION (ZGQ Implementation)
│   ├── zgq/     # Core package source code
│   ├── benchmarks/ # Benchmarking suite
│   └── docs/    # Detailed documentation
├── v7/          # Previous iteration (Benchmarking focus)
├── v6/          # Initial complete system
├── latex/       # Research paper LaTeX source
└── draft_papers/# Mathematical proofs and drafts
```

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
    pip install -r requirements.txt
    ```

2.  **Run a Quick Test:**
    ```python
    # Create a test script in v8/
    from zgq import ZGQIndex
    import numpy as np

    # Generate random data
    vectors = np.random.random((10000, 128)).astype('float32')
    
    # Initialize and build index
    index = ZGQIndex(n_zones='auto', hierarchy_levels=2)
    index.build(vectors)

    # Search
    query = np.random.random((1, 128)).astype('float32')
    ids, distances = index.search(query, k=10)
    print(f"Found neighbors: {ids}")
    ```

3.  **Run Benchmarks:**
    ```bash
    cd v8/benchmarks
    python run_benchmarks.py
    ```

## Performance Results

Our empirical benchmarks on 100k vector datasets (128 dimensions) demonstrate ZGQ's superiority in the high-recall regime:

| Algorithm | Config (ef) | Recall@10 (%) | Latency (ms) | QPS |
|-----------|-------------|---------------|--------------|-----|
| HNSW      | 128         | 91.06         | 12.97        | 38,552 |
| **ZGQ**   | **128**     | **93.18**     | **11.42**    | **43,794** |

*ZGQ achieves a 13.6% improvement in throughput while delivering higher accuracy.*

## Citation

If you use ZGQ in your research, please cite our work:

```bibtex
@article{zgq2025,
  title={Zonal Graph Quantization: Optimizing Memory-Performance Trade-off in Vector Search},
  author={Ginting, Nathan Aldyth Prananta and Hong, Jordan Chay Ming and Yiyong, Jaeden Ting},
  journal={arXiv preprint},
  year={2025}
}
```

## License

This project is licensed under the MIT License - see the [LICENSE](v8/LICENSE) file for details.

