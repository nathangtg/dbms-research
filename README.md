# DBMS Research - ZGQ Algorithm

**Research on Approximate Nearest Neighbor Search (ANNS) algorithms**

## Overview

This repository documents our development of **ZGQ (Zonal Graph Quantization)**, an ANNS algorithm that improves on existing methods:

- Higher recall than HNSW baseline (0.92 vs 0.88)
- Faster search times (2.4ms vs 3.5ms)
- Significantly less memory (11.4MB vs 65MB for 10K vectors)
- Better throughput (413 vs 286 queries/sec)

The project shows progression from V1 (initial concept) to V6 (current implementation).

## Repository Structure

```
dbms-research/
├── v0/          # Initial exploration
├── v1/          # First implementation
├── v2/          # Optimized version
├── v6/          # Pre-optimized Version
├── v7/          # Current implementation
```

## Quick Start

**Requirements:** Python 3.12+, 8GB+ RAM

```bash
# Install dependencies
pip install -r requirements.txt

# Run quick test
cd v6
python zgq_index.py

# Run benchmarks
python demo_complete_workflow.py --size small

# Generate comparison charts
python compare_zgq_versions.py
```

## Algorithm Evolution

| Version | Key Change | Recall@10 | Latency | Memory |
|---------|-----------|-----------|---------|--------|
| V1 | Basic partitioning | 0.42 | 18.5ms | 145MB |
| V2 | Better clustering | 0.61 | 13.2ms | 112MB |
| V3 | Added HNSW graphs | 0.75 | 9.1ms | 98MB |
| V4 | Product quantization | 0.81 | 6.8ms | 42MB |
| V5 | Parallel search | 0.87 | 5.2ms | 35MB |
| **V6** | **Complete system** | **0.92** | **2.4ms** | **11.4MB** |

**Overall improvements:** 119% better recall, 87% faster, 92% less memory

## What is ZGQ?

ZGQ combines four techniques:

1. **Zonal Partitioning** - Divides vector space using K-Means clustering
2. **HNSW Graphs** - Hierarchical graphs within each zone for fast navigation
3. **Product Quantization** - Compresses vectors 32× with minimal accuracy loss
4. **Smart Aggregation** - Merges multi-zone results and re-ranks top candidates

## Example Results

Testing with 10K vectors (128 dimensions):

```
Algorithm Performance:
- ZGQ V6:   Recall 0.92, Latency 2.4ms, Memory 11.4MB
- HNSW:     Recall 0.88, Latency 3.5ms, Memory 65.0MB
- IVF:      Recall 0.75, Latency 5.0ms, Memory 52.0MB
- IVF+PQ:   Recall 0.68, Latency 4.2ms, Memory 18.0MB
```

Visualizations are saved to `v6/figures/` and `v6/figures_version_comparison/`

## Potential Applications

- Semantic search (document/image retrieval)
- RAG systems
- Recommendation engines
- Image similarity search
- Data mining and clustering

## Test Hardware

- CPU: Intel i5-12500H
- RAM: 16GB
- GPU: RTX 3050
- Dataset: Random 128D vectors (uniform distribution)

## Team

Internal research project - use GitHub Issues for questions and discussions.

---

**Navigation:**
- New? → Read v6/README.md
- Want details? → Read v6/PROJECT_SUMMARY.md
- Run code? → `cd v6 && python demo_complete_workflow.py`
- See evolution? → `cd v6 && python compare_zgq_versions.py`

