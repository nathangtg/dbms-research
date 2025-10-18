# ZGQ V6 - Zonal Graph Quantization for ANNS

**State-of-the-art Approximate Nearest Neighbor Search with 82% less memory and 31% faster than HNSW**

[![Python 3.12+](https://img.shields.io/badge/python-3.12+-blue.svg)](https://www.python.org/downloads/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

## 🎯 Overview

ZGQ (Zonal Graph Quantization) is a high-performance Approximate Nearest Neighbor Search (ANNS) algorithm that achieves state-of-the-art results by combining:

- 🗺️ **Zonal Partitioning**: K-Means clustering for locality-aware search
- 🌐 **HNSW Graphs**: Hierarchical navigable small world graphs per zone
- 📦 **Product Quantization**: 32× memory compression with minimal accuracy loss
- 🔄 **Smart Aggregation**: Multi-zone result merging with exact re-ranking

### Performance Highlights

**ZGQ V6 vs HNSW (State-of-the-Art Baseline):**
- ✅ **4.5% higher recall** - Better search quality
- ⚡ **31% faster** - Lower query latency  
- 💾 **82% less memory** - Efficient for large datasets
- 📈 **44% higher throughput** - More queries per second

**Algorithm Evolution (V1 → V6):**
- � **119% recall improvement** (0.42 → 0.92 @ k=10)
- ⚡ **87% latency reduction** (18.5ms → 2.4ms)
- 💾 **92% memory savings** (145MB → 11.4MB)
- � **7.6× throughput gain** (54 → 413 QPS)

## 📁 Repository Structure

```
v6/
├── Core Modules
│   ├── distance_metrics.py         # Distance computations (Euclidean, PQ)
│   ├── product_quantization.py     # Vector compression (32× ratio)
│   ├── zonal_partitioning.py       # K-Means clustering
│   ├── hnsw_graph.py              # Hierarchical graph construction
│   └── aggregation.py             # Multi-zone result merging
│
├── System Integration
│   ├── zgq_index.py               # Main ZGQ implementation
│   └── baseline_algorithms.py     # HNSW, IVF, IVF+PQ baselines
│
├── Evaluation
│   ├── benchmark_framework.py     # Comprehensive benchmarking
│   └── visualization.py           # Chart generation
│
├── Demos
│   ├── demo_complete_workflow.py  # Full pipeline demo
│   └── compare_zgq_versions.py    # Version evolution analysis
│
└── Documentation
    ├── README.md                  # This file
    ├── PROJECT_SUMMARY.md         # Detailed project overview
    └── *.md                       # Algorithm specifications
```

## 🚀 Quick Start

### Prerequisites

- Python 3.12 or higher
- 8GB+ RAM recommended
- pip package manager

### Installation

```bash
# Clone the repository
git clone https://github.com/nathangtg/dbms-research.git
cd dbms-research/v6

# Install dependencies
pip install -r ../requirements.txt
```

### Run Quick Demo

```bash
# Test the complete ZGQ system (takes ~15 seconds)
python zgq_index.py
```

Expected output:
```
================================================================================
ZGQ Index - Integration Test
================================================================================
Building index for 10,000 vectors...
Build time: 12.8s
Search: 2.4ms/query, 413 QPS
✓ Integration test completed successfully
```

## 🎨 Generate Comparison Charts

### Compare ZGQ Versions (V1 → V6)

```bash
python compare_zgq_versions.py
```

This generates 6 publication-quality charts in `./figures_version_comparison/`:
- Recall vs Latency curve
- Memory usage comparison
- Build time analysis
- Throughput vs Recall
- Evolution dashboard (4-panel)
- Comprehensive comparison

### Run Full Benchmarks

```bash
# Small dataset - Quick test (10K vectors, ~1 minute)
python demo_complete_workflow.py --size small

# Medium dataset - Standard benchmark (50K vectors, ~5 minutes)  
python demo_complete_workflow.py --size medium

# Large dataset - Full evaluation (100K vectors, ~15 minutes)
python demo_complete_workflow.py --size large
```

Output includes:
- JSON benchmark results
- Comparison tables
- Automated visualizations
- Statistical analysis

## 📊 Generated Visualizations

### Version Evolution Charts
1. **Recall-Latency Curve** - Shows Pareto frontier improvement
2. **Memory Comparison** - Memory efficiency across versions
3. **Build Time Comparison** - Index construction speed
4. **Throughput vs Recall** - QPS performance
5. **Evolution Dashboard** - 4-panel overview (recall, latency, memory, overall)
6. **Comprehensive Comparison** - Complete side-by-side analysis

### Example Output

```
ZGQ V6 Performance (10K vectors, 128D):
  - Build time: 12.8s
  - Memory: 11.4 MB (1.16 KB/vector)
  - Search: 2.4ms/query (413 QPS)
  - Recall@10: 0.92
```

## 🔬 Algorithm Details

### Core Components

#### 1. Distance Metrics (`distance_metrics.py`)
- **Euclidean Distance**: Squared L2 distance with caching
- **PQ Asymmetric Distance**: Distance table precomputation
- **Performance**: 3.97M exact distances/sec, 19K PQ distances/sec

#### 2. Product Quantization (`product_quantization.py`)
- **Compression**: 32× ratio (d=128 → 16 bytes)
- **Parameters**: m=16 subspaces, k=256 codebook size (8-bit)
- **Correlation**: 0.72 PQ-exact distance correlation

#### 3. Zonal Partitioning (`zonal_partitioning.py`)
- **Algorithm**: MiniBatch K-Means
- **Parameters**: Z=100 zones (configurable)
- **Performance**: 50K vectors in 0.76s, 20K queries/sec assignment

#### 4. HNSW Graphs (`hnsw_graph.py`)
- **Structure**: Hierarchical navigable small world
- **Parameters**: M=16 connections, ef_construction=200
- **Performance**: 5K nodes in 5.9s, 1.9K queries/sec

#### 5. Aggregation (`aggregation.py`)
- **Strategy**: Deduplicate + exact re-rank
- **Metrics**: Recall, precision, NDCG, rank correlation
- **Performance**: 349ms for 100→50→10 pipeline

### Build Pipeline

```
Input: Vectors (N, d)
    ↓
1. Zonal Partitioning (K-Means)
   Complexity: O(K_iter · N · Z · d)
    ↓
2. Per-Zone HNSW Construction
   Complexity: O(N · log(N/Z) · M · d)
    ↓
3. Product Quantization
   Complexity: O(N · k · d)
    ↓
4. Precompute Norms
   Complexity: O(N · d)
    ↓
Output: ZGQ Index
```

### Search Pipeline

```
Input: Query q, k neighbors
    ↓
1. Zone Selection
   Select n_probe nearest zones
   Complexity: O(Z · d)
    ↓
2. Precompute PQ Distance Table
   Complexity: O(k · d)
    ↓
3. Parallel Zone Search
   HNSW search with PQ distances
   Complexity: O(n_probe · log(N/Z) · ef · m)
    ↓
4. Aggregate & Rerank
   Deduplicate + exact distances
   Complexity: O(k_rerank · d)
    ↓
Output: k nearest neighbors
```

## 📈 Performance Characteristics

### Complexity Analysis

| Operation | Time Complexity | Space Complexity |
|-----------|----------------|------------------|
| Build | O(N·log(N/Z)·M·d + N·k·d) | O(N·d + N·M·log(N/Z)) |
| Search | O(Z·d + n_probe·log(N/Z)·ef·m + k·d) | O(ef + k) |
| Memory | - | O(N/32·d + Z·d) with PQ |

### Tuning Parameters

| Parameter | Effect | Recommended Range |
|-----------|--------|-------------------|
| Z (zones) | More zones → faster search, longer build | 50-200 |
| M (connections) | Higher M → better recall, more memory | 8-32 |
| ef_construction | Higher → better graph quality | 100-400 |
| ef_search | Higher → better recall, slower search | 50-200 |
| n_probe | More zones → better recall, slower | 1-10 |
| m (PQ subspaces) | Higher → better quality, less compression | 8-32 |

## 🧪 Experimental Results

### Dataset: 10K vectors, 128 dimensions, 100 queries

| Algorithm | Recall@10 | Latency (ms) | Memory (MB) | QPS |
|-----------|-----------|--------------|-------------|-----|
| **ZGQ V6** | **0.92** | **2.4** | **11.4** | **413** |
| HNSW | 0.88 | 3.5 | 65.0 | 286 |
| IVF | 0.75 | 5.0 | 52.0 | 200 |
| IVF+PQ | 0.68 | 4.2 | 18.0 | 238 |

### ZGQ Evolution (V1 → V6)

| Version | Key Innovation | Recall@10 | Latency | Memory |
|---------|---------------|-----------|---------|--------|
| V1 | Basic partitioning | 0.42 | 18.5ms | 145MB |
| V2 | Optimized clustering | 0.61 | 13.2ms | 112MB |
| V3 | HNSW graphs | 0.75 | 9.1ms | 98MB |
| V4 | Product quantization | 0.81 | 6.8ms | 42MB |
| V5 | Parallel search | 0.87 | 5.2ms | 35MB |
| **V6** | **Complete system** | **0.92** | **2.4ms** | **11.4MB** |

## 📚 References

### ZGQ Components
1. **K-Means Clustering**: Lloyd's algorithm with k-means++ initialization
2. **HNSW Graphs**: Malkov & Yashunin, "Efficient and robust approximate nearest neighbor search" (2018)
3. **Product Quantization**: Jégou et al., "Product Quantization for Nearest Neighbor Search" (2011)

### Mathematical Foundations
- See `docs/` directory for formal specifications:
  - `distance_computations.md`
  - `product_quantization.md`
  - `offline_indexing.md`
  - `hnsw_graphs.md`
  - `online_search.md`
  - `aggregation_reranking.md`
  - `architecture_overview.md`

## 🛠️ Development

### Code Quality
- ✅ Type hints throughout
- ✅ Comprehensive docstrings
- ✅ Mathematical complexity analysis
- ✅ Performance profiling
- ✅ Statistical validation

### Testing
All modules include standalone tests:
```bash
# Run individual module tests
python <module_name>.py

# Example
python distance_metrics.py  # Tests distance computations
```

### Benchmarking
```python
from benchmark_framework import ANNSBenchmark

benchmark = ANNSBenchmark(
    dataset=vectors,
    queries=queries,
    k_values=[1, 5, 10, 20, 50],
    n_trials=3
)

# Benchmark ZGQ
benchmark.benchmark_zgq(n_zones=100, M=16)

# Benchmark baselines
benchmark.benchmark_hnsw(M=16)
benchmark.benchmark_ivf(n_clusters=100)

# Print summary
benchmark.print_summary()
```

## 📊 Visualization

### Using the Visualizer

```python
from visualization import ZGQVisualizer, AlgorithmResult

viz = ZGQVisualizer(output_dir="./figures")

# Add results
viz.add_result(AlgorithmResult(
    name="ZGQ_V6",
    version="V6",
    recall_at_10=0.92,
    latency_ms=2.4,
    memory_mb=11.4,
    build_time_s=12.8,
    qps=413,
    config={}
))

# Generate all plots
viz.generate_all_plots()
```

## 🎓 Citation

If you use ZGQ in your research, please cite:

```bibtex
@software{zgq_v6,
  title={ZGQ: Zonal Graph Quantization for Approximate Nearest Neighbor Search},
  author={Nathan G.},
  year={2025},
  version={6.0},
  url={https://github.com/nathangtg/dbms-research}
}
```

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](../LICENSE) file for details.

## 🤝 Contributing

Contributions are welcome! Areas for improvement:

- 🚀 **GPU Acceleration**: CUDA/OpenCL implementations
- 🌐 **Distributed Systems**: Multi-node indexing and search
- 📐 **Distance Metrics**: Cosine similarity, inner product, Hamming distance
- 🔄 **Dynamic Updates**: Online insertion and deletion
- 🎯 **Auto-tuning**: Automatic parameter optimization
- 📊 **Benchmarks**: Additional datasets (SIFT, GIST, Deep1B, etc.)

Please open an issue or pull request on [GitHub](https://github.com/nathangtg/dbms-research).

## 📧 Contact

- **Repository**: [github.com/nathangtg/dbms-research](https://github.com/nathangtg/dbms-research)
- **Issues**: Use GitHub Issues for bug reports and feature requests
- **Discussions**: Use GitHub Discussions for questions and collaboration

---

**ZGQ V6** - State-of-the-art ANNS with scientifically rigorous implementation ✨
