# ZGQ V7 - Zonal Graph Quantization for ANNS

**Next-generation Approximate Nearest Neighbor Search with theoretical foundation and academic validation**

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

## 🎯 Overview

ZGQ (Zonal Graph Quantization) V7 is an advanced Approximate Nearest Neighbor Search (ANNS) algorithm that combines:

- 🗺️ **Zonal Partitioning**: K-Means clustering for locality-aware search
- 🌐 **HNSW Graphs**: Hierarchical navigable small world graphs per zone
- 📦 **Product Quantization**: 16-32× memory compression with minimal accuracy loss
- 🔄 **Smart Aggregation**: Multi-zone result merging with exact re-ranking

### Performance Highlights

**ZGQ V7 vs HNSW (State-of-the-Art Baseline):**
- ✅ **Higher recall** - Better search quality through multi-zone approach
- ⚡ **Faster search** - Parallel zone search with optimized parameters
- 💾 **Less memory** - Efficient for large-scale deployments
- 📈 **Higher throughput** - More queries per second

## 📁 Repository Structure

```
v7/
├── docs/                    # Documentation files
│   ├── README.md            # This file
│   ├── THEORY.md            # Theoretical foundation
│   ├── ARCHITECTURE.md      # System architecture
│   ├── API.md               # API documentation
│   ├── PARAMETERS.md        # Parameter tuning guide
│   └── VALIDATION.md        # Academic validation plan
├── src/                     # Source code
│   ├── core/                # Core modules
│   │   ├── distances.py     # Distance computations
│   │   ├── kmeans.py        # Zonal partitioning
│   │   ├── hnsw_wrapper.py  # Per-zone HNSW management
│   │   └── product_quantizer.py # PQ training & encoding
│   ├── index.py             # Main ZGQIndex class
│   ├── search.py            # Search algorithms
│   └── serialization.py     # Save/load index
├── tests/                   # Test files
├── benchmarks/              # Benchmark implementations
└── examples/                # Usage examples
```

## 🚀 Quick Start

### Prerequisites

- Python 3.9 or higher
- 16GB+ RAM recommended for large datasets
- pip package manager

### Installation

```bash
# Create virtual environment
python3.10 -m venv zgq_env
source zgq_env/bin/activate  # On Windows: zgq_env\Scripts\activate

# Install dependencies
pip install numpy>=1.24.0
pip install scikit-learn>=1.3.0
pip install hnswlib>=0.7.0
pip install faiss-cpu>=1.7.4  # or faiss-gpu for GPU support
pip install matplotlib>=3.7.0
pip install tqdm>=4.65.0
pip install psutil>=5.9.0  # for memory profiling
```

### Minimal Working Example

```python
import numpy as np
from zgq import ZGQIndex

# Generate sample data
N, d = 10000, 128
vectors = np.random.randn(N, d).astype('float32')
queries = np.random.randn(100, d).astype('float32')

# Build index
index = ZGQIndex(
    n_zones=100,
    hnsw_M=16,
    hnsw_ef_construction=200,
    use_pq=True,
    pq_m=16,
    pq_nbits=8
)

print("Building index...")
index.build(vectors)

# Search
print("Searching...")
k = 10
n_probe = 8
results = index.search(queries[0], k=k, n_probe=n_probe)
print(f"Top-{k} neighbors: {results}")
```