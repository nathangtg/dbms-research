# Benchmark Methodology

This document describes the benchmarking methodology used to evaluate ZGQ v8 against baseline algorithms.

## Experimental Setup

### Hardware

All experiments should be run on consistent hardware. Document:
- CPU model and cores
- RAM size
- Storage type (SSD recommended)

### Software

- Python 3.8+
- NumPy with MKL/OpenBLAS
- hnswlib for HNSW baseline
- Numba for JIT compilation

## Datasets

### Synthetic Data

Generated using clustered Gaussian distributions:

```python
# Cluster-based generation
n_clusters = max(10, n_vectors // 1000)
centers = np.random.randn(n_clusters, dimension)
vectors = centers[cluster_idx] + noise * 0.3
```

### Dataset Sizes

| Name | Vectors | Dimension | Queries |
|------|---------|-----------|---------|
| Small | 10,000 | 128 | 100 |
| Medium | 100,000 | 128 | 100 |
| Large | 1,000,000 | 128 | 100 |

### Real-World Datasets

For publication, evaluate on standard benchmarks:
- SIFT1M (128-d, 1M vectors)
- GIST1M (960-d, 1M vectors)
- Deep1B (96-d, 1B vectors)

## Metrics

### Recall@k

```
Recall@k = |Predicted ∩ GroundTruth| / k
```

Measured at k = 1, 5, 10, 20, 50, 100.

### Query Latency

- Mean latency per query (ms)
- Median latency (ms)
- P95 and P99 latency (ms)
- Throughput (queries/second)

### Memory Usage

- Index size on disk
- Peak memory during search
- Compression ratio vs raw vectors

### Build Time

- Total index construction time
- Per-component breakdown

## Benchmarking Protocol

### Warmup

Run 3-5 warmup iterations before measurement to:
- JIT compile Numba functions
- Populate CPU caches
- Stabilize system state

### Measurement Runs

- Minimum 5 runs per configuration
- Report mean and standard deviation
- Exclude obvious outliers (>3σ)

### Avoiding Bias

1. Shuffle query order between runs
2. Clear caches between algorithms
3. Run algorithms in different orders
4. Use same random seed for reproducibility

## Comparison Protocol

### Fair Comparison with HNSW

Both algorithms use:
- Same M parameter (16)
- Same ef_construction (200)
- Same ef_search (varies for curve)
- Same distance metric (L2)

### Recall-Latency Curves

Generate by varying:
- ZGQ: n_probe (4, 8, 16, 32) and ef_search (32, 64, 128)
- HNSW: ef_search (16, 32, 64, 128, 256)

## Running Benchmarks

### Quick Benchmark

```bash
cd v8
make benchmark-smoke   # tiny-scale run plus schema check
make benchmark         # 10K, 1000 queries, 5 repeats
```

### Full Benchmark Suite

```bash
python -m benchmarks.run_benchmarks --scale 10k 100k --out-dir results/$(date +%F)
python -m benchmarks.check_results results/<date>/summary.json --canonical
```

Output layout: `results/<date>/summary.json` (schema `zgq-bench/1`) and
`results/<date>/raw/<scale>_<index>.json` with every per-repeat measurement.
`--scale` accepts `tiny`, `10k`, `100k`, `1m`. Data and ground truth are
regenerated from the seed on every run, so no data files are stored.

## Reporting Results

### Required Information

1. Hardware specifications
2. Software versions
3. Dataset description
4. Parameter settings
5. Number of runs
6. Statistical measures (mean, std)

### Visualization

Generate plots for:
- Recall vs Latency curves
- Recall vs Memory trade-off
- Scaling with dataset size
- Parameter sensitivity

### Statistical Significance

- Use paired t-tests for comparisons
- Report p-values and effect sizes
- Consider multiple comparison corrections

## Reproducibility

All benchmarks should be reproducible:

```python
# Set seeds
np.random.seed(42)
config = ZGQConfig(random_state=42)
```

Save all results with timestamps:

```bash
benchmark_results_10k_20241204_143022.json
```

## Known Limitations

1. **Synthetic vs Real Data:** Synthetic data may not represent real distributions
2. **Cold Start:** First query may be slower due to caching
3. **Multi-tenancy:** Results assume dedicated system resources
4. **Scaling:** Large-scale (>10M) requires distributed evaluation
