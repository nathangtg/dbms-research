# ZGQ Benchmark Visualization Guide

This guide explains how to visualize benchmark results for ZGQ and baseline ANNS methods.

## Quick Start

### Option 1: Automatic (Run Benchmark + Visualize)

When you run the benchmark, it automatically generates visualizations:

```bash
python benchmarks/comprehensive_benchmark.py
```

This will:
1. Run ZGQ, HNSW, and FAISS-IVF-PQ benchmarks
2. Save results to `benchmark_results.json`
3. Automatically generate all visualizations in `figures/`

### Option 2: Manual (Visualize Existing Results)

If you already have benchmark results, generate visualizations separately:

```bash
python visualize_results.py
```

Or specify custom files:

```bash
python visualize_results.py my_results.json output_dir/
```

## Generated Visualizations

The system generates 6 publication-quality figures:

### 1. **recall_latency_curve.png** (Primary Metric)
- **What it shows**: Recall vs. query latency trade-off
- **Why it matters**: This is the MAIN metric for ANNS evaluation
- **How to read**: 
  - X-axis: Query latency (lower is better)
  - Y-axis: Recall@10 (higher is better)
  - Points closer to top-left are optimal
  - Each point is labeled with its configuration parameter

**Key Insight from Your Results:**
- HNSW dominates: 96.7% recall @ 0.167ms
- ZGQ best: 98.9% recall @ 2.906ms (17x slower!)
- ZGQ's recall-latency trade-off is NOT superior to HNSW

### 2. **pareto_frontier.png**
- **What it shows**: Pareto-optimal configurations (black dashed line)
- **Why it matters**: Shows which method dominates at different recall ranges
- **How to read**: Points on the frontier are non-dominated solutions

**Key Insight:**
- HNSW dominates the entire Pareto frontier
- No ZGQ configuration is Pareto-optimal
- This contradicts the hypothesis that ZGQ is superior

### 3. **throughput_comparison.png**
- **What it shows**: Queries per second at different recall ranges
- **Why it matters**: Shows scalability for high-throughput applications
- **How to read**: Higher bars = better throughput

**Key Insight:**
- HNSW: 5,975 QPS @ 96.7% recall
- ZGQ: 344 QPS @ 98.9% recall
- HNSW is 17x faster in throughput

### 4. **latency_distribution.png**
- **What it shows**: p50, p95, p99 latency percentiles at ~70% recall
- **Why it matters**: Tail latencies are critical for production systems
- **How to read**: Log scale shows latency distribution spread

**Key Insight:**
- HNSW has consistent low latency across percentiles
- ZGQ has higher tail latencies

### 5. **build_time_comparison.png**
- **What it shows**: Index construction time
- **Why it matters**: Impacts offline indexing costs
- **How to read**: Lower bars = faster build

**Key Insight:**
- HNSW: 0.27s (14x faster build!)
- ZGQ: 3.95s
- FAISS: 0.48s

### 6. **memory_comparison.png**
- **What it shows**: Index memory footprint
- **Why it matters**: Memory constraints in production
- **How to read**: Lower bars = smaller index

**Key Insight:**
- All methods have similar memory (~5-6 MB)
- FAISS-IVF-PQ is smallest (0.2 MB) but broken (0% recall)

## Understanding Your Results

### ⚠️ Critical Finding: Hypothesis NOT Validated

Your benchmark results show that **ZGQ does NOT achieve superior performance compared to HNSW**:

| Metric | ZGQ Best | HNSW Best | Winner |
|--------|----------|-----------|--------|
| **Recall@10** | 98.9% | 96.7% | ZGQ (+2.2%) |
| **Latency** | 2.906ms | 0.167ms | **HNSW (17x faster)** |
| **Throughput** | 344 QPS | 5,975 QPS | **HNSW (17x higher)** |
| **Build Time** | 3.95s | 0.27s | **HNSW (14x faster)** |
| **Memory** | 5.77 MB | 5.49 MB | HNSW (similar) |

### Why is ZGQ Slower?

Possible reasons:

1. **Zone selection overhead**: Computing distances to all zone centroids
2. **Python overhead**: HNSW uses C++ backend, ZGQ is pure Python
3. **Small dataset**: 10K vectors may be too small to show ZGQ's benefits
4. **Parameter tuning**: ZGQ may need different parameters
5. **Implementation inefficiency**: Room for optimization

### What to Do Next?

#### Option A: Optimize ZGQ Implementation
- Profile the code to find bottlenecks
- Implement Numba/Cython acceleration
- Optimize zone selection algorithm
- Add caching for zone distances

#### Option B: Test on Larger Datasets
- Current: 10K vectors × 128D
- Try: 1M vectors × 128D (or higher)
- ZGQ's zonal approach may benefit from scale

#### Option C: Re-evaluate Hypothesis
- Accept that ZGQ may not be superior to HNSW on this dataset
- Focus on specific use cases where ZGQ excels:
  - Memory-constrained environments?
  - Specific data distributions?
  - Update-heavy workloads?

## Programmatic Usage

You can also use the visualization module in your own scripts:

```python
from src.visualization import BenchmarkVisualizer

# Create visualizer
viz = BenchmarkVisualizer('benchmark_results.json', 'output_dir/')

# Generate specific plots
viz.plot_recall_latency_curve()
viz.plot_pareto_frontier()
viz.plot_throughput_comparison()

# Or generate all at once
viz.generate_all_figures()

# Get text summary
report = viz.generate_summary_report()
print(report)
```

## Customization

Edit `src/visualization.py` to customize:

- Colors: Modify the `colors` dict in each plot method
- Figure size: Change `plt.rcParams['figure.figsize']`
- DPI: Adjust `plt.rcParams['savefig.dpi']` for resolution
- Style: Change `sns.set_style()` and `sns.set_context()`

## Requirements

The visualization module requires:

```bash
pip install matplotlib seaborn pandas numpy
```

These are included in `requirements.txt`.

## Troubleshooting

### ImportError: No module named 'matplotlib'
```bash
pip install matplotlib seaborn pandas
```

### No figures generated
- Check that `benchmark_results.json` exists
- Ensure at least one method has non-zero recall
- Check terminal for error messages

### FAISS shows 0% recall
This is a known issue with the FAISS benchmark implementation. The FAISS-IVF-PQ index may need:
- More training iterations
- Different quantization parameters
- Proper index training with representative data

## File Outputs

After running visualization:

```
v7/
├── benchmark_results.json       # Raw benchmark data
├── figures/                     # Generated visualizations
│   ├── recall_latency_curve.png
│   ├── pareto_frontier.png
│   ├── throughput_comparison.png
│   ├── latency_distribution.png
│   ├── build_time_comparison.png
│   ├── memory_comparison.png
│   └── benchmark_summary.txt
└── visualize_results.py         # Standalone script
```

## Citation

If you use these visualizations in a paper, ensure to:

1. Report both recall AND latency (not just one metric)
2. Show Pareto frontier analysis
3. Include build time and memory comparisons
4. Acknowledge limitations if ZGQ underperforms baselines

## Next Steps

1. ✓ Visualizations generated
2. ⚠️ Analyze why ZGQ is slower than expected
3. 🔍 Profile code to find bottlenecks
4. 🚀 Optimize critical paths
5. 📊 Re-run benchmarks on larger datasets
6. 📝 Document findings and conclusions
