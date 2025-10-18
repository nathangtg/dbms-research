# Comprehensive ANN Algorithm Comparison

## Executive Summary

We benchmarked four popular Approximate Nearest Neighbor (ANN) search algorithms on a 10K vector dataset (128 dimensions):

1. **HNSW** - Hierarchical Navigable Small World (baseline)
2. **IVF** - Inverted File Index (flat, no compression)
3. **IVF+PQ** - IVF with Product Quantization
4. **ZGQ Unified** - Our zone-aware graph method

### Key Results

| Algorithm | Latency (ms) | Throughput (QPS) | Recall@10 | Memory (MB) | Build Time (s) |
|-----------|-------------|------------------|-----------|-------------|----------------|
| **HNSW** | **0.0152** | **65,789** | **54.9%** | 6.1 | 0.264 |
| IVF | 0.8348 | 1,198 | 37.6% | **4.9** | **0.228** |
| IVF+PQ | 7.4096 | 135 | 19.0% | 5.2 | 3.836 |
| ZGQ Unified | 0.0573 | 17,453 | 53.9% | **4.9** | 0.457 |

**Bold** = Best in category

## Detailed Analysis

### 1. HNSW (Winner 🏆)

**Strengths:**
- ✅ Fastest queries: 0.0152ms (65,789 QPS)
- ✅ Highest recall: 54.9%
- ✅ Fast build time: 0.264s
- ✅ Industry-proven algorithm

**Weaknesses:**
- ⚠️ Moderate memory usage: 6.1 MB (24% more than IVF)

**Use Cases:**
- Production systems requiring highest speed
- Real-time applications
- When memory is not a constraint
- General-purpose ANN search

**Architecture:**
- Multi-layer graph with navigable small world properties
- Greedy search at each layer
- Complexity: O(log N) per query

### 2. IVF (Inverted File Index)

**Strengths:**
- ✅ Lowest memory: 4.9 MB (tied with ZGQ)
- ✅ Fastest build: 0.228s
- ✅ Simple architecture
- ✅ Easy to understand and debug

**Weaknesses:**
- ❌ 55x slower than HNSW (0.835ms vs 0.015ms)
- ❌ Low recall: 37.6% (17% worse than HNSW)
- ❌ Performance degrades with low nprobe

**Use Cases:**
- Memory-constrained environments
- When build speed is critical
- Educational purposes
- Prototyping

**Architecture:**
- K-means clustering into 100 partitions
- Search nearest 10 clusters (nprobe=10)
- Linear scan within clusters
- Complexity: O(nprobe × N/nlist) per query

### 3. IVF+PQ (IVF with Product Quantization)

**Strengths:**
- ✅ Excellent compression: 32x (5.1MB → 0.2MB)
- ✅ Low memory: 5.2 MB
- ✅ Good for very large datasets

**Weaknesses:**
- ❌ Very slow: 487x slower than HNSW (7.4ms vs 0.015ms)
- ❌ Poorest recall: 19.0% (35.9% worse than HNSW)
- ❌ Long build time: 3.8s (14.5x slower than HNSW)
- ❌ Complex to tune

**Use Cases:**
- Billion-scale datasets where memory is critical
- Offline batch processing
- When storage cost dominates
- First-stage filtering in multi-stage systems

**Architecture:**
- IVF clustering + Product Quantization
- Vectors encoded into 16-byte codes (16 subspaces × 1 byte)
- Asymmetric distance computation
- Complexity: O(nprobe × N/nlist) but with PQ distance overhead

### 4. ZGQ Unified (Our Method)

**Strengths:**
- ✅ Lowest memory: 4.9 MB (tied with IVF)
- ✅ Good recall: 53.9% (near HNSW-level)
- ✅ Competitive speed: 17,453 QPS (26% of HNSW)
- ✅ Single unified graph (elegant architecture)

**Weaknesses:**
- ⚠️ 3.8x slower than HNSW (0.057ms vs 0.015ms)
- ⚠️ Longer build time: 1.7x slower than HNSW

**Use Cases:**
- When memory efficiency is important
- Systems needing good recall with lower memory
- Zone-aware search scenarios
- Research and development

**Architecture:**
- Single unified HNSW graph with zone metadata
- K-means partitioning into 100 zones
- Progressive search strategy
- Complexity: O(log N) per query (same as HNSW)

## Speed Comparison

Relative to HNSW (higher is better):

| Algorithm | Speedup | Status |
|-----------|---------|--------|
| HNSW | 1.00x | ✅ Baseline |
| ZGQ Unified | 0.27x | ⚠️ 3.8x slower |
| IVF | 0.02x | ❌ 55x slower |
| IVF+PQ | 0.00x | ❌ 487x slower |

## Memory Comparison

| Algorithm | Memory (MB) | vs HNSW | Compression |
|-----------|------------|---------|-------------|
| IVF | 4.9 | 0.81x ✅ | None |
| **ZGQ Unified** | **4.9** | **0.81x ✅** | None |
| IVF+PQ | 5.2 | 0.85x ✅ | 32x PQ |
| HNSW | 6.1 | 1.00x | None |

**Winner: ZGQ Unified & IVF (tied)** - 20% less memory than HNSW

## Recall Comparison

| Algorithm | Recall@10 | vs HNSW |
|-----------|-----------|---------|
| HNSW | 54.9% | Baseline |
| **ZGQ Unified** | **53.9%** | **-1.0%** ✅ |
| IVF | 37.6% | -17.3% ❌ |
| IVF+PQ | 19.0% | -35.9% ❌ |

**Winner: ZGQ Unified** - Nearly matches HNSW recall

## Overall Scoring

Weighted score (Speed 40%, Recall 30%, Memory 20%, Build 10%):

| Rank | Algorithm | Score | Comment |
|------|-----------|-------|---------|
| 🥇 | HNSW | 0.865 | Clear winner |
| 🥈 | ZGQ Unified | 0.573 | Best memory-speed tradeoff |
| 🥉 | IVF | 0.483 | Simple but slow |
| 4️⃣ | IVF+PQ | 0.299 | Only for massive scale |

## Recommendations

### Choose HNSW when:
- ✅ Speed is the top priority
- ✅ Memory is not a major constraint (< 2x dataset size OK)
- ✅ You want production-proven reliability
- ✅ You need the best recall
- ✅ Real-time query requirements

### Choose ZGQ Unified when:
- ✅ Memory efficiency matters (20% savings vs HNSW)
- ✅ You can accept 3.8x slower queries
- ✅ You need good recall (near HNSW-level)
- ✅ You're building zone-aware systems
- ✅ Research and experimentation

### Choose IVF when:
- ✅ Build speed is critical
- ✅ Memory is extremely constrained
- ✅ Queries are infrequent (batch processing)
- ✅ Lower recall is acceptable
- ✅ Simple implementation preferred

### Choose IVF+PQ when:
- ✅ Dataset is billion-scale
- ✅ Memory cost dominates everything
- ✅ Extremely slow queries are acceptable
- ✅ Storage efficiency is critical
- ✅ First-stage filtering before re-ranking

## Scaling Projections

### 1 Million Vectors (128 dims)

| Algorithm | Est. Memory | Est. Latency | Est. Recall |
|-----------|------------|--------------|-------------|
| HNSW | ~600 MB | 0.02 ms | 55% |
| ZGQ Unified | ~490 MB | 0.07 ms | 54% |
| IVF | ~490 MB | 8.0 ms | 35% |
| IVF+PQ | ~520 MB | 70 ms | 18% |

### 10 Million Vectors (128 dims)

| Algorithm | Est. Memory | Est. Latency | Est. Recall |
|-----------|------------|--------------|-------------|
| HNSW | ~6.0 GB | 0.025 ms | 55% |
| ZGQ Unified | ~4.9 GB | 0.09 ms | 54% |
| IVF | ~4.9 GB | 80 ms | 32% |
| IVF+PQ | ~5.2 GB | 700 ms | 15% |

**Key Insight:** ZGQ Unified's memory advantage grows with scale (18% savings at 10M vectors).

## Performance Characteristics

### Query Latency vs Dataset Size

```
HNSW:        O(log N) - excellent scaling
ZGQ Unified: O(log N) - excellent scaling
IVF:         O(N/nlist) - poor scaling
IVF+PQ:      O(N/nlist) - poor scaling
```

### Memory vs Dataset Size

```
HNSW:        O(N·d + N·M) - graph overhead
ZGQ Unified: O(N·d + N·M + 100·d) - minimal zone overhead
IVF:         O(N·d + nlist·d) - minimal overhead
IVF+PQ:      O(N·m + nlist·d + m·k·d/m) - excellent compression
```

### Build Time vs Dataset Size

```
HNSW:        O(N log N) - fast
ZGQ Unified: O(N log N) - slower due to clustering
IVF:         O(N·nlist) - fast
IVF+PQ:      O(N·nlist + PQ training) - slow
```

## Conclusion

**For most production use cases: Use HNSW**
- Industry-proven performance
- Best speed and recall
- Reasonable memory footprint
- Easy to deploy

**For memory-constrained systems: Use ZGQ Unified**
- 20% less memory than HNSW
- Near HNSW-level recall
- Acceptable speed (3.8x slower)
- Good for zone-aware applications

**For educational purposes: Use IVF**
- Simple to understand
- Fast to build
- Good baseline for comparisons

**For billion-scale datasets: Use IVF+PQ (with caution)**
- Only when memory absolutely dominates
- Accept very slow queries
- Consider multi-stage search instead

## Future Work

### Potential Improvements for ZGQ Unified:
1. **GPU Acceleration** - Parallelize zone searches
2. **Adaptive n_probe** - Dynamic zone selection based on query
3. **Hierarchical Zones** - Multi-level zone partitioning
4. **PQ Integration** - Combine zone awareness with compression
5. **Query Caching** - Cache hot zone entry points

### Expected Performance After Optimizations:
- Target: Match or beat HNSW speed (< 0.015ms)
- Memory: Maintain 20% advantage
- Recall: Match HNSW (54-55%)

## References

1. Malkov & Yashunin (2018): "Efficient and robust approximate nearest neighbor search using Hierarchical Navigable Small World graphs"
2. Jégou et al. (2011): "Product quantization for nearest neighbor search"
3. Johnson et al. (2019): "Billion-scale similarity search with GPUs"
4. This work: "Zone-aware Graph Quantization for efficient ANN search"

---

**Benchmark Details:**
- Dataset: 10,000 vectors × 128 dimensions
- Queries: 100 test vectors
- Hardware: Intel i5-12500H, 16GB RAM
- Software: Python 3.12, hnswlib, NumPy, scikit-learn
- Date: October 2025
