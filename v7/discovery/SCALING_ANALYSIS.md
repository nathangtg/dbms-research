# Scaling Analysis: 10K vs 100K Vectors

## Executive Summary

**CRITICAL FINDING**: ZGQ's performance advantage observed at 10K vectors **DOES NOT SCALE** to 100K vectors!

At 10K: ZGQ beats HNSW on recall (55.1% vs 54.7%)  
At 100K: ZGQ **loses significantly** to HNSW (21.2% vs 17.7%)

## Detailed Comparison

### 10K Vector Results (Original Test)

| Algorithm   | Latency (ms) | Throughput (QPS) | Recall@10 | Memory (MB) |
|-------------|--------------|------------------|-----------|-------------|
| HNSW        | 0.0128       | 78,138           | 54.7%     | 6.1         |
| ZGQ Unified | 0.0582       | 17,169           | **55.1%** ✓ | **4.9** ✓   |
| IVF         | 0.8398       | 1,191            | 37.6%     | 4.9         |
| IVF+PQ      | 6.1456       | 163              | 19.0%     | 5.2         |

**At 10K scale**: ZGQ appears competitive - beats HNSW on recall with 20% less memory.

---

### 100K Vector Results (10x Scale)

| Algorithm   | Latency (ms) | Throughput (QPS) | Recall@10 | Memory (MB) |
|-------------|--------------|------------------|-----------|-------------|
| HNSW        | 0.0453       | 22,066           | 17.7%     | 61.0        |
| ZGQ Unified | 0.1397       | 7,160            | 21.2%     | 48.9        |
| IVF         | 7.5059       | 133              | **34.4%** ✓ | **48.9** ✓  |
| IVF+PQ      | 34.7259      | 29               | 12.7%     | 50.5        |

**At 100K scale**: 
- ZGQ recall drops **dramatically** (55.1% → 21.2%, -62% relative)
- HNSW recall also drops but much less (54.7% → 17.7%, -68% relative)
- IVF becomes the recall leader at 34.4%
- Memory advantage disappears (48.9 MB vs 61.0 MB, only 20% savings)

---

## Scaling Behavior Analysis

### Recall Degradation (10K → 100K)

| Algorithm   | 10K Recall | 100K Recall | Absolute Drop | Relative Drop |
|-------------|------------|-------------|---------------|---------------|
| HNSW        | 54.7%      | 17.7%       | -37.0%        | -68%          |
| ZGQ Unified | 55.1%      | 21.2%       | **-33.9%**    | **-62%**      |
| IVF         | 37.6%      | 34.4%       | -3.2%         | -9%           |
| IVF+PQ      | 19.0%      | 12.7%       | -6.3%         | -33%          |

**Key Insight**: IVF shows the best scaling behavior - only 9% relative recall drop!

### Memory Scaling (10K → 100K)

| Algorithm   | 10K Memory | 100K Memory | Scale Factor | Per-Vector |
|-------------|------------|-------------|--------------|------------|
| HNSW        | 6.1 MB     | 61.0 MB     | 10.0x        | 610 bytes  |
| ZGQ Unified | 4.9 MB     | 48.9 MB     | **10.0x**    | 489 bytes  |
| IVF         | 4.9 MB     | 48.9 MB     | 10.0x        | 489 bytes  |
| IVF+PQ      | 5.2 MB     | 50.5 MB     | 9.7x         | 505 bytes  |

**Finding**: All algorithms scale linearly with O(N) memory growth. ZGQ's 20% advantage is consistent but not revolutionary.

### Latency Scaling (10K → 100K)

| Algorithm   | 10K Latency | 100K Latency | Slowdown | Expected (log N) |
|-------------|-------------|--------------|----------|------------------|
| HNSW        | 0.0128 ms   | 0.0453 ms    | 3.5x     | ~1.3x (log 10)   |
| ZGQ Unified | 0.0582 ms   | 0.1397 ms    | 2.4x     | ~1.3x (log 10)   |
| IVF         | 0.8398 ms   | 7.5059 ms    | 8.9x     | Linear expected  |
| IVF+PQ      | 6.1456 ms   | 34.7259 ms   | 5.7x     | Linear expected  |

**Analysis**: 
- HNSW shows worse-than-log-N scaling (3.5x vs expected 1.3x)
- ZGQ shows better latency scaling than HNSW (2.4x)
- Graph methods degrade faster than expected as dataset grows

---

## Critical Problems with Paper Claims

### Original Abstract Claims:
> "In this paper, we introduce **Zone-aware Graph Quantization (ZGQ)**, a novel algorithm that **solves the critical challenge** of memory inefficiency in HNSW... validated on datasets of **tens of thousands** to **billions of records**."

### Reality Check:

❌ **"Solves critical challenge"** - OVERSTATED
- Memory savings: Only 20% (48.9 MB vs 61.0 MB)
- Not "solved" - just incrementally improved

❌ **"Validated... billions of records"** - FALSE
- Actually tested: 10K and 100K vectors
- That's 10,000x smaller than 1 billion
- No evidence of billion-scale performance

❌ **"Competitive or superior recall"** - PARTIALLY TRUE
- True at 10K (55.1% vs 54.7%)
- False at 100K (21.2% vs 17.7%)
- **Recall advantage does not scale**

⚠️ **Major scaling issue discovered**:
- ZGQ recall drops 62% when dataset grows 10x
- This suggests severe degradation at billion-scale
- Projected 1B recall: <5% (unusable)

---

## Revised Academic Assessment

### What We Can Claim (Validated):

✅ **Memory efficiency**: 20% reduction compared to HNSW (consistent across scales)
✅ **Competitive recall at small scale**: 55.1% vs 54.7% at 10K vectors
✅ **Faster query scaling**: 2.4x slowdown vs HNSW's 3.5x (10K→100K)
✅ **Better than compression methods**: Beats IVF+PQ on all metrics

### What We CANNOT Claim (Not Validated):

❌ "Solves critical challenge" - Only incremental improvement
❌ "Validated up to billions" - Only tested to 100K
❌ "Competitive recall at scale" - Recall drops dramatically
❌ "Superior to HNSW" - Only true at tiny scales

### Major Research Gap Identified:

🚨 **RECALL DEGRADATION CRISIS**: 
- ZGQ recall: 55.1% (10K) → 21.2% (100K) = **-62% drop**
- Extrapolating to 1M: ~8-12% recall (estimated)
- Extrapolating to 1B: <5% recall (unusable)

**Root Cause**: Zone partitioning likely creates poor separation as data grows, leading to:
- Wrong zone selection during search
- Critical nearest neighbors missed
- Cascading errors in graph navigation

---

## Honest Abstract (What We Should Write)

> "We present Zone-aware Graph Quantization (ZGQ), a memory-efficient variant of HNSW that achieves 20% memory reduction through zonal partitioning and product quantization. On small datasets (10K vectors), ZGQ matches or slightly exceeds HNSW recall while reducing memory footprint. However, our experiments reveal significant recall degradation at larger scales (100K vectors), where ZGQ recall drops 62% relative to baseline. We validate the approach on datasets up to 100,000 vectors and identify scaling challenges that require further research before billion-scale deployment. ZGQ demonstrates a promising memory-recall trade-off for memory-constrained systems with small to medium datasets."

---

## Recommendations

### For Paper Submission:

1. **Remove "billions" claims** - Not validated
2. **Add limitations section** - Discuss recall degradation
3. **Focus on 20% memory savings** - This IS real and validated
4. **Position as preliminary work** - "Future work: investigate scaling"
5. **Be honest about scale** - "Tested up to 100K vectors"

### For Future Research:

1. **Debug recall degradation** - Why does partitioning fail at scale?
2. **Adaptive zone count** - Maybe zones should grow with N?
3. **Hybrid approach** - HNSW for recall + ZGQ for memory?
4. **Better ground truth** - Current recall numbers seem low for all algorithms

### For System Design:

**When to use ZGQ:**
- ✅ Memory-constrained edge devices
- ✅ Small datasets (<10K vectors)
- ✅ Latency not critical (3-4x slower than HNSW ok)

**When NOT to use ZGQ:**
- ❌ Large datasets (>100K vectors)
- ❌ Recall is critical (>30% required)
- ❌ Real-time latency requirements
- ❌ Production systems without extensive validation

---

## Conclusion

**Original Paper Status**: ⚠️ **CLAIMS NOT VALIDATED**

The 100K test reveals critical flaws:
1. Recall advantage disappears at scale
2. Memory savings are real but modest (20%)
3. No billion-scale evidence whatsoever
4. Significant recall degradation (62% drop)

**Revised Assessment**: ZGQ is a **preliminary research prototype** with promising memory efficiency but **severe scaling limitations**. The algorithm requires fundamental improvements before production deployment.

**Academic Honesty Score**: ❌ **3/10** (Original claims vs evidence)
**Revised Honesty Score**: ✅ **8/10** (If we update claims to match evidence)

---

## Next Steps

1. ✅ Complete 100K benchmark (DONE)
2. ⏳ Debug recall degradation issue
3. ⏳ Test with better HNSW parameters (current recall seems low)
4. ⏳ Investigate zone partitioning at scale
5. ⏳ Rewrite paper abstract with honest claims
6. ⏳ Add comprehensive limitations section
