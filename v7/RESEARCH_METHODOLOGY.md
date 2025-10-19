# Research Methodology

## Introduction to the Methodology

In order to solve the memory-performance trade-off problem in billion-scale Approximate Nearest Neighbor Search (ANNS) systems, the Zonal Graph Quantization (ZGQ) solution was designed and evaluated using the methodological framework described in this section. This methodology's main objective is to compare ZGQ to its predecessors, which are well-known algorithms, in order to quantitatively demonstrate its benefits in balancing search performance, construction cost, and memory efficiency.

---

## Research Design

A quantitative experimental study design is used in the research. It is specifically a comparative study with the goal of comparing the suggested ZGQ framework's performance metrics to two main baseline indexing models:

1. **Hierarchical Navigable Small World (HNSW)**: A pure graph-based index.
2. **Inverted File (IVF)**: A pure partitioning-based index, usually enhanced for efficiency with Product Quantization (PQ) (IVF-PQ).

---

## Steps Involved

### A. Identifying Research Objectives

The main objective of our applied methodology is to discover and observe the actual performance of our solution when applied to high-dimensional vector datasets. The ZGQ hybrid indexing approach should offer a better balance between memory efficiency and search performance (Recall@k and Latency) than its monolithic predecessors such as HNSW and conventional IVF-PQ.

### B. Literature Review and Analysis

The current limitations and research gap were determined by a thorough examination of the existing ANNS algorithms (graph-based, quantization-based, and hybrid approaches) obtained through a structured Literature Review (Section II). The architectural design of ZGQ was directly influenced by this analysis, which inspired our combination of localized HNSW graphs and IVF-style partitioning to lessen the construction cost and monolithic memory overhead associated with full HNSW indices.

### C. Solution Design - Zonal Graph Quantization (ZGQ)

Our proposed solution, Zonal Graph Quantization (ZGQ), is a hybrid in-memory indexing solution which differs from methods such as Zonal HNSW by combining a quantization step and tight optimization design for memory-performance trade-offs. The architecture consists of the following components:

1. **Partitioning**: Uses K-means clustering to divide vector space into Z non-overlapping zones (clusters), mimicking the IVF approach.

2. **Local Indexing**: Involves constructing independent and smaller HNSW graphs within each of the Z zones.

3. **Quantization**: Applies optimal vector quantization (which may include product quantization or residual quantization) to further compress the index size.

4. **Search**: Queries the nearest n_probe zone centroids while running parallel lightweight HNSW searches within the identified zones.

---

## Evaluation Metrics

The evaluation will compare our proposed solution, Zonal Graph Quantization (ZGQ), against established algorithms such as HNSW (Graph Baseline) and IVF-PQ (Partitioning Baseline) based on the following quantitative metrics:

| Metric | Units | Goal | Relation to Problem |
|--------|-------|------|---------------------|
| **Recall@k** | Accuracy / Hit Rate (%) | Maximize | Measures search accuracy and quality |
| **Query Latency** | Time (milliseconds) | Minimize | Measures query performance and efficiency |
| **Throughput** | Speed (Queries Per Second) | Maximize | Measures system scalability |
| **Index Size** | Memory Footprint (MB/GB) | Minimize | Directly addresses the HNSW memory overhead problem |
| **Index Build Time** | Time (Seconds) | Minimize | Directly addresses the HNSW construction cost problem |

---

## Data Collection and Experimental Setup

### D. Dataset Generation

The evaluation of our proposed solution, namely the Zonal Graph Quantization (ZGQ), will utilize synthetic datasets with real-world characteristics to ensure that the validation is comprehensive and controlled with realistic conditions.

#### Synthetic Data Generation

Controlled and artificial datasets are going to be created using a custom Python script (`generate_test_data.py`). This script allows the systematic variation of key parameters revolving around the datasets:

- **Number of Vectors (n_vectors)**: Experiments are conducted at different scales of datasets, ranging from 10,000 vectors with an upwards trajectory to 100,000 vectors or more, which can be used to analyze the scalability of the solution.

- **Vector Dimensions (dim)**: A standard vector dimension of 128 is used to simulate common embedding sizes (e.g., BERT, word2vec).

- **Number of Queries (n_queries)**: A fixed set of query vectors (100 or 1,000) is generated for consistent evaluation across different algorithms.

The generation process for all data follows a standard normal distribution (`numpy.random.randn`), followed by L2 normalization, ensuring that all vectors lie on a hypersphere—a common ANNS benchmark practice. Ground truth results are computed using brute-force search to ensure 100% accuracy for validation purposes.

### E. Experimental Environment

The device used to demonstrate and run the experiments to obtain our current figures follows the specification of:

- **Hardware**: 
  - CPU: Intel Core i5-12500H (12 cores, 16 threads)
  - GPU: NVIDIA RTX 3050 (4GB VRAM)
  - RAM: 32 GB DDR4
  - Storage: 512 GB NVMe SSD

- **Operating System**: WSL2 Ubuntu 24.04 under Windows 11

- **Execution Configuration**:
  - Index construction utilizes multithreading capabilities to use the available CPU cores to measure practical build times
  - All experiments are run with consistent system settings to ensure reproducibility
  - Background processes are minimized to reduce interference

---

## Implementation Details

### F. Implementation of Solutions

#### Baseline Implementations

To ensure fair and accurate comparisons, we leverage well-established, optimized library implementations for the baseline algorithms:

1. **HNSW Baseline**:
   - **Library**: `hnswlib` (version 0.8.0 or later)
   - **Justification**: hnswlib is the reference implementation of HNSW, providing highly optimized C++ code with Python bindings
   - **Configuration**: 
     - M=16 (number of bidirectional links per node)
     - ef_construction=200 (controls index construction quality)
     - ef_search=50 (controls search quality, varied in experiments)

2. **IVF and IVF-PQ Baseline**:
   - **Library**: Custom implementation using `scikit-learn` for K-Means clustering
   - **Components**:
     - K-Means clustering: `sklearn.cluster.KMeans` (version 1.3.0+)
     - Product Quantization: Custom implementation based on standard PQ algorithms
   - **Configuration**:
     - nlist=100 (number of partitions/clusters)
     - nprobe=10 (number of clusters to search, varied in experiments)
     - For IVF-PQ: m=16 subspaces, 8 bits per subspace

#### ZGQ Implementation

Our proposed Zonal Graph Quantization (ZGQ) is implemented by combining the following components:

1. **Partitioning Module**:
   - **Library**: `scikit-learn.cluster.KMeans` (version 1.3.0+)
   - **Purpose**: Divides the vector space into Z non-overlapping zones
   - **Configuration**: Z=4 zones (default, tunable)

2. **Local Graph Construction**:
   - **Library**: `hnswlib` (version 0.8.0+)
   - **Purpose**: Constructs independent HNSW graphs within each zone
   - **Configuration**: Reduced M=8 (vs M=16 for full HNSW) to decrease memory

3. **Quantization Module**:
   - **Implementation**: Custom Product Quantization (PQ) module
   - **Purpose**: Compresses vector representations within each zone
   - **Configuration**: m=16 subspaces, 8 bits per subspace

4. **Search Orchestration**:
   - **Implementation**: Custom Python module (`zgq_index.py`)
   - **Purpose**: Coordinates zone selection and parallel graph search
   - **Process**: 
     1. Compute query-to-centroid distances
     2. Select top n_probe zones
     3. Perform parallel HNSW search in selected zones
     4. Merge and rank results

### G. Tools and Technologies Used

The implementation leverages the following software stack:

#### Core Libraries

```
numpy==1.24.3              # Numerical computations and array operations
scikit-learn==1.3.0        # K-Means clustering for partitioning
hnswlib==0.8.0             # HNSW graph construction and search
numba==0.58.0              # JIT compilation for performance optimization
```

#### Visualization and Analysis

```
matplotlib==3.7.2          # Plotting and visualization
seaborn==0.12.2            # Statistical visualizations
pandas==2.0.3              # Data manipulation and analysis
```

#### Development Tools

```
pytest==7.4.0              # Unit testing framework
jupyterlab==4.0.5          # Interactive development environment
```

#### Version Control

- **Git**: Version 2.42.0
- **Repository**: GitHub (private repository for development)

All dependencies are managed via `requirements.txt` to ensure reproducibility across different environments.

---

## Evaluation Process

### H. Experimental Procedure

The evaluation follows a systematic process to ensure comprehensive and reproducible results:

#### 1. Index Construction Phase

For each algorithm (HNSW, IVF, IVF-PQ, ZGQ) and dataset size:

- **Build the index** from the training vectors
- **Measure and record**:
  - Build time (seconds)
  - Index size (MB)
  - Peak memory usage during construction

#### 2. Query Execution Phase

For each constructed index:

- Load a **fixed set of query vectors** (100-1000 queries)
- **Warm-up phase**: Execute 10 warm-up queries to initialize caches
- **Measurement phase**:
  - Execute all queries sequentially
  - Record latency for each query (milliseconds)
  - Aggregate to compute:
    - Average query latency
    - Throughput (Queries Per Second)
    - 95th percentile latency

#### 3. Parameter Sweeping

To generate performance curves, key search parameters are systematically varied:

- **HNSW**: `ef_search` ∈ {10, 20, 50, 100, 200, 400}
- **IVF/ZGQ**: `n_probe` ∈ {1, 2, 5, 10, 20, 50}

For each parameter setting:
- Run the query execution phase
- Record all metrics
- Plot **Recall@10 vs Query Latency** curves

#### 4. Recall Computation

For each query result set:

- Compare predicted neighbors against **ground truth** (computed via brute-force)
- Compute **Recall@k**: 
  ```
  Recall@k = (Number of correct neighbors in top-k) / k
  ```
- Average across all queries to get overall Recall@k

#### 5. Comparative Analysis

Performance is compared across algorithms at specific recall targets:

- **Target Recall Levels**: 50%, 70%, 90%, 95%
- **Comparison Metrics**:
  - Query latency required to achieve target recall
  - Index size at optimal parameter settings
  - Build time comparison
  - Memory-recall trade-off curves

### I. Validation of Results

To ensure the reliability and validity of experimental results:

1. **Baseline Validation**:
   - Compare HNSW and IVF performance against published results from [ann-benchmarks.com](http://ann-benchmarks.com)
   - Verify that our baseline implementations achieve similar performance characteristics
   - Expected HNSW Recall@10 on synthetic data: 70-90%

2. **Ground Truth Verification**:
   - All ground truth results are computed using brute-force search (100% accurate)
   - Random sampling of query results is manually verified against ground truth

3. **Reproducibility**:
   - All experiments use fixed random seeds (`seed=42`)
   - Results are repeated 3 times and averaged to account for system variance
   - Standard deviation is reported for key metrics

4. **Statistical Significance**:
   - Performance differences are evaluated using paired t-tests
   - Significance level: α = 0.05
   - Only statistically significant improvements are claimed

---

## Methodology Workflow

### J. Visual Flowchart

The following diagram illustrates the complete experimental methodology workflow:

```
┌─────────────────────────────────────────────────────────────────┐
│                    RESEARCH METHODOLOGY                          │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│  PHASE 1: DATA PREPARATION                                      │
│  ┌──────────────────────────────────────────────────────────┐  │
│  │ 1. Generate Synthetic Data (generate_test_data.py)      │  │
│  │    • n_vectors: 10K, 100K, 1M                           │  │
│  │    • dim: 128                                            │  │
│  │    • Normalize vectors (L2)                              │  │
│  │ 2. Compute Ground Truth (Brute Force)                   │  │
│  │    • Recall@10 reference                                 │  │
│  └──────────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│  PHASE 2: INDEX CONSTRUCTION                                    │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐            │
│  │   HNSW      │  │   IVF/IVF-PQ│  │   ZGQ       │            │
│  │  (hnswlib)  │  │  (sklearn)  │  │  (hybrid)   │            │
│  └──────┬──────┘  └──────┬──────┘  └──────┬──────┘            │
│         │                │                │                     │
│         └────────────────┴────────────────┘                     │
│                          ↓                                       │
│         Record: Build Time, Index Size, Memory                  │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│  PHASE 3: QUERY EXECUTION                                       │
│  ┌──────────────────────────────────────────────────────────┐  │
│  │ For each algorithm and parameter setting:               │  │
│  │  1. Warm-up: 10 queries                                 │  │
│  │  2. Execute: 100-1000 queries                           │  │
│  │  3. Record: Latency, Throughput                         │  │
│  │  4. Compute: Recall@10 vs Ground Truth                  │  │
│  └──────────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│  PHASE 4: PARAMETER SWEEPING                                    │
│  ┌──────────────────────────────────────────────────────────┐  │
│  │ HNSW: ef_search = {10, 20, 50, 100, 200, 400}          │  │
│  │ IVF/ZGQ: n_probe = {1, 2, 5, 10, 20, 50}               │  │
│  │                                                          │  │
│  │ Generate: Recall-Latency Curves                         │  │
│  └──────────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│  PHASE 5: COMPARATIVE ANALYSIS                                  │
│  ┌──────────────────────────────────────────────────────────┐  │
│  │ Compare at Target Recall: 50%, 70%, 90%, 95%           │  │
│  │                                                          │  │
│  │ Metrics:                                                 │  │
│  │  • Query Latency                                        │  │
│  │  • Index Size (Memory)                                  │  │
│  │  • Build Time                                           │  │
│  │  • Throughput                                           │  │
│  └──────────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│  PHASE 6: VALIDATION & VISUALIZATION                            │
│  ┌──────────────────────────────────────────────────────────┐  │
│  │ 1. Validate against ann-benchmarks.com                  │  │
│  │ 2. Statistical significance testing (t-test)            │  │
│  │ 3. Generate publication-quality figures                 │  │
│  │ 4. Document findings and limitations                    │  │
│  └──────────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────────┘
```

---

## Challenges and Limitations

### K. Potential Challenges

1. **Synthetic Data Limitations**:
   - **Challenge**: Synthetic random data may not capture the real-world distribution characteristics of actual embeddings
   - **Mitigation**: Normalized vectors on hypersphere mimic common embedding properties; future work will include real-world datasets (SIFT, GIST, DEEP1B)

2. **Hardware Constraints**:
   - **Challenge**: Limited RAM (32GB) prevents testing at true billion-scale (requires ~610GB for HNSW at 1B vectors)
   - **Mitigation**: Test at 10K and 100K scales with linear extrapolation to estimate billion-scale performance

3. **Implementation Optimization Differences**:
   - **Challenge**: HNSW baseline uses highly optimized C++ (hnswlib), while ZGQ components include Python overhead
   - **Mitigation**: Use Numba JIT compilation for performance-critical Python code; report results transparently

4. **Parameter Tuning**:
   - **Challenge**: Optimal parameters differ per algorithm and dataset; exhaustive grid search is computationally expensive
   - **Mitigation**: Use standard parameter ranges from literature; acknowledge that further tuning might improve specific algorithm performance

5. **Recall Degradation at Scale**:
   - **Challenge**: Preliminary results show significant recall degradation when scaling from 10K to 100K vectors (e.g., ZGQ: -62% relative drop)
   - **Mitigation**: Investigate adaptive zone count strategies; report scaling characteristics transparently

### L. Known Limitations

1. **Dataset Diversity**:
   - Experiments use synthetic Gaussian data; generalization to other distributions (clustered, skewed) is not guaranteed

2. **Single-Node Setup**:
   - All experiments are conducted on a single workstation; distributed/parallel implementations are not evaluated

3. **Comparison Scope**:
   - Limited to three algorithm classes (HNSW, IVF-PQ, ZGQ); other advanced methods (NSG, DiskANN, SPANN) are not included

4. **Query Workload**:
   - Single-vector queries only; batch query optimization is not evaluated

5. **Dynamic Updates**:
   - Index construction is offline; dynamic insertion/deletion performance is not measured

---

## Summary

This methodology provides a comprehensive framework for evaluating the proposed Zonal Graph Quantization (ZGQ) algorithm against established baselines (HNSW and IVF-PQ). The approach combines:

- **Controlled synthetic data generation** to enable systematic scalability analysis
- **Well-established library implementations** for fair baseline comparisons  
- **Comprehensive metrics** covering accuracy, speed, memory, and build time
- **Systematic parameter sweeping** to generate performance curves
- **Statistical validation** to ensure result reliability

The evaluation process follows a rigorous experimental protocol:
1. Generate datasets at multiple scales (10K, 100K, 1M vectors)
2. Construct indices for all algorithms, measuring build metrics
3. Execute fixed query sets, recording latency and throughput
4. Compute recall against ground truth
5. Generate Recall-Latency trade-off curves
6. Perform comparative analysis at target recall levels
7. Validate results against published benchmarks

**Key strengths** of this methodology include:
- Use of standard, reproducible libraries (hnswlib, scikit-learn)
- Fixed random seeds and multiple runs for reproducibility
- Transparent reporting of hardware constraints and limitations
- Visual workflow diagram for clarity

**Acknowledged limitations** include synthetic data characteristics, hardware memory constraints, and implementation optimization differences. These limitations are mitigated through normalization techniques, linear scaling projections, and transparent reporting.

This methodology enables a fair, comprehensive, and reproducible evaluation of ZGQ's effectiveness in addressing the memory-performance trade-off in large-scale ANNS systems. The results will provide quantitative evidence to support or refute the hypothesis that hybrid approaches can maintain competitive recall while significantly reducing memory footprint compared to monolithic HNSW indices.

---

**End of Research Methodology Section**
