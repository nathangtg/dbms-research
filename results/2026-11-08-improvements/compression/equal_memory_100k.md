# Equal-memory comparison, 100k (100,000 vectors, dim 128)

Bytes/vector is the analytic model (`zgq.memory_model`), counting everything the index needs to answer queries. `max recall` is the best point in the sweep. `ms @ R>=x` is the lowest median latency among sweep points reaching recall x (`-` = never reached).

## Band A: 60-120 B/vec

| index | B/vec | max recall % | ms at max | ms @ R>=70 | ms @ R>=90 | ms @ R>=95 |
|---|---:|---:|---:|---:|---:|---:|
| IVFPQR-pq32-r64 | 109.9 | 92.9 | 0.148 | 0.039 | 0.059 | - |
| IVFPQ-pq64 | 76.5 | 87.3 | 0.302 | 0.034 | - | - |
| IVFPQR-pq32-r32 | 77.9 | 86.1 | 0.147 | 0.037 | - | - |
| IVFPQR-pq16-r64 | 93.9 | 65.6 | 0.089 | - | - | - |
| HNSWPQ-M8-pq32 | 113.9 | 18.8 | 0.276 | - | - | - |
| ZGQ-C-PQ-M8-pq16 | 82.9 | 5.5 | 0.022 | - | - | - |
| HNSWPQ-M8-pq16 | 97.9 | 4.3 | 0.206 | - | - | - |

## Band B: 120-200 B/vec

| index | B/vec | max recall % | ms at max | ms @ R>=70 | ms @ R>=90 | ms @ R>=95 |
|---|---:|---:|---:|---:|---:|---:|
| ZGQ-C-RPQ-M16-pq16 | 199.0 | 32.2 | 0.171 | - | - | - |
| ZGQ-C-RPQ-M8-pq16 | 135.0 | 32.1 | 0.144 | - | - | - |
| HNSWPQ-M16-pq32 | 177.6 | 19.1 | 0.298 | - | - | - |
| ZGQ-C-PQ-M16-pq32 | 162.9 | 19.0 | 0.178 | - | - | - |
| HNSWPQ-M16-pq16 | 161.6 | 4.7 | 0.216 | - | - | - |

## Band C: 200-300 B/vec

| index | B/vec | max recall % | ms at max | ms @ R>=70 | ms @ R>=90 | ms @ R>=95 |
|---|---:|---:|---:|---:|---:|---:|
| HNSWSQ8-M16 | 272.3 | 93.5 | 0.341 | 0.027 | 0.341 | - |
| HNSWSQ8-M8 | 208.6 | 90.0 | 0.317 | 0.037 | 0.317 | - |
| ZGQ-C-RPQ-M16-pq32 | 265.2 | 60.7 | 0.206 | - | - | - |
| ZGQ-C-RPQ-M8-pq32 | 201.2 | 60.1 | 0.180 | - | - | - |

## Band ref: >=500 B/vec

| index | B/vec | max recall % | ms at max | ms @ R>=70 | ms @ R>=90 | ms @ R>=95 |
|---|---:|---:|---:|---:|---:|---:|
| HNSW-M16 | 660.5 | 96.7 | 0.209 | 0.050 | 0.150 | 0.209 |
| ZGQ-C-RPQ-M16-pq32+rerank50 | 777.2 | 95.7 | 0.216 | 0.075 | 0.100 | 0.216 |
| ZGQ-C-RPQ-M8-pq16+rerank50 | 647.0 | 69.0 | 0.154 | - | - | - |

## Model cross-check

| index | model B/vec | measured/serialized B/vec | ratio |
|---|---:|---:|---:|
| HNSW-M16 | 660.5 | 660.5 | 1.000 |
| ZGQ-C-RPQ-M8-pq16 | 135.0 | 135.0 | 1.000 |
| ZGQ-C-RPQ-M8-pq16+rerank50 | 647.0 | 647.0 | 1.000 |
| ZGQ-C-RPQ-M8-pq32 | 201.2 | 201.2 | 1.000 |
| ZGQ-C-RPQ-M16-pq16 | 199.0 | 199.0 | 1.000 |
| ZGQ-C-RPQ-M16-pq32 | 265.2 | 265.2 | 1.000 |
| ZGQ-C-RPQ-M16-pq32+rerank50 | 777.2 | 777.2 | 1.000 |
| ZGQ-C-PQ-M16-pq32 | 162.9 | 162.9 | 1.000 |
| ZGQ-C-PQ-M8-pq16 | 82.9 | 82.9 | 1.000 |
| HNSWPQ-M8-pq16 | 97.9 | 97.9 | 1.000 |
| HNSWPQ-M8-pq32 | 113.9 | 113.9 | 1.000 |
| HNSWPQ-M16-pq16 | 161.6 | 161.6 | 1.000 |
| HNSWPQ-M16-pq32 | 177.6 | 177.6 | 1.000 |
| HNSWSQ8-M8 | 208.6 | 208.6 | 1.000 |
| HNSWSQ8-M16 | 272.3 | 272.3 | 1.000 |
| IVFPQ-pq16 | 28.5 | 28.6 | 1.002 |
| IVFPQ-pq32 | 44.5 | 44.6 | 1.001 |
| IVFPQ-pq64 | 76.5 | 76.6 | 1.001 |
| IVFPQR-pq16-r64 | 93.9 | 93.9 | 1.001 |
| IVFPQR-pq32-r32 | 77.9 | 77.9 | 1.001 |
| IVFPQR-pq32-r64 | 109.9 | 109.9 | 1.000 |
