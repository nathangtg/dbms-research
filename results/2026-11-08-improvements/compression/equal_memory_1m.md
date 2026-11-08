# Equal-memory comparison, 1m (1,000,000 vectors, dim 128)

Bytes/vector is the analytic model (`zgq.memory_model`), counting everything the index needs to answer queries. `max recall` is the best point in the sweep. `ms @ R>=x` is the lowest median latency among sweep points reaching recall x (`-` = never reached).

## Band A: 60-120 B/vec

| index | B/vec | max recall % | ms at max | ms @ R>=70 | ms @ R>=90 | ms @ R>=95 |
|---|---:|---:|---:|---:|---:|---:|
| IVFPQR-pq32-r64 | 105.3 | 85.7 | 0.479 | 0.083 | - | - |
| IVFPQ-pq64 | 73.2 | 82.7 | 0.971 | 0.329 | - | - |
| IVFPQR-pq32-r32 | 73.3 | 79.5 | 0.484 | 0.206 | - | - |
| IVFPQR-pq16-r64 | 89.3 | 56.8 | 0.295 | - | - | - |
| HNSWPQ-M8-pq32 | 112.7 | 14.1 | 0.466 | - | - | - |
| ZGQ-C-RPQ-M8-pq16 | 99.4 | 12.2 | 0.287 | - | - | - |
| ZGQ-C-PQ-M8-pq16 | 80.7 | 3.4 | 0.253 | - | - | - |
| HNSWPQ-M8-pq16 | 96.7 | 2.8 | 0.373 | - | - | - |

## Band B: 120-200 B/vec

| index | B/vec | max recall % | ms at max | ms @ R>=70 | ms @ R>=90 | ms @ R>=95 |
|---|---:|---:|---:|---:|---:|---:|
| ZGQ-C-RPQ-M16-pq32 | 196.2 | 33.7 | 0.411 | - | - | - |
| ZGQ-C-RPQ-M8-pq32 | 132.2 | 32.7 | 0.354 | - | - | - |
| ZGQ-C-PQ-M16-pq32 | 160.7 | 16.2 | 0.343 | - | - | - |
| HNSWPQ-M16-pq32 | 176.4 | 14.7 | 0.452 | - | - | - |
| ZGQ-C-RPQ-M16-pq16 | 163.4 | 12.4 | 0.109 | - | - | - |
| HNSWPQ-M16-pq16 | 160.4 | 2.6 | 0.364 | - | - | - |

## Band C: 200-300 B/vec

| index | B/vec | max recall % | ms at max | ms @ R>=70 | ms @ R>=90 | ms @ R>=95 |
|---|---:|---:|---:|---:|---:|---:|
| HNSWSQ8-M16 | 272.3 | 86.9 | 0.452 | 0.087 | - | - |
| HNSWSQ8-M8 | 208.6 | 80.3 | 0.478 | 0.243 | - | - |

## Band ref: >=500 B/vec

| index | B/vec | max recall % | ms at max | ms @ R>=70 | ms @ R>=90 | ms @ R>=95 |
|---|---:|---:|---:|---:|---:|---:|
| HNSW-M16 | 660.5 | 91.3 | 0.314 | 0.122 | 0.314 | - |
| ZGQ-C-RPQ-M16-pq32+rerank50 | 708.2 | 66.8 | 0.420 | - | - | - |
| ZGQ-C-RPQ-M8-pq16+rerank50 | 611.4 | 32.1 | 0.296 | - | - | - |

## Model cross-check

| index | model B/vec | measured/serialized B/vec | ratio |
|---|---:|---:|---:|
| HNSW-M16 | 660.5 | 660.5 | 1.000 |
| ZGQ-C-RPQ-M8-pq16 | 99.4 | 99.4 | 1.000 |
| ZGQ-C-RPQ-M8-pq16+rerank50 | 611.4 | 611.4 | 1.000 |
| ZGQ-C-RPQ-M8-pq32 | 132.2 | 132.2 | 1.000 |
| ZGQ-C-RPQ-M16-pq16 | 163.4 | 163.4 | 1.000 |
| ZGQ-C-RPQ-M16-pq32 | 196.2 | 196.2 | 1.000 |
| ZGQ-C-RPQ-M16-pq32+rerank50 | 708.2 | 708.2 | 1.000 |
| ZGQ-C-PQ-M16-pq32 | 160.7 | 160.7 | 1.000 |
| ZGQ-C-PQ-M8-pq16 | 80.7 | 80.7 | 1.000 |
| HNSWPQ-M8-pq16 | 96.7 | 96.7 | 1.000 |
| HNSWPQ-M8-pq32 | 112.7 | 112.7 | 1.000 |
| HNSWPQ-M16-pq16 | 160.4 | 160.4 | 1.000 |
| HNSWPQ-M16-pq32 | 176.4 | 176.4 | 1.000 |
| HNSWSQ8-M8 | 208.6 | 208.6 | 1.000 |
| HNSWSQ8-M16 | 272.3 | 272.3 | 1.000 |
| IVFPQ-pq16 | 25.2 | 25.2 | 1.001 |
| IVFPQ-pq32 | 41.2 | 41.2 | 1.000 |
| IVFPQ-pq64 | 73.2 | 73.2 | 1.000 |
| IVFPQR-pq16-r64 | 89.3 | 89.3 | 1.000 |
| IVFPQR-pq32-r32 | 73.3 | 73.3 | 1.000 |
| IVFPQR-pq32-r64 | 105.3 | 105.3 | 1.000 |
