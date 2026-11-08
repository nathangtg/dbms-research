### 10k (10,000 vectors)

Build time and analytic index bytes vs HNSW-M16:

| Index | Build s | Build ratio | Analytic MB | Bytes ratio |
|---|---:|---:|---:|---:|
| HNSW-M16 | 0.8 | 1.00 | 6.6 | 1.00 |
| ZGQ-M16 | 2.4 | 2.97 | 12.1 | 1.84 |
| ZGQ-M16+global-entry-nopq | 2.4 | 2.92 | 12.1 | 1.84 |
| ZGQ-M16+zonescan-nopq | 2.4 | 2.92 | 12.1 | 1.84 |
| ZGQ-M16+overlap2-nopq | 2.4 | 2.93 | 12.2 | 1.84 |
| ZGQ-M16+adaptive-nopq | 2.4 | 2.92 | 12.1 | 1.84 |
| ZGQ-M16+global-entry | 2.4 | 2.92 | 12.1 | 1.84 |
| ZGQ-M16+zonescan | 2.4 | 2.92 | 12.1 | 1.84 |
| ZGQ-M16+overlap2 | 2.4 | 2.93 | 12.2 | 1.84 |
| ZGQ-M16+adaptive | 2.4 | 2.92 | 12.1 | 1.84 |
| ZGQ-M16+random | 2.4 | 2.95 | 12.1 | 1.84 |
| ZGQ-M16+random+global-entry-nopq | 2.4 | 2.95 | 12.1 | 1.84 |
| ZGQ-M16+random+global-entry | 2.4 | 2.95 | 12.1 | 1.84 |

Equal ef (variants swept over ef), recall delta in points and latency ratio:

| Index | ef=16 | ef=32 | ef=64 | ef=128 | ef=256 |
|---|---|---|---|---|---|
| ZGQ-M16 | +13.1 pt, 1.7x | +0.0 pt, 1.1x | -0.1 pt, 1.1x | -0.9 pt, 1.0x | -0.5 pt, 1.0x |
| ZGQ-M16+global-entry-nopq | +17.8 pt, 10.8x | +3.5 pt, 6.9x | -5.4 pt, 4.3x | -9.8 pt, 2.8x | -11.4 pt, 2.1x |
| ZGQ-M16+global-entry | -15.6 pt, 18.0x | -29.8 pt, 11.7x | -38.7 pt, 7.2x | -43.1 pt, 4.9x | -44.9 pt, 3.4x |
| ZGQ-M16+random | +13.1 pt, 1.7x | -0.2 pt, 1.2x | +0.2 pt, 1.1x | -0.2 pt, 1.1x | +0.0 pt, 1.1x |
| ZGQ-M16+random+global-entry-nopq | +18.2 pt, 10.8x | +3.9 pt, 7.0x | -5.0 pt, 4.4x | -9.4 pt, 3.0x | -10.9 pt, 2.1x |
| ZGQ-M16+random+global-entry | -15.1 pt, 18.2x | -29.4 pt, 11.7x | -38.3 pt, 7.4x | -42.7 pt, 4.8x | -44.5 pt, 3.4x |

HNSW-M16 reference: ef=16 recall 69.2% 0.018 ms, ef=32 recall 83.5% 0.027 ms, ef=64 recall 92.4% 0.044 ms, ef=128 recall 96.8% 0.067 ms, ef=256 recall 98.7% 0.095 ms

Equal recall: latency ms (ratio to HNSW-M16 at the same recall); max recall of each curve in the last column:

| Index | R=80% | R=90% | R=95% | Max recall (latency) |
|---|---|---|---|---|
| HNSW-M16 | 0.025 (1.0x) | 0.039 (1.0x) | 0.057 (1.0x) | 98.7% (0.095 ms) |
| ZGQ-M16 | 0.030 (1.2x) | 0.044 (1.1x) | 0.064 (1.1x) | 98.2% (0.100 ms) |
| ZGQ-M16+global-entry-nopq | 0.189 (7.6x) | not reached | not reached | 87.3% (0.196 ms) |
| ZGQ-M16+zonescan-nopq | 0.111 (4.4x) | 0.140 (3.6x) | 0.154 (2.7x) | 100.0% (0.642 ms) |
| ZGQ-M16+overlap2-nopq | 0.091 (3.6x) | 0.139 (3.5x) | 0.173 (3.0x) | 100.0% (0.752 ms) |
| ZGQ-M16+adaptive-nopq | 0.203 (8.1x) | 0.203 (5.2x) | 0.203 (3.5x) | 100.0% (0.335 ms) |
| ZGQ-M16+global-entry | not reached | not reached | not reached | 53.9% (0.323 ms) |
| ZGQ-M16+zonescan | not reached | not reached | not reached | 40.6% (0.189 ms) |
| ZGQ-M16+overlap2 | not reached | not reached | not reached | 44.3% (0.208 ms) |
| ZGQ-M16+adaptive | not reached | not reached | not reached | 43.5% (0.300 ms) |
| ZGQ-M16+random | 0.030 (1.2x) | 0.044 (1.1x) | 0.062 (1.1x) | 98.8% (0.101 ms) |
| ZGQ-M16+random+global-entry-nopq | 0.189 (7.6x) | not reached | not reached | 87.8% (0.196 ms) |
| ZGQ-M16+random+global-entry | not reached | not reached | not reached | 54.3% (0.326 ms) |

### 100k (100,000 vectors)

Build time and analytic index bytes vs HNSW-M16:

| Index | Build s | Build ratio | Analytic MB | Bytes ratio |
|---|---:|---:|---:|---:|
| HNSW-M16 | 11.4 | 1.00 | 66.1 | 1.00 |
| ZGQ-M16 | 18.6 | 1.64 | 119.9 | 1.82 |
| ZGQ-M16+global-entry-nopq | 18.5 | 1.63 | 119.9 | 1.82 |
| ZGQ-M16+zonescan-nopq | 18.5 | 1.63 | 119.9 | 1.82 |
| ZGQ-M16+overlap2-nopq | 18.5 | 1.63 | 120.3 | 1.82 |
| ZGQ-M16+adaptive-nopq | 18.5 | 1.63 | 119.9 | 1.82 |
| ZGQ-M16+global-entry | 18.5 | 1.63 | 119.9 | 1.82 |
| ZGQ-M16+zonescan | 18.5 | 1.63 | 119.9 | 1.82 |
| ZGQ-M16+overlap2 | 18.5 | 1.63 | 120.3 | 1.82 |
| ZGQ-M16+adaptive | 18.5 | 1.63 | 119.9 | 1.82 |
| ZGQ-M16+random | 20.7 | 1.82 | 119.9 | 1.82 |
| ZGQ-M16+random+global-entry-nopq | 20.7 | 1.82 | 119.9 | 1.82 |
| ZGQ-M16+random+global-entry | 20.7 | 1.82 | 119.9 | 1.82 |

Equal ef (variants swept over ef), recall delta in points and latency ratio:

| Index | ef=16 | ef=32 | ef=64 | ef=128 | ef=256 |
|---|---|---|---|---|---|
| ZGQ-M16 | +6.8 pt, 1.3x | -5.2 pt, 0.9x | -7.1 pt, 0.8x | -8.3 pt, 0.8x | -9.4 pt, 0.8x |
| ZGQ-M16+global-entry-nopq | +20.4 pt, 6.3x | +7.5 pt, 4.1x | -1.9 pt, 2.7x | -6.1 pt, 1.9x | -9.9 pt, 1.4x |
| ZGQ-M16+global-entry | -19.9 pt, 10.1x | -32.8 pt, 6.6x | -42.2 pt, 4.7x | -46.4 pt, 3.1x | -50.2 pt, 2.2x |
| ZGQ-M16+random | +12.1 pt, 1.7x | -0.1 pt, 1.2x | -1.0 pt, 1.1x | +0.1 pt, 1.1x | -0.8 pt, 1.1x |
| ZGQ-M16+random+global-entry-nopq | +28.0 pt, 7.9x | +15.0 pt, 5.1x | +5.7 pt, 3.3x | +1.5 pt, 2.4x | -2.2 pt, 1.7x |
| ZGQ-M16+random+global-entry | -16.9 pt, 11.8x | -29.9 pt, 7.6x | -39.2 pt, 5.0x | -43.4 pt, 3.6x | -47.3 pt, 2.6x |

HNSW-M16 reference: ef=16 recall 63.8% 0.034 ms, ef=32 recall 76.7% 0.052 ms, ef=64 recall 86.1% 0.080 ms, ef=128 recall 90.3% 0.113 ms, ef=256 recall 94.2% 0.157 ms

Equal recall: latency ms (ratio to HNSW-M16 at the same recall); max recall of each curve in the last column:

| Index | R=80% | R=90% | R=95% | Max recall (latency) |
|---|---|---|---|---|
| HNSW-M16 | 0.062 (1.0x) | 0.110 (1.0x) | not reached | 94.2% (0.157 ms) |
| ZGQ-M16 | 0.071 (1.1x) | not reached | not reached | 84.8% (0.119 ms) |
| ZGQ-M16+global-entry-nopq | 0.216 (3.5x) | not reached | not reached | 84.3% (0.219 ms) |
| ZGQ-M16+zonescan-nopq | 0.343 (5.5x) | 1.351 (12.2x) | 2.763 (HNSW n/a) | 100.0% (4.372 ms) |
| ZGQ-M16+overlap2-nopq | 0.437 (7.0x) | 2.008 (18.2x) | 3.841 (HNSW n/a) | 100.0% (6.034 ms) |
| ZGQ-M16+adaptive-nopq | 0.476 (7.7x) | 0.476 (4.3x) | 0.476 (HNSW n/a) | 100.0% (1.630 ms) |
| ZGQ-M16+global-entry | not reached | not reached | not reached | 44.0% (0.351 ms) |
| ZGQ-M16+zonescan | not reached | not reached | not reached | 32.1% (0.220 ms) |
| ZGQ-M16+overlap2 | not reached | not reached | not reached | 34.4% (0.246 ms) |
| ZGQ-M16+adaptive | not reached | not reached | not reached | 38.6% (0.378 ms) |
| ZGQ-M16+random | 0.071 (1.1x) | 0.122 (1.1x) | not reached | 93.4% (0.165 ms) |
| ZGQ-M16+random+global-entry-nopq | 0.267 (4.3x) | 0.267 (2.4x) | not reached | 92.0% (0.274 ms) |
| ZGQ-M16+random+global-entry | not reached | not reached | not reached | 46.9% (0.406 ms) |

### 1m (1,000,000 vectors)

Build time and analytic index bytes vs HNSW-M16:

| Index | Build s | Build ratio | Analytic MB | Bytes ratio |
|---|---:|---:|---:|---:|
| HNSW-M16 | 171.8 | 1.00 | 660.5 | 1.00 |
| ZGQ-M16 | 179.0 | 1.04 | 1197.2 | 1.81 |
| ZGQ-M16+global-entry-nopq | 179.4 | 1.04 | 1197.2 | 1.81 |
| ZGQ-M16+zonescan-nopq | 179.4 | 1.04 | 1197.2 | 1.81 |
| ZGQ-M16+overlap2-nopq | 180.9 | 1.05 | 1201.2 | 1.82 |
| ZGQ-M16+adaptive-nopq | 179.4 | 1.04 | 1197.2 | 1.81 |
| ZGQ-M16+global-entry | 179.4 | 1.04 | 1197.2 | 1.81 |
| ZGQ-M16+zonescan | 179.4 | 1.04 | 1197.2 | 1.81 |
| ZGQ-M16+overlap2 | 180.8 | 1.05 | 1201.2 | 1.82 |
| ZGQ-M16+adaptive | 179.4 | 1.04 | 1197.2 | 1.81 |
| ZGQ-M16+random | 245.4 | 1.43 | 1197.2 | 1.81 |
| ZGQ-M16+random+global-entry-nopq | 245.4 | 1.43 | 1197.2 | 1.81 |
| ZGQ-M16+random+global-entry | 245.4 | 1.43 | 1197.2 | 1.81 |

Equal ef (variants swept over ef), recall delta in points and latency ratio:

| Index | ef=16 | ef=32 | ef=64 | ef=128 | ef=256 |
|---|---|---|---|---|---|
| ZGQ-M16 | +4.8 pt, 1.2x | -6.8 pt, 0.8x | -9.5 pt, 0.7x | -9.7 pt, 0.8x | -10.2 pt, 0.8x |
| ZGQ-M16+global-entry-nopq | +20.5 pt, 5.5x | +7.8 pt, 3.6x | -2.1 pt, 2.4x | -7.2 pt, 1.8x | -10.8 pt, 1.3x |
| ZGQ-M16+global-entry | -23.5 pt, 7.9x | -36.2 pt, 5.2x | -46.2 pt, 3.5x | -51.2 pt, 2.6x | -55.3 pt, 1.9x |
| ZGQ-M16+random | +10.9 pt, 1.5x | -0.4 pt, 1.1x | -1.0 pt, 1.0x | -0.4 pt, 1.0x | -0.1 pt, 1.0x |
| ZGQ-M16+random+global-entry-nopq | +30.9 pt, 6.6x | +18.2 pt, 4.3x | +8.2 pt, 2.8x | +3.1 pt, 2.1x | -1.0 pt, 1.6x |
| ZGQ-M16+random+global-entry | -19.5 pt, 8.9x | -32.2 pt, 5.9x | -42.2 pt, 3.9x | -47.2 pt, 3.0x | -51.5 pt, 2.2x |

HNSW-M16 reference: ef=16 recall 54.9% 0.053 ms, ef=32 recall 67.6% 0.080 ms, ef=64 recall 77.6% 0.120 ms, ef=128 recall 82.7% 0.163 ms, ef=256 recall 87.0% 0.222 ms

Equal recall: latency ms (ratio to HNSW-M16 at the same recall); max recall of each curve in the last column:

| Index | R=80% | R=90% | R=95% | Max recall (latency) |
|---|---|---|---|---|
| HNSW-M16 | 0.140 (1.0x) | not reached | not reached | 87.0% (0.222 ms) |
| ZGQ-M16 | not reached | not reached | not reached | 76.8% (0.171 ms) |
| ZGQ-M16+global-entry-nopq | not reached | not reached | not reached | 76.2% (0.292 ms) |
| ZGQ-M16+zonescan-nopq | 11.541 (82.3x) | 29.165 (HNSW n/a) | 38.259 (HNSW n/a) | 97.7% (43.079 ms) |
| ZGQ-M16+overlap2-nopq | 23.386 (166.8x) | 67.467 (HNSW n/a) | 99.618 (HNSW n/a) | 99.0% (125.275 ms) |
| ZGQ-M16+adaptive-nopq | 5.183 (37.0x) | 10.477 (HNSW n/a) | 14.895 (HNSW n/a) | 97.3% (26.886 ms) |
| ZGQ-M16+global-entry | not reached | not reached | not reached | 31.7% (0.426 ms) |
| ZGQ-M16+zonescan | not reached | not reached | not reached | 16.7% (0.533 ms) |
| ZGQ-M16+overlap2 | not reached | not reached | not reached | 16.6% (1.557 ms) |
| ZGQ-M16+adaptive | not reached | not reached | not reached | 17.9% (6.146 ms) |
| ZGQ-M16+random | 0.151 (1.1x) | not reached | not reached | 86.9% (0.224 ms) |
| ZGQ-M16+random+global-entry-nopq | 0.341 (2.4x) | not reached | not reached | 86.1% (0.348 ms) |
| ZGQ-M16+random+global-entry | not reached | not reached | not reached | 35.5% (0.498 ms) |

