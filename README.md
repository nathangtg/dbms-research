# ZGQ v8: Zone-Guided Quantization for High-Performance Approximate Nearest Neighbor Search

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

Zone-Guided Quantization (ZGQ) is a research Approximate Nearest Neighbor Search (ANNS) algorithm that combines
hierarchical zonal partitioning with graph navigation. The algorithm is described in [docs/THEORY.md](docs/THEORY.md).

## Project status: archived

**This repository is archived and read-only.** There is no further development, and no issues or pull
requests are expected. The code, results and history are kept as a reproducible record of the study.

## Conclusion

ZGQ did not beat a matched HNSW-M16 on recall, latency or memory on the synthetic data tested. It ties
HNSW-M16 at 10K vectors and trails it at 100K and 1M.

* **Cause of the gap.** The gap comes from zone-ordered graph insertion. Random insertion closes it, at the
  cost of making ZGQ a slower, larger HNSW.
* **Negative variants.** Overlapping zones, adaptive probing, a global graph with zone-aware entry points and
  compression-first search at equal memory were all negative (opt-in, details in [docs/BENCHMARKS.md](docs/BENCHMARKS.md)).
* **The earlier preprint.** The performance, memory-reduction and IVF-PQ speedup claims of the TechRxiv
  preprint v1 ([docs/preprint/v1.md](docs/preprint/v1.md)) are not supported by the matched evaluation and are
  superseded by this repository's results. The preprint compared ZGQ at M=32 against HNSW at M=16, never
  measured memory, and compared IVF-PQ at a recall it cannot reach.
* **Real data.** The build-order experiments on SIFT and GloVe were abandoned and left unverified, so no
  real-dataset conclusion is claimed.

Canonical run `results/2026-11-08` (synthetic clustered data, dim 128, 1000 queries, single thread,
median of 5 repeats, Apple M4). Matched comparison HNSW-M16 vs ZGQ-M16, recall@10 % / ms per query:

| Scale | `ef` | HNSW-M16 | ZGQ-M16 |
|-------|------|----------|---------|
| 10K   | 256  | 98.7 / 0.099 | 98.2 / 0.101 |
| 100K  | 256  | 94.2 / 0.158 | 84.8 / 0.122 |
| 1M    | 256  | 87.0 / 0.226 | 76.8 / 0.174 |

HNSW is hnswlib (C++); ZGQ is Python with numba, so timing is not a pure algorithmic comparison.

Open questions: full-scale runs on real datasets, disk-resident or billion-scale settings, memory measurement on Linux, and shard-parallel builds and updates.

Detail: [docs/BENCHMARKS.md](docs/BENCHMARKS.md) (tables, limitations and threats to validity),
[docs/CLAIMS.md](docs/CLAIMS.md) (claim-by-claim traceability), [docs/RESEARCH_LOG.md](docs/RESEARCH_LOG.md)
(study narrative) and [CHANGELOG.md](CHANGELOG.md).

## Installation

```bash
uv sync              # dev environment (default group), creates .venv from uv.lock
uv sync --no-dev     # runtime dependencies only
uv run pytest        # run the tests
```

## Quick Start

```python
from zgq import ZGQConfig, ZGQIndex

import numpy as np

# Random data so the snippet runs as is
rng = np.random.default_rng(0)
vectors = rng.standard_normal((5000, 64)).astype(np.float32)
queries = rng.standard_normal((10, 64)).astype(np.float32)
query = queries[0]

# Create and build index
config = ZGQConfig(
    n_zones='auto',        # Automatically determine optimal zones
    use_hierarchy=True,    # Multi-level zone hierarchy
    use_simd=True,         # Enable SIMD acceleration
    use_pq=True,           # Enable (residual) product quantization
)
index = ZGQIndex(config)
index.build(vectors)

# Search
ids, distances = index.search(query, k=10)

# Batch search
ids, distances = index.batch_search(queries, k=10, n_jobs=4)
```

## Repository map

`zgq/` (package), `benchmarks/` (harness), `scripts/` (claims and link checkers), `results/` (canonical
and variant runs), `docs/` (API, theory, benchmarks, claims, research log), `latex/` (paper source),
`References/` (cited papers).

## History

Earlier iterations (v1 to v7) were removed from the working tree. They remain available in git history at the tag `pre-v8-cleanup`.

## Citation

If you use ZGQ in your research, please cite:

```bibtex
@misc{ginting_zgq_preprint_v1,
  title={Zone-Guided Quantization: High-Performance Approximate Nearest Neighbor Search},
  author={Ginting, Nathan Aldyth Prananta},
  howpublished={TechRxiv preprint v1},
  doi={10.36227/techrxiv.176704895.53855035/v1}
}
```

That preprint (archived at [docs/preprint/v1.md](docs/preprint/v1.md)) is superseded by the findings in this
repository. Its performance, memory-reduction and IVF-PQ speedup claims are not supported by the matched
evaluation described here.

## License

MIT License - see [LICENSE](LICENSE) for details.

## Farewell

Thank you to everyone who read this work or reused any of it. ZGQ did not turn out to beat HNSW, but a
negative result is still a result, and I would rather leave an honest record than a flattering one. The code
is free to reuse under the MIT licence. This repository is now closed.
