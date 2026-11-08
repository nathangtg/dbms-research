# Research log

An undated account of the ZGQ study, written as a falsification study: state what was claimed, test it
under matched conditions, record what failed. Phases follow the order of the git history. Every entry links
to a commit or a file in this repository. Commit hashes can be inspected with `git show <hash>`.
Per-release detail is in [../CHANGELOG.md](../CHANGELOG.md) and the status of each number is in
[CLAIMS.md](CLAIMS.md).

## Phase 1: Initial hypotheses

ZGQ (Zone-Guided Quantization) combines K-Means zones, per-zone or zone-aware HNSW graphs and product
quantization. The working hypothesis was that zoning would let a graph index reach HNSW recall with less
search work and less memory.

- First implementation and benchmark framework (V5): `b837268`.
- V6 modules and documentation (zonal partitioning `fe01324`, per-zone HNSW `86a73cd`, PQ `dc232d5`,
  main index `9cda852`, architecture `7307a43`): see the Phase 1 list in the changelog.
- Comparison against HNSW, IVF and IVF+PQ, and a first scaling analysis: `8c37f9a`, `563a1e2`, `ce86880`.
  The scaling analysis already recorded trouble at larger sizes (`ce86880`, `89a88b0`).
- Optimization plans for V7 (`64f947b`, `0354f0c`, `6d1190e`) and mathematical arguments for efficiency
  (`e9b8775`, `59a65c0`). The theory assumed model parameters that were never measured (see Phase 4).

Status: these documents state hypotheses and plans. Their numbers are not carried forward unless they
appear in [CLAIMS.md](CLAIMS.md).

## Phase 2: The v8 rewrite and the v1 preprint claims

v8 restructured the code as a package: adaptive hierarchical zones `87df652`, zone-guided navigation `d9fc363`,
residual PQ `5be6dfd`, `ZGQIndex` `c5e1d66`, zone-ordered insertion `d513d37`, benchmark suite `6d91a31`.

The TechRxiv v1 preprint, archived verbatim in [preprint/v1.md](preprint/v1.md) (`fdf2a72`, see
[preprint/README.md](preprint/README.md)), claimed that ZGQ improved recall, throughput and latency over
HNSW at matched settings, that it was much faster than IVF-PQ at high recall, and that it reduced memory footprint and index construction time. The numbers behind those claims came from runs that are not in the
repository.

## Phase 3: Matched-M re-evaluation

The canonical run compares ZGQ-M16 against hnswlib HNSW-M16 (same M, same `ef_construction`) and FAISS
baselines on one synthetic dataset family, with a committed harness, schema check and tests:
`3882848`, results in [`results/2026-11-08/`](../results/2026-11-08/README.md), written up in
[BENCHMARKS.md](BENCHMARKS.md). Numbers are traced in [CLAIMS.md](CLAIMS.md).

Outcome: ZGQ ties HNSW at 10K, trails it by roughly 7 to 10 recall points at 100K and 1M at equal `ef`, builds
1.2x to 2.7x slower, and uses more memory.

### v1 claim versus measured outcome

| v1 claim (paraphrased) | Measured outcome | Evidence |
|---|---|---|
| ZGQ has higher recall than HNSW | Recall ties at 10K (92.3% vs 92.4% at `ef=64`). ZGQ is lower at 100K (78.9% vs 86.1%) and 1M (68.1% vs 77.6%) at `ef=64`. | [CLAIMS.md](CLAIMS.md), [`results/2026-11-08/summary.json`](../results/2026-11-08/summary.json) |
| ZGQ has lower latency and higher throughput at equal recall | ZGQ is faster per query at equal `ef`, but at equal recall it needs a larger `ef`. At 100K, ZGQ-M32 at `ef=256` (86.2%) takes 0.128 ms against 0.084 ms for HNSW-M16 at `ef=64` (86.1%). | [CLAIMS.md](CLAIMS.md) iso-recall row, [`results/2026-11-08/summary.json`](../results/2026-11-08/summary.json) |
| ZGQ is many times faster than IVF-PQ at high recall | IVF-PQ never reaches 90% recall (it saturates at 64.4%, 61.5% and 56.2%), so the ratio cannot be computed. | [CLAIMS.md](CLAIMS.md) |
| ZGQ reduces memory footprint | Analytic index bytes are 1.81x to 1.84x HNSW-M16. RSS deltas on macOS give the same ordering. | [`results/2026-11-08-improvements/final/README.md`](../results/2026-11-08-improvements/final/README.md) |
| ZGQ reduces index construction time | Build is 1.2x to 2.7x slower than HNSW-M16. | [CLAIMS.md](CLAIMS.md) |
| Results validate the theoretical predictions of the model | Recall at `ef=64` falls faster for ZGQ than for HNSW as size grows, and the model's conclusion is not confirmed at 100K and 1M. | [BENCHMARKS.md](BENCHMARKS.md) |

The retired numbers, with locations and replacements, are listed in [CLAIMS.md](CLAIMS.md) and are blocked by
`make claims-check` (`scripts/check_claims.py`). The preprint and `latex/paper.pdf` were not edited.

## Phase 4: Consolidation and removal of unsupported content

Work that made the repository agree with its measurements:

- v1 to v7 removed from the working tree, kept at tag `pre-v8-cleanup`: `8444b40`.
- Figures and tables generated only from `summary.json`; claims ledger created: `8fe518d`.
- Draft documents with untraceable tables removed; model parameters such as `alpha = 0.65` relabelled as
  assumed: `1a48ae1`, `03b8609`, `bbdb449`.
- A process abort in the numba distance kernel under threaded `batch_search` was fixed and tested: `46bc14b`.

The theory in [THEORY.md](THEORY.md) is unvalidated against the data: at 10K the measurements sit at the
boundary of the regime the theorem needs, and inside the regime (100K, 1M) they do not confirm it
([BENCHMARKS.md](BENCHMARKS.md)).

## Phase 5: Why the gap, and four attempts to close it

Diagnostics (`2120e38`, `205b297`, [`diagnostics/README.md`](../results/2026-11-08-improvements/diagnostics/README.md)):

- The benchmarked fast path never calls zone selection, so zones play no role at search time on it.
- Zone-ordered insertion is the main cause of the gap. At 100K and `ef=256`: 84.77% with zone order, 93.41% with
  random order, 94.19% for plain hnswlib.

Four improvement ideas, each implemented as an opt-in option (`ff80de7`, `ba6c3a5`, `24fec5a`) and run against
matched HNSW-M16 at 10K, 100K and 1M (`36364e4`,
[`final/README.md`](../results/2026-11-08-improvements/final/README.md),
[`final/comparison.md`](../results/2026-11-08-improvements/final/comparison.md)):

| Idea | Outcome | Evidence |
|---|---|---|
| Overlapping zones (`zone_replicas=2`) | Negative. At 1M, 80% recall costs 166.8x the HNSW-M16 latency. | [`final/comparison.md`](../results/2026-11-08-improvements/final/comparison.md) |
| Adaptive probing | Negative. 37.0x HNSW-M16 latency at 80% recall (1M). | [`final/comparison.md`](../results/2026-11-08-improvements/final/comparison.md) |
| Global graph with zone-aware entry points | Negative. Gains recall at low `ef`, loses 9.9 to 10.8 points at `ef=256` and costs 1.3x to 6.3x HNSW-M16 latency. | [`final/comparison.md`](../results/2026-11-08-improvements/final/comparison.md) |
| Compressed search at equal memory | Negative. Loses to IVF-PQ(R) and HNSW+SQ8 in every memory band. | [`compression/README.md`](../results/2026-11-08-improvements/compression/README.md), `bee9583`, `81b6065` |

The one change that closes the recall gap is random insertion order, which removes the zone ordering: recall
86.9% against 87.0% for HNSW-M16 at 1M and `ef=256`, with 1.43x build time and 1.81x analytic bytes. That is
HNSW with extra memory, so it is not an improvement over HNSW. The write-up is `7a68875`.

## Phase 6: Abandoned build-order study

A study of insertion order on real datasets (SIFT1M, GloVe-100) was started to ask whether the recall loss
from zone ordering is specific to the synthetic clustered data:

- Order strategies (`042b443`), loaders (`743d41f`), graph parser and statistics (`22811d8`), runner
  (`fbcf497`), report generator (`b090313`), with tests (`0fe6315`, `c2a4c41`, `e066d0d`, `d116e6c`).
- The run was interrupted and cancelled. The partial output was committed as is (`8da48cf`, `e6ae65a`) and is
  described in [`results/2026-11-09-novelty/README.md`](../results/2026-11-09-novelty/README.md): SIFT and
  GloVe at 100K only, 3 seeds, no `summary.json`, not aggregated, not verified.

Unverified partial indication: in the partial runs the recall loss from zone ordering appeared on the synthetic
data and not on the real datasets. Nothing in the docs or paper relies on this. It needs a complete,
aggregated run before it can be stated as a finding.

## What remains untested

- Full-scale real-data runs (SIFT1M, GloVe at full size). Only the partial 100K runs above exist.
- Disk-resident settings. Everything here is in memory.
- Memory measured on Linux. Memory figures on macOS are RSS deltas and are used only for ordering; the
  analytic model in `zgq.memory_model` was checked against serialized sizes, not against peak process memory.
- ZGQ-M24 and ZGQ-M32 in the improvement sweep (skipped).
- Multi-thread and incremental-insertion behaviour. Raw partial files exist under `results/2026-11-09-novelty/`
  (`incremental/`, `parallel/`) and are unanalyzed.

## Repository layout

The `v8/` directory was hoisted to the repository root, and dependency management moved to uv. Paths in
older entries of this log and of the changelog refer to the layout of their time.

## Limitations

- Synthetic data. The canonical and improvement runs use clustered Gaussian vectors of dimension 128. Rankings
  could change on real embeddings.
- One machine (Apple M4, 16 GB, macOS), one seed (42), single-thread timing.
- Implementation mismatch. HNSW is hnswlib (C++); ZGQ is Python with numba. Latency ratios mix algorithm and
  implementation effects.
- Low-`ef` ZGQ recall is inflated by a `fast_mode` rerank, so conclusions use `ef` of 64 and above.
- The theory is analytic and has not been validated by the measurements.
