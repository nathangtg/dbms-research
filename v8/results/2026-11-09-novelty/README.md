# Partial build-order novelty runs (interrupted)

Work-in-progress raw output of the build-order novelty study. The run was
interrupted and the study was cancelled, so this folder is incomplete. The
numbers have not been verified or aggregated by the study's aggregator, there
is no `summary.json`, and nothing here is quoted in the docs or paper.

Raw per-run files are in `<dataset>/raw/e1/`, one JSON per
(dataset, n, strategy, seed, trial). Seeds are 42, 43 and 44 with one trial
each. Logs are in `logs/`.

| Dataset | n | Strategies present |
|---|---|---|
| synthetic | 10,000 and 100,000 | zone, random, lid_asc, lid_desc, roundrobin, skeleton-0.01, skeleton-0.05, skeleton-0.1 |
| SIFT (sift-128-euclidean) | 100,000 | same eight strategies |
| GloVe (glove-100-angular) | 100,000 | zone, random, lid_asc, lid_desc, skeleton-0.01, skeleton-0.05, skeleton-0.1, and roundrobin (added later, see below) |

Not present: SIFT1M or GloVe at full size, and any aggregated output.
`make benchmark-check` only visits `results/*/summary.json`, so this folder is
not touched by the checks.

## Late additions (also partial and unverified)

Written by the study just before it stopped. All files parse as valid JSON, so
none is known to be half-written, but they were not checked further.

- `glove/raw/e1/glove-100-angular_n100000_roundrobin_s{42,43,44}_t1.json`:
  GloVe roundrobin runs for seeds 42, 43 and 44 (one trial each).
- `incremental/raw/e3/`: 12 files, apparently the e3 (incremental insertion)
  experiment. Synthetic and SIFT at n=100,000, strategies zone and random,
  seeds 42 to 44.
- `parallel/raw/e2/`: 18 files, apparently the e2 (parallel build, 8 threads,
  `_t8`) experiment. Synthetic and SIFT at n=100,000, strategies zone, random
  and skeleton-0.05, seeds 42 to 44.
