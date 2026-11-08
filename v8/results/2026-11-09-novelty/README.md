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
| GloVe (glove-100-angular) | 100,000 | zone, random, lid_asc, lid_desc, skeleton-0.01, skeleton-0.05, skeleton-0.1 (no roundrobin) |

Not present: SIFT1M or GloVe at full size, and any aggregated output.
`make benchmark-check` only visits `results/*/summary.json`, so this folder is
not touched by the checks.
