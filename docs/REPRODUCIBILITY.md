# Reproducibility protocol

## Immutable experimental unit

Each run is identified by `(dataset, split manifest, corruption manifest, model configuration, seed, commit hash)`. The corruption file is generated once and reused by TiDAL-Net and every baseline. This prevents different methods from receiving easier or harder affected positions.

## Ground truth and evaluation mask

Natural operational gaps do not have verifiable physical ground truth and are excluded from supervised MAE/RMSE/MAPE. Controlled self-masking is applied only to originally valid positions. The concealed values are retained as targets. A timestamp-displaced sample remains observed and receives a nonzero timestamp offset; a missing sample receives an observation mask of zero. The two sets are disjoint.

## Split and leakage prevention

1. Split chronologically into train/validation/test.
2. Fit normalization statistics on the training split only.
3. Compute MIC using valid paired observations from the training split only.
4. Freeze the candidate graph before validation/test.
5. Generate corruption masks independently within each split and save them (`generate_corruptions.py` does this by default).
6. Reuse the same masks and seeds for paired method comparisons.

## Repeated runs

The paper response specifies 30 paired runs. `scripts/run_repeated.py` writes one JSON record per run, and `scripts/aggregate_runs.py` calculates mean, sample standard deviation, confidence intervals, and two-sided paired t-tests.

## Exact reproduction checklist

A run is paper-exact only when:

- no configuration field contains `REVIEW_DEFAULT` or `AUTHOR_REQUIRED`;
- channel and split manifests match the revised experiment;
- upstream baseline commit hashes are pinned;
- public raw-file checksums are committed;
- the Git commit, package lock, GPU/CUDA details, masks, and seeds are archived.
