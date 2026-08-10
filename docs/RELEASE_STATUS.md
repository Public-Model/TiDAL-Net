# Release status

## Current tag target: `v0.1.0-review`

This package is a runnable but restricted review-stage repository, not yet a paper-exact archival release. The associated manuscript is currently under revision. The repository exposes the main architecture and workflow, while selected key parameters, exact manifests, final checkpoints, processed data assets, and the complete experimental datasets are temporarily withheld. All substitutions are visible rather than silently presented as paper-exact values.

After manuscript acceptance, the target `v1.0.0-paper` release will add the complete accepted-version source code, final configurations, exact data and split manifests, corruption masks and seeds, normalization statistics, checkpoints, logs, table and figure reproduction scripts, and all datasets that may be redistributed under the applicable licenses and data-owner agreements.

Run:

```bash
python scripts/check_release.py --review-stage
```

for a review-stage audit. Before making a final archival tag, run without `--allow-review-defaults`; the command must pass, all tests must pass, and the revised manuscript must cite the immutable release tag or DOI rather than the moving default branch.
