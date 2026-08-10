# Validation performed for this release

The repository was validated in a CPU-only environment with Python 3 and PyTorch.

Checks completed before packaging:

- Python byte-code compilation for `src/`, `scripts/`, and `tests/`;
- six deterministic unit tests covering corruption masks, MIC graph construction, model shapes and gradients, masked metrics, configuration inheritance, and training-only normalization;
- end-to-end smoke execution: demo generation → MIC graph → split-wise corruption → validation → one-epoch training → checkpoint → test evaluation;
- one complete repeated-run orchestration with a per-seed corruption file, resolved configuration, normalization statistics, checkpoint, history, and metrics.

These checks establish software executability. They do not claim numerical reproduction of unpublished tables until the author-required values in `AUTHOR_CHECKLIST.md` are resolved.
