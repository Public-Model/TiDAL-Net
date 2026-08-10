# Dataset preparation

## NASA MSL and SMAP

The downloader clones the official Telemanom release. The public release organizes channels as separate arrays and does not by itself define the exact multivariate channel stacking used by this manuscript. Therefore, `preprocess_public.py` requires an explicit ordered channel manifest and a declared alignment policy. The review defaults are only examples and must be replaced by the exact manuscript selections.

For datasets without recoverable UTC timestamps, the configured nominal one-minute grid is used and controlled offsets are added relative to this grid.

## ESA-ADB

The downloader clones the official benchmark repository and records the expected data location. Because the supplied response and current upstream release may use different channel subsets or versions, the exact raw release, channel count, channel identifiers, and checksums must be verified and pinned.

## Private datasets

Use `scripts/convert_matrix.py` to convert authorized CSV/NPY data to the canonical NPZ schema. Do not publish raw values without permission.
