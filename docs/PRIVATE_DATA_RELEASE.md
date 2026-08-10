# Private-data reproducibility package

When raw telemetry cannot be released, publish the maximum non-sensitive material permitted by the data owner:

1. channel count, physical categories, units after anonymization, sampling interval, and sequence length;
2. cryptographic hashes of raw files held by the authors;
3. chronological split indices and channel ordering;
4. training-only normalization statistics or a documented reason they cannot be shared;
5. MIC candidate graph after anonymization;
6. all artificial corruption masks, timestamp offsets, seeds, and realized statistics;
7. model checkpoints and inference outputs at evaluation positions;
8. aggregate summaries for natural gaps without claiming unavailable ground truth.

A synthetic demo is useful for testing software but is not a substitute for releasing the public benchmark pipeline and exact experimental manifests.
