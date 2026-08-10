# Reality-informed irregularity construction

The generator treats the affected-data ratio as an occurrence statistic, not a displacement magnitude.

## Missingness operators

- **isolated:** single channel-time positions;
- **short burst:** contiguous channel-specific intervals;
- **long gap:** longer communication-like interruptions;
- **correlated-channel:** simultaneous intervals over graph-related channels.

## Timestamp drift operators

- Gaussian jitter;
- bounded uniform advance/delay;
- heavy-tailed displacement;
- accumulated random-walk drift;
- reset-type step offset;
- mixed type selected by configured weights.

Missing and drift positions are sampled as disjoint sets. Missing values are removed from the received matrix and set to zero after normalization. Drifted values remain present, while `timestamp_offsets` modifies the observation time consumed by Bi-NSDE. Measurement noise is optional and is not counted in the affected-data ratio unless explicitly configured.

Every generated NPZ stores masks, offsets, seed, realized ratios, and a JSON-serialized configuration. Commit the small corruption manifests or checksums used for paper tables.
