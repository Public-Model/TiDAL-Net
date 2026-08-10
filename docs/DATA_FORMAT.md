# Canonical NPZ schema

Required arrays:

| key | shape | meaning |
|---|---:|---|
| `values` | `[T, M]` | complete reference telemetry where available |
| `reference_times` | `[T]` | shared reconstruction grid in elapsed seconds |
| `natural_mask` | `[T, M]` | 1 where original reference is valid |

A corrupted file additionally contains:

| key | shape | meaning |
|---|---:|---|
| `received_values` | `[T, M]` | model input after artificial missingness |
| `observation_mask` | `[T, M]` | received-value availability |
| `timestamp_offsets` | `[T, M]` | advance/delay relative to reference grid |
| `artificial_missing_mask` | `[T, M]` | controlled missing positions |
| `artificial_drift_mask` | `[T, M]` | controlled timestamp-displacement positions |
| `evaluation_mask` | `[T, M]` | positions with exact held-out reference labels |
| `metadata_json` | scalar string | seed, realized ratios, and configuration |

Optional arrays include channel names, satellite IDs, subsystem IDs, link reachability, and communication delay. Protocol fields and categorical status variables should not be mixed with continuous reconstruction targets.
