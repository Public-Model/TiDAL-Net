# Reality-informed degradation protocol

This document specifies the executable degradation process used by
`src/tidalnet/data/corruption.py` and `scripts/generate_corruptions.py`.
The goal is to make the distinction between **missing observations** and
**timestamp displacement** auditable in code rather than leaving it as a prose-only claim.

## 1. What the affected-data ratio means

Let `natural_mask[t, i] = 1` denote an originally valid telemetry value. For a requested
affected-data ratio `rho`, the generator first computes

```text
valid_count     = sum(natural_mask)
affected_count  = round(rho * valid_count)
drift_count     = round(drift_fraction * affected_count)
missing_count   = affected_count - drift_count
```

Only originally valid positions are eligible. Missing and timestamp-displaced positions are
sampled as **disjoint sets**. In the public revision profile, `drift_fraction = 0.40`, so the
requested affected mass is split into 60% missing observations and 40% timestamp displacement
before collision handling.

The experimental ratios are `0.03`, `0.05`, `0.08`, and `0.10`. Corruptions are generated
independently inside the chronological train/validation/test splits by default. The same saved
corruption file is then reused by every method in a paired comparison.

## 2. How missing observations are created

A missing observation is a **concealed received value**, not deletion of the reference target.
For an affected position `(t, i)`:

```text
values[t, i]                    = original reference value  # retained for evaluation only
received_values[t, i]           = 0 after normalization
observation_mask[t, i]          = 0
artificial_missing_mask[t, i]   = 1
evaluation_mask[t, i]           = 1
```

The generator uses four operators.

### 2.1 Isolated loss

One valid channel-time position is selected and concealed.

### 2.2 Short-burst loss

One channel is selected and a contiguous interval is concealed. The public configuration uses
`2--8` samples.

### 2.3 Long contiguous gap

One channel is selected and a longer interval is concealed. Gap length is sampled from a
log-normal distribution. The public calibration uses a median of `15` samples and a 95th
percentile of `90` samples, clipped to the configured finite bounds.

### 2.4 Correlated-channel loss

An anchor channel is selected. If a candidate graph is supplied, the most strongly related
channels in the anchor row are selected; otherwise channels are sampled uniformly. The same
short interval is then concealed across the selected channel group. This represents packet,
subsystem, or transmission-unit failures that affect physically related channels together.

### 2.5 Ratio-dependent mixture

`configs/corruption/reality_informed.yaml` contains an explicit mixture for each evaluated
affected-data ratio. The public profile preserves the manuscript rule that isolated loss becomes
less dominant while long contiguous gaps become more prominent as the affected-data ratio
increases. The resolved mixture is saved in `metadata_json` with every generated NPZ.

## 3. How timestamp displacement is created

Timestamp displacement **does not delete the telemetry value**. The value remains available, but
its physical observation time is moved away from the reference grid time:

```text
received_values[t, i]          = original received value
observation_mask[t, i]         = 1
timestamp_offsets[t, i]        = non-zero physical-time offset
observation_times[t, i]        = reference_times[t] + timestamp_offsets[t, i]
artificial_drift_mask[t, i]    = 1
evaluation_mask[t, i]          = 1
```

`timestamp_offsets` is stored in the **same physical-time unit as `reference_times`**. Offset
magnitudes are generated in nominal sampling intervals and then multiplied by the median positive
interval of `reference_times`. Therefore, for a 60-s nominal grid, an offset of `+0.8` intervals is
stored as `+48 s`.

The public profile contains the four processes described in the manuscript.

### 3.1 Local jitter

An isolated received sample receives a zero-mean Gaussian time offset. The public profile uses a
standard deviation of `2` nominal intervals and clips the magnitude at `5` intervals.

### 3.2 Heavy-tailed transient offset

An isolated received sample receives a Student-t offset with `df = 3`, scaled and clipped by the
same maximum magnitude. This creates infrequent but substantially larger timing errors than the
Gaussian process.

### 3.3 Accumulated drift

A contiguous channel interval is selected. Small zero-mean increments are cumulatively summed
through the interval, producing a gradually varying clock drift trajectory. The cumulative offset is
clipped to the configured maximum magnitude.

### 3.4 Reset drift

A contact-like interval is selected with a configurable mean reset period. The offset ramps from a
small value toward a sampled terminal offset and returns to zero after the interval. This mimics
clock error that accumulates within a contact period and is corrected or reset at the next timing
reference.

The default drift-type weights are equal. Exact operator counts realized in a generated file are
stored in `metadata_json` and can also be recovered from `drift_type_id`.

## 4. Collision handling

A displaced event may move outside the current split or map to a nominal reception bin already
occupied by another received event. If `collision_to_missing: true` is enabled, such a displaced record is
converted to a missing observation. The public default keeps this option disabled because Bi-NSDE consumes continuous event times directly; the option is provided for grid-remapping experiments:

```text
artificial_drift_mask -> 0
timestamp_offset       -> 0
artificial_missing_mask -> 1
observation_mask        -> 0
```

These positions are recorded with `missing_pattern_id = 5` (`drift_collision`). Collision conversion
does not create a new evaluation target; the same originally valid reference value remains the target.

## 5. Audit arrays written to each corrupted NPZ

In addition to the canonical arrays, the generator writes:

| Array | Meaning |
|---|---|
| `observation_times` | `reference_times + timestamp_offsets` for every channel |
| `missing_pattern_id` | 0 none, 1 isolated, 2 short burst, 3 long gap, 4 correlated-channel, 5 drift collision |
| `drift_type_id` | 0 none, 1 jitter, 2 heavy-tailed, 3 accumulated, 4 reset |
| `metadata_json` | seed, resolved configuration, requested/realized counts, operator counts, nominal interval |

This makes a generated corruption file self-describing and allows the exact degradation realization
used by a run to be inspected without inferring it from the reconstructed values.

## 6. Reproduce one corruption realization

```bash
python scripts/generate_corruptions.py \
  --input data/processed/demo.npz \
  --graph data/graphs/demo_mic.npy \
  --config configs/corruption/reality_informed.yaml \
  --dataset esa_adb \
  --ratio 0.05 \
  --seed 2026 \
  --output data/processed/demo_corrupted.npz
```

The command prints the saved `metadata_json`. Re-running the same command with the same input,
configuration, graph, ratio, dataset profile, and seed reproduces the same masks and offsets.

## 7. Paper-exact dataset profiles

The implementation supports `dataset_profiles` in the YAML file. A profile can override the missing
mixture, gap calibration, drift fraction, or drift-process parameters without changing the code. The
public review-stage file currently exposes the complete operator mechanics and the calibration values
stated in the revision material. Final frozen dataset-specific empirical mixtures/manifests can be
inserted under `dataset_profiles` and archived with the accepted-paper release.
