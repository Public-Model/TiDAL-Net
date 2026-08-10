# Baseline adaptation policy

Forecasting models retain their temporal/spatio-temporal backbones and receive an output head matching the concealed multichannel segment. Prediction/reconstruction-based anomaly detectors retain their estimation backbone but remove anomaly scoring, threshold selection, and point adjustment. Representation-only models receive a linear reconstruction head. Native imputation models retain their missing-data mechanisms.

All methods must receive the same normalized incomplete input, observation mask, timestamp information available under the method interface, split, window, concealed positions, early-stopping rule, and evaluation mask. The masked objective and metric code in this repository are shared.

Third-party implementations are not copied into this repository because their licenses and dependency stacks differ. `configs/baselines/registry.yaml` is the required provenance record. The final release must pin upstream URLs and commit hashes and provide thin adapters under `external/` or a reproducible environment per method.
