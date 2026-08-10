# Paper-to-code map

| Manuscript concept | Code |
|---|---|
| shared reference timeline, masks, offsets | `src/tidalnet/data/schema.py` |
| controlled missingness and drift | `src/tidalnet/data/corruption.py` |
| training-only MIC graph | `src/tidalnet/graph/mic.py` |
| forward/backward neural SDE | `src/tidalnet/models/binsde.py` |
| Euler–Maruyama and Monte Carlo variance | `BiNSDECompensator._propagate` |
| variance/distance fusion and residual gate | `BiNSDECompensator.forward` |
| liquid time constants and temporal indicator | `src/tidalnet/models/ligru.py` |
| dynamic local graph and feedback | `src/tidalnet/models/ligat.py` |
| closed-loop framework | `src/tidalnet/models/tidalnet.py` |
| masked reconstruction objective | `src/tidalnet/training/losses.py` |
| 30 paired runs and significance tests | `scripts/run_repeated.py`, `scripts/aggregate_runs.py` |
| single-window latency protocol | `scripts/benchmark_latency.py` |
