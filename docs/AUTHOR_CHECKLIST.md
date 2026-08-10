# Author verification required before public submission

The following information is not recoverable from the supplied original manuscript and response text. It must be copied from the actual revised experiment scripts/checkpoints rather than guessed.

- [ ] Exact ordered identifiers of the 8 MSL channels.
- [ ] Exact ordered identifiers of the 20 SMAP channels.
- [ ] Exact ESA-ADB release/version, channel count, ordered identifiers, and raw checksums. The response text states 327 physical channels; verify this against the exact downloaded release.
- [ ] Exact train/validation/test boundaries for every dataset, not only the 6:2:2 ratio.
- [ ] Exact input window, output horizon, stride, and handling of sequence boundaries.
- [ ] Euler–Maruyama step count used in main tables.
- [ ] Monte Carlo path counts used for training and evaluation.
- [ ] Bi-NSDE hidden dimensions, activation functions, diffusion floor, and parameter-sharing details.
- [ ] Li-GRU layer widths, minimum time constant, discretization equation, and initialization.
- [ ] MIC implementation/version, maximum grid setting, threshold, top-k, and symmetrization settings.
- [ ] Li-GAT head count, maximum radius, base temporal window, edge-weight coefficients, and graph history handling.
- [ ] Exact pattern proportions and gap-length distributions derived for each dataset.
- [ ] Exact timestamp displacement distributions and magnitudes for every table/figure.
- [ ] Loss weights, weight decay, scheduler, gradient clipping, checkpoint selection rule, and precision mode.
- [ ] Complete list of 30 seeds and saved corruption-mask checksums.
- [ ] Exact upstream repository URL and commit hash for each of the 14 baselines.
- [ ] Exact model-specific baseline hyperparameters after task adaptation.
- [ ] Final hardware, PyTorch/CUDA/cuDNN versions, deterministic settings, and latency logs.
- [ ] Permission statement or release plan for Real-A, Real-B, and Constellation-Sim.
- [ ] Replace `OWNER` in README/CITATION and add the final repository URL to the manuscript and response letter.

Do not remove this checklist merely to make the repository look complete. Resolve each item and preserve an archived release tag corresponding to the revision submission.
