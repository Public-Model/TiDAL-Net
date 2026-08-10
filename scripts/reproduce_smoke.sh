#!/usr/bin/env bash
set -euo pipefail
python scripts/make_demo_data.py --output data/processed/demo.npz --length 96 --channels 4 --seed 7
python scripts/build_mic_graph.py --input data/processed/demo.npz --output data/graphs/demo_mic.npy --top-k 3 --threshold 0.0
python scripts/generate_corruptions.py --input data/processed/demo.npz --graph data/graphs/demo_mic.npy --output data/processed/demo_corrupted.npz --ratio 0.05 --seed 7
python scripts/validate_data.py data/processed/demo_corrupted.npz
python scripts/train.py --config configs/smoke.yaml --device cpu
python scripts/evaluate.py --config configs/smoke.yaml --checkpoint outputs/smoke/best.pt --device cpu
