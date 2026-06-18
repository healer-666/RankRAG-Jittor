#!/usr/bin/env bash
set -euo pipefail

python scripts/prepare_data.py
python src/train_torch.py --config configs/default.yaml --epochs 1 --max_train_pairs 20
python src/train_jittor.py --config configs/default.yaml --epochs 1 --max_train_pairs 20
python src/eval_torch.py --config configs/default.yaml
python src/eval_jittor.py --config configs/default.yaml
