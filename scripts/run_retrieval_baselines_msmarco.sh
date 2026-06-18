#!/usr/bin/env bash
set -euo pipefail

python src/baseline_retrieval.py \
  --data_path data/processed/msmarco/test.jsonl \
  --output_path outputs/msmarco_retrieval_baseline_metrics.json \
  --rankings_path outputs/msmarco_retrieval_baseline_rankings.json \
  --run_name msmarco
