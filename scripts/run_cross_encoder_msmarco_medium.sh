#!/usr/bin/env bash
set -euo pipefail

python src/pretrained_cross_encoder_reference.py \
  --data_path data/processed/msmarco_medium/test.jsonl \
  --model_name cross-encoder/ms-marco-MiniLM-L6-v2 \
  --output_metrics outputs/msmarco_medium_cross_encoder_metrics.json \
  --output_rankings outputs/msmarco_medium_cross_encoder_rankings.json \
  --batch_size 32 \
  --max_queries 500
