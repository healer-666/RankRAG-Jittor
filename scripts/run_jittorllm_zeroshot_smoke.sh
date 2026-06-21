#!/usr/bin/env bash
set -euo pipefail

export use_cuda=0
export nvcc_path=""

python src/jittorllm_reranker/evaluate_jittorllm_zeroshot.py \
  --config configs/jittorllm_zeroshot_medium.yaml \
  --max_queries 20 \
  --output_metrics outputs/jittorllm_zeroshot_medium/smoke_metrics.json \
  --output_rankings outputs/jittorllm_zeroshot_medium/smoke_rankings.json \
  --cache_path outputs/jittorllm_zeroshot_medium/smoke_score_cache.jsonl \
  --run_summary outputs/jittorllm_zeroshot_medium/smoke_run_summary.md
