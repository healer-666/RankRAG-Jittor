$ErrorActionPreference = "Stop"
$PY = "D:\anaconda\envs\pytorch\python.exe"

& $PY src/lora_reranker/evaluate_lora_reranker.py `
  --model_name Qwen/Qwen2.5-0.5B-Instruct `
  --adapter_dir outputs/lora_adapters/qwen_0_5b_debug `
  --test_path data/processed/lora_debug/test_queries.jsonl `
  --output_metrics outputs/lora_debug/qwen_0_5b_lora_metrics.json `
  --output_rankings outputs/lora_debug/qwen_0_5b_lora_rankings.json `
  --max_queries 20 `
  --max_length 192
