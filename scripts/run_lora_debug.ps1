$ErrorActionPreference = "Stop"
$PY = if ($env:PYTHON) { $env:PYTHON } else { "python" }

& $PY src/lora_reranker/build_lora_data.py `
  --train_in data/processed/msmarco_medium/train.jsonl `
  --valid_in data/processed/msmarco_medium/valid.jsonl `
  --test_in data/processed/msmarco_medium/test.jsonl `
  --out_dir data/processed/lora_debug `
  --max_train_pairs 200 `
  --max_valid_pairs 100 `
  --max_test_queries 20 `
  --negatives_per_query 2 `
  --seed 42

& $PY src/lora_reranker/train_lora_reranker.py `
  --config configs/lora_qwen_0_5b_debug.yaml
