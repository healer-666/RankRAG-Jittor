#!/usr/bin/env bash
set -euo pipefail
cd /root/autodl-tmp/RankRAG-Jittor-fd98e37

export QWEN_LORA_MODEL_PATH=/root/autodl-tmp/hf_cache/hub/models--Qwen--Qwen2.5-1.5B-Instruct/snapshots/989aa7980e4cf806f80c7fef2b1adb7bc71aa306
export QWEN_GENERATOR_MODEL_PATH="$QWEN_LORA_MODEL_PATH"
export RANKRAG_GIT_COMMIT=fd98e372e54e5b7308a4117c4eafca217ac8a63c
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

mkdir -p outputs logs/e1_autodl_4090d logs/e2_autodl_4090d docs
STATUS=logs/stage_e_status.tsv
printf 'timestamp\tstage\tstatus\tdetail\n' > "$STATUS"

record() {
  printf '%s\t%s\t%s\t%s\n' "$(date -Is)" "$1" "$2" "$3" | tee -a "$STATUS"
}

start_monitor() {
  local out="$1"
  printf 'timestamp,memory.used [MiB],utilization.gpu [%%],temperature.gpu,power.draw [W],clocks.sm [MHz]\n' > "$out"
  while true; do
    nvidia-smi --query-gpu=timestamp,memory.used,utilization.gpu,temperature.gpu,power.draw,clocks.sm --format=csv,noheader,nounits >> "$out" || true
    sleep 5
  done
}

run_with_monitor() {
  local stage="$1"; shift
  local log="$1"; shift
  local gpu_csv="$1"; shift
  record "$stage" start "$*"
  start_monitor "$gpu_csv" &
  local mon_pid=$!
  set +e
  "$@" > "$log" 2>&1
  local code=$?
  set -e
  kill "$mon_pid" 2>/dev/null || true
  wait "$mon_pid" 2>/dev/null || true
  printf '%s\n' "$code" > "${log}.exitcode"
  if [[ "$code" -ne 0 ]]; then
    record "$stage" failed "exit=$code log=$log"
    tail -n 120 "$log" || true
    exit "$code"
  fi
  record "$stage" finished "exit=0 log=$log"
}

validate_train() {
  local name="$1" outdir="$2" adapter="$3" pairs="$4"
  python - "$name" "$outdir" "$adapter" "$pairs" <<'PY'
import json, math, sys
from pathlib import Path
name, outdir, adapter, pairs = sys.argv[1], Path(sys.argv[2]), Path(sys.argv[3]), int(sys.argv[4])
summary_path = outdir / 'train_summary.json'
log_path = outdir / 'train_log.jsonl'
assert summary_path.exists(), f'{name}: missing {summary_path}'
assert log_path.exists(), f'{name}: missing {log_path}'
summary = json.loads(summary_path.read_text(encoding='utf-8'))
last = None
for line in log_path.read_text(encoding='utf-8').splitlines():
    if line.strip():
        last = json.loads(line)
assert summary.get('steps') == 800, f'{name}: summary steps={summary.get("steps")}'
assert last and last.get('step') == 800, f'{name}: last log step={None if last is None else last.get("step")}'
assert summary.get('train_pairs') == pairs, f'{name}: train_pairs={summary.get("train_pairs")}'
assert summary.get('device') == 'cuda', f'{name}: device={summary.get("device")}'
assert summary.get('device_name') == 'NVIDIA GeForce RTX 4090 D', f'{name}: device_name={summary.get("device_name")}'
assert summary.get('git_commit') == 'fd98e372e54e5b7308a4117c4eafca217ac8a63c', f'{name}: git_commit={summary.get("git_commit")}'
assert adapter.joinpath('adapter_config.json').exists(), f'{name}: missing adapter_config.json'
assert adapter.joinpath('adapter_model.safetensors').exists(), f'{name}: missing adapter_model.safetensors'
loss = summary.get('loss_end')
assert isinstance(loss, (int, float)) and math.isfinite(float(loss)), f'{name}: bad loss_end={loss}'
print(f'{name}: train validation passed')
PY
  record "$name" verified "train_summary/adapters/step800 OK"
}

validate_eval() {
  local name="$1" metrics_path="$2" rankings_path="$3" method="$4"
  python - "$name" "$metrics_path" "$rankings_path" "$method" <<'PY'
import json, math, sys
from pathlib import Path
name, metrics_path, rankings_path, method = sys.argv[1], Path(sys.argv[2]), Path(sys.argv[3]), sys.argv[4]
assert metrics_path.exists(), f'{name}: missing {metrics_path}'
assert rankings_path.exists(), f'{name}: missing {rankings_path}'
metrics = json.loads(metrics_path.read_text(encoding='utf-8'))
rankings = json.loads(rankings_path.read_text(encoding='utf-8'))
assert metrics.get('num_queries') == 500, f'{name}: num_queries={metrics.get("num_queries")}'
assert len(rankings) == 500, f'{name}: rankings={len(rankings)}'
if method != 'any':
    assert metrics.get('scoring_method') == method, f'{name}: scoring_method={metrics.get("scoring_method")}'
assert metrics.get('device_name') == 'NVIDIA GeForce RTX 4090 D', f'{name}: device_name={metrics.get("device_name")}'
assert metrics.get('git_commit') == 'fd98e372e54e5b7308a4117c4eafca217ac8a63c', f'{name}: git_commit={metrics.get("git_commit")}'
for key in ['recall@1','recall@3','recall@5','recall@10','ndcg@5','ndcg@10','mrr','pairwise_accuracy']:
    value = metrics.get(key)
    assert isinstance(value, (int, float)) and math.isfinite(float(value)) and 0 <= float(value) <= 1, f'{name}: invalid {key}={value}'
print(f'{name}: eval validation passed')
PY
  record "$name" verified "metrics/rankings OK"
}

# E1: data-size ablation
run_with_monitor e1_1k_train logs/e1_autodl_4090d/1k_train.log logs/e1_autodl_4090d/1k_gpu.csv python src/lora_reranker/train_lora_reranker.py --config configs/lora_qwen_1_5b_1k_lr1e4_s800.yaml
validate_train e1_1k outputs/lora_ablation_data_1k outputs/lora_adapters/qwen_1_5b_1k_lr1e4_s800 1000
run_with_monitor e1_1k_eval logs/e1_autodl_4090d/1k_eval.log logs/e1_autodl_4090d/1k_eval_gpu.csv python src/lora_reranker/evaluate_lora_reranker.py --model_name Qwen/Qwen2.5-1.5B-Instruct --adapter_dir outputs/lora_adapters/qwen_1_5b_1k_lr1e4_s800 --test_path data/processed/lora_qwen_1_5b_10k/test_queries.jsonl --output_metrics outputs/lora_ablation_data_1k/qwen_1_5b_lora_metrics.json --output_rankings outputs/lora_ablation_data_1k/qwen_1_5b_lora_rankings.json --max_queries 500 --max_length 256 --scoring_method logprob_margin
validate_eval e1_1k outputs/lora_ablation_data_1k/qwen_1_5b_lora_metrics.json outputs/lora_ablation_data_1k/qwen_1_5b_lora_rankings.json logprob_margin

run_with_monitor e1_3k_train logs/e1_autodl_4090d/3k_train.log logs/e1_autodl_4090d/3k_gpu.csv python src/lora_reranker/train_lora_reranker.py --config configs/lora_qwen_1_5b_3k_lr1e4_s800.yaml
validate_train e1_3k outputs/lora_ablation_data_3k outputs/lora_adapters/qwen_1_5b_3k_lr1e4_s800 3000
run_with_monitor e1_3k_eval logs/e1_autodl_4090d/3k_eval.log logs/e1_autodl_4090d/3k_eval_gpu.csv python src/lora_reranker/evaluate_lora_reranker.py --model_name Qwen/Qwen2.5-1.5B-Instruct --adapter_dir outputs/lora_adapters/qwen_1_5b_3k_lr1e4_s800 --test_path data/processed/lora_qwen_1_5b_10k/test_queries.jsonl --output_metrics outputs/lora_ablation_data_3k/qwen_1_5b_lora_metrics.json --output_rankings outputs/lora_ablation_data_3k/qwen_1_5b_lora_rankings.json --max_queries 500 --max_length 256 --scoring_method logprob_margin
validate_eval e1_3k outputs/lora_ablation_data_3k/qwen_1_5b_lora_metrics.json outputs/lora_ablation_data_3k/qwen_1_5b_lora_rankings.json logprob_margin

run_with_monitor e1_10k_rerun_train logs/e1_autodl_4090d/10k_rerun_train.log logs/e1_autodl_4090d/10k_rerun_gpu.csv python src/lora_reranker/train_lora_reranker.py --config configs/lora_qwen_1_5b_10k_lr1e4_s800_rerun.yaml
validate_train e1_10k_rerun outputs/lora_ablation_data_10k_rerun outputs/lora_adapters/qwen_1_5b_10k_lr1e4_s800_rerun 10000
run_with_monitor e1_10k_rerun_eval logs/e1_autodl_4090d/10k_rerun_eval.log logs/e1_autodl_4090d/10k_rerun_eval_gpu.csv python src/lora_reranker/evaluate_lora_reranker.py --model_name Qwen/Qwen2.5-1.5B-Instruct --adapter_dir outputs/lora_adapters/qwen_1_5b_10k_lr1e4_s800_rerun --test_path data/processed/lora_qwen_1_5b_10k/test_queries.jsonl --output_metrics outputs/lora_ablation_data_10k_rerun/qwen_1_5b_lora_metrics.json --output_rankings outputs/lora_ablation_data_10k_rerun/qwen_1_5b_lora_rankings.json --max_queries 500 --max_length 256 --scoring_method logprob_margin
validate_eval e1_10k_rerun outputs/lora_ablation_data_10k_rerun/qwen_1_5b_lora_metrics.json outputs/lora_ablation_data_10k_rerun/qwen_1_5b_lora_rankings.json logprob_margin

python scripts/aggregate_lora_data_size_ablation.py > logs/e1_autodl_4090d/aggregate.log 2>&1
record e1_aggregate finished outputs/lora_ablation_results.json

# E2: scoring-method ablation on the 10k-rerun adapter
for method in generate_parse relevant_logprob logprob_margin; do
  mkdir -p "outputs/lora_scoring_ablation/$method"
  extra=()
  if [[ "$method" == "generate_parse" ]]; then
    extra=(--output_tokenization outputs/lora_scoring_label_tokenization.json)
  fi
  run_with_monitor "e2_${method}_eval" "logs/e2_autodl_4090d/${method}_eval.log" "logs/e2_autodl_4090d/${method}_gpu.csv" python src/lora_reranker/evaluate_lora_reranker.py --model_name Qwen/Qwen2.5-1.5B-Instruct --adapter_dir outputs/lora_adapters/qwen_1_5b_10k_lr1e4_s800_rerun --test_path data/processed/lora_qwen_1_5b_10k/test_queries.jsonl --output_metrics "outputs/lora_scoring_ablation/$method/qwen_1_5b_lora_metrics.json" --output_rankings "outputs/lora_scoring_ablation/$method/qwen_1_5b_lora_rankings.json" --max_queries 500 --max_length 256 --scoring_method "$method" "${extra[@]}"
  validate_eval "e2_$method" "outputs/lora_scoring_ablation/$method/qwen_1_5b_lora_metrics.json" "outputs/lora_scoring_ablation/$method/qwen_1_5b_lora_rankings.json" "$method"
done

python scripts/aggregate_lora_scoring_ablation.py > logs/e2_autodl_4090d/aggregate.log 2>&1
record e2_aggregate finished outputs/lora_scoring_ablation_results.json
record stage_e complete fd98e372e54e5b7308a4117c4eafca217ac8a63c
