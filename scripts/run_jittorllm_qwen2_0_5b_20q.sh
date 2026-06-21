#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
if [[ -f "$ROOT/.venv-jittor/bin/activate" ]]; then
  source "$ROOT/.venv-jittor/bin/activate"
else
  CONDA_BIN="${CONDA_EXE:-$(command -v conda || true)}"
  [[ -n "$CONDA_BIN" ]] || CONDA_BIN="$HOME/miniconda3/bin/conda"
  eval "$("$CONDA_BIN" shell.bash hook)"
  conda activate "$ROOT/.venv-jittor"
fi
/sbin/ub-device-create 2>/dev/null || true
export HF_HOME="$ROOT/external/hf_cache"

if [[ ! -f external/hf_models/Qwen2.5-0.5B-Instruct/model.fp16.pth ]]; then
  python scripts/convert_qwen2_safetensors_to_pth.py
fi
python src/jittorllm_reranker/evaluate_qwen2_jittor.py \
  --config configs/jittorllm_qwen2_0_5b_20q.yaml
