#!/usr/bin/env bash

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
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

if [[ ! -f "$ROOT/external/hf_models/Qwen2.5-0.5B-Instruct/model.fp16.pth" ]]; then
  python "$ROOT/scripts/convert_qwen2_safetensors_to_pth.py"
fi
