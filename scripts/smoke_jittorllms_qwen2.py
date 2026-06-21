"""Run one real Qwen2.5-0.5B generation through Jittor CUDA."""

from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from jittorllm_reranker.backend_qwen2_jittor import Qwen2JittorBackend


PROMPT = """Answer with exactly one label: Relevant or Irrelevant.

Query: what is machine learning

Passage: Machine learning is a field of artificial intelligence that studies algorithms that improve from data.

Label:"""


def gpu_memory() -> str:
    try:
        return subprocess.check_output(
            ["nvidia-smi", "--query-gpu=memory.used,memory.total", "--format=csv,noheader"],
            text=True,
        ).strip()
    except Exception as exc:
        return f"unavailable: {exc}"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", default="external/hf_models/Qwen2.5-0.5B-Instruct")
    parser.add_argument("--jittorllms_root", default="external/JittorLLMs")
    parser.add_argument("--max_input_length", type=int, default=256)
    parser.add_argument("--max_new_tokens", type=int, default=4)
    args = parser.parse_args()

    import jittor as jt

    jt.flags.use_cuda = 1
    print("model backend: qwen2_jittorllms")
    print("model path:", (ROOT / args.model_path).resolve())
    print("jittor:", jt.__version__)
    print("has_cuda:", jt.has_cuda)
    print("use_cuda:", jt.flags.use_cuda)
    print("GPU memory before:", gpu_memory())
    print("raw prompt:\n" + PROMPT)

    load_started = time.perf_counter()
    backend = Qwen2JittorBackend(
        model_path=ROOT / args.model_path,
        jittorllms_root=ROOT / args.jittorllms_root,
        max_input_length=args.max_input_length,
        max_new_tokens=args.max_new_tokens,
    )
    print(f"model load runtime: {time.perf_counter() - load_started:.3f}s")
    print("GPU memory loaded:", gpu_memory())
    result = backend.generate(PROMPT)
    print("raw generated output:", repr(result.text))
    print("generated token ids:", result.generated_token_ids)
    print(f"generation runtime: {result.runtime_sec:.3f}s")
    print("GPU memory after generation:", gpu_memory())
    print("JittorLLMs Qwen2 generation smoke ok")


if __name__ == "__main__":
    main()
