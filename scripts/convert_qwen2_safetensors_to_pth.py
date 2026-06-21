"""Convert the local Hugging Face Qwen2.5 checkpoint for JittorLLMs."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
from safetensors.torch import load_file


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_dir",
        default="external/hf_models/Qwen2.5-0.5B-Instruct",
    )
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    model_dir = Path(args.model_dir).resolve()
    source = model_dir / "model.safetensors"
    target = model_dir / "model.fp16.pth"
    if target.exists() and not args.force:
        print(f"Converted checkpoint already exists: {target}")
        return
    if not source.exists():
        raise FileNotFoundError(f"Missing Hugging Face checkpoint: {source}")

    state = load_file(str(source), device="cpu")
    converted = {
        key: value.to(torch.float16) if value.is_floating_point() else value
        for key, value in state.items()
    }
    if "lm_head.weight" not in converted and "model.embed_tokens.weight" in converted:
        converted["lm_head.weight"] = converted["model.embed_tokens.weight"]
    torch.save(converted, target)
    print(f"Converted {len(converted)} tensors to: {target}")
    print(f"Size: {target.stat().st_size} bytes")


if __name__ == "__main__":
    main()
