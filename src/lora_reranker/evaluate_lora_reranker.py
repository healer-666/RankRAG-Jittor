"""Evaluate a LoRA relevance reranker with answer-token log probabilities."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch
from peft import PeftModel
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

try:
    from src.lora_reranker.lora_utils import (
        build_prompt,
        compute_label_logprob,
        compute_ranking_metrics,
        load_jsonl,
        save_markdown_metrics,
    )
except ModuleNotFoundError:
    import sys

    sys.path.append(str(Path(__file__).resolve().parents[2]))
    from src.lora_reranker.lora_utils import (
        build_prompt,
        compute_label_logprob,
        compute_ranking_metrics,
        load_jsonl,
        save_markdown_metrics,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", default="Qwen/Qwen2.5-0.5B-Instruct")
    parser.add_argument("--adapter_dir", default="outputs/lora_adapters/qwen_0_5b_debug")
    parser.add_argument("--test_path", default="data/processed/lora_debug/test_queries.jsonl")
    parser.add_argument("--output_metrics", default="outputs/lora_debug/qwen_0_5b_lora_metrics.json")
    parser.add_argument("--output_rankings", default="outputs/lora_debug/qwen_0_5b_lora_rankings.json")
    parser.add_argument("--max_queries", type=int, default=50)
    parser.add_argument("--max_length", type=int, default=256)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.float16 if device.type == "cuda" else torch.float32

    tokenizer = AutoTokenizer.from_pretrained(args.model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    base_model = AutoModelForCausalLM.from_pretrained(args.model_name, torch_dtype=dtype, trust_remote_code=True)
    model = PeftModel.from_pretrained(base_model, args.adapter_dir)
    model.to(device)
    model.eval()

    queries = load_jsonl(args.test_path)[: args.max_queries]
    ranking_rows = []
    flat_rows = []
    start = time.time()
    for query in tqdm(queries, desc="LoRA reranking eval"):
        ranked_candidates = []
        for candidate in query["candidates"]:
            prompt = build_prompt(query["query"], candidate["passage"])
            rel_lp = compute_label_logprob(model, tokenizer, prompt, "Relevant", max_length=args.max_length, device=device)
            irr_lp = compute_label_logprob(model, tokenizer, prompt, "Irrelevant", max_length=args.max_length, device=device)
            score = rel_lp - irr_lp
            row = {
                "query_id": query["query_id"],
                "passage_id": candidate["passage_id"],
                "score": score,
                "label": int(candidate["label"]),
                "logprob_relevant": rel_lp,
                "logprob_irrelevant": irr_lp,
            }
            flat_rows.append(row)
            ranked_candidates.append(
                {
                    **row,
                    "passage": candidate["passage"],
                }
            )
        ranked_candidates.sort(key=lambda item: item["score"], reverse=True)
        ranking_rows.append({"query_id": query["query_id"], "query": query["query"], "candidates": ranked_candidates})
    elapsed = time.time() - start

    metrics = compute_ranking_metrics(flat_rows, [1, 3, 5, 10])
    metrics.update(
        {
            "device": str(device),
            "inference_time_sec": elapsed,
            "pairs_per_second": len(flat_rows) / elapsed if elapsed > 0 else 0.0,
        }
    )

    output_metrics = Path(args.output_metrics)
    output_rankings = Path(args.output_rankings)
    output_metrics.parent.mkdir(parents=True, exist_ok=True)
    output_rankings.parent.mkdir(parents=True, exist_ok=True)
    output_metrics.write_text(json.dumps(metrics, indent=2, ensure_ascii=False), encoding="utf-8")
    output_rankings.write_text(json.dumps(ranking_rows, indent=2, ensure_ascii=False), encoding="utf-8")
    save_markdown_metrics(metrics, output_metrics.with_suffix(".md"))
    print(json.dumps(metrics, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
