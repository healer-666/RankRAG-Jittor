"""Evaluate a LoRA relevance reranker with answer-token log probabilities."""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import time
from collections import Counter
from pathlib import Path
from typing import Any

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
    parser.add_argument(
        "--scoring_method",
        choices=["generate_parse", "relevant_logprob", "logprob_margin"],
        default="logprob_margin",
    )
    parser.add_argument("--max_new_tokens", type=int, default=4)
    parser.add_argument("--output_tokenization", default=None)
    parser.add_argument("--overwrite", action="store_true", help="Allow overwriting existing metrics/rankings files.")
    return parser.parse_args()


def ensure_output_file_safe(path: Path, *, overwrite: bool) -> None:
    if path.exists() and not overwrite:
        raise RuntimeError(f"Output file already exists: {path}. Use --overwrite intentionally.")


def resolve_model_name(model_name: str) -> tuple[str, bool]:
    local_path = os.environ.get("QWEN_LORA_MODEL_PATH") or os.environ.get("QWEN_GENERATOR_MODEL_PATH")
    return (local_path, True) if local_path else (model_name, False)


def resolve_git_commit() -> str | None:
    env_commit = os.environ.get("RANKRAG_GIT_COMMIT")
    if env_commit:
        return env_commit
    commit_file = Path("GIT_COMMIT.txt")
    if commit_file.exists():
        return commit_file.read_text(encoding="utf-8").strip()
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    except Exception:
        return None


def label_tokenization_info(tokenizer) -> dict[str, dict[str, Any]]:
    info = {}
    for label in ("Relevant", "Irrelevant"):
        token_ids = tokenizer.encode(label, add_special_tokens=False)
        info[label] = {
            "exact_label_string": label,
            "token_ids": token_ids,
            "token_count": len(token_ids),
            "decoded_tokens": tokenizer.convert_ids_to_tokens(token_ids),
            "contains_special_token": any(token_id in set(tokenizer.all_special_ids) for token_id in token_ids),
            "uses_full_label_token_sequence": True,
        }
    return info


def normalize_generated_label(text: str) -> str | None:
    normalized = re.sub(r"[^a-z]+", " ", text.lower()).strip()
    if not normalized:
        return None
    first = normalized.split()[0]
    if first == "relevant":
        return "Relevant"
    if first == "irrelevant":
        return "Irrelevant"
    return None


@torch.no_grad()
def generate_label(model, tokenizer, prompt: str, *, max_length: int, max_new_tokens: int, device: torch.device) -> tuple[str, str | None]:
    prompt_ids = tokenizer.encode(prompt, add_special_tokens=False)
    if len(prompt_ids) > max_length:
        prompt_ids = prompt_ids[:max_length]
    tensor = torch.tensor([prompt_ids], dtype=torch.long, device=device)
    generated = model.generate(
        input_ids=tensor,
        attention_mask=torch.ones_like(tensor),
        do_sample=False,
        max_new_tokens=max_new_tokens,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id,
    )
    new_tokens = generated[0, tensor.shape[1] :]
    text = tokenizer.decode(new_tokens, skip_special_tokens=True)
    return text, normalize_generated_label(text)


def tie_summary(ranking_rows: list[dict]) -> dict[str, float]:
    tied_queries = 0
    total_tie_rate = 0.0
    max_tie_group = 0
    for query in ranking_rows:
        scores = [candidate["score"] for candidate in query["candidates"]]
        counts = Counter(scores)
        tied_items = sum(count for count in counts.values() if count > 1)
        if tied_items:
            tied_queries += 1
        total_tie_rate += tied_items / max(len(scores), 1)
        max_tie_group = max(max_tie_group, max(counts.values(), default=0))
    query_count = len(ranking_rows)
    return {
        "queries_with_ties": tied_queries,
        "query_tie_rate": tied_queries / max(query_count, 1),
        "average_candidate_tie_rate": total_tie_rate / max(query_count, 1),
        "max_tie_group_size": max_tie_group,
    }


def main() -> None:
    args = parse_args()
    output_metrics = Path(args.output_metrics)
    output_rankings = Path(args.output_rankings)
    ensure_output_file_safe(output_metrics, overwrite=args.overwrite)
    ensure_output_file_safe(output_rankings, overwrite=args.overwrite)
    ensure_output_file_safe(output_metrics.with_suffix(".md"), overwrite=args.overwrite)
    if args.output_tokenization:
        ensure_output_file_safe(Path(args.output_tokenization), overwrite=args.overwrite)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.float16 if device.type == "cuda" else torch.float32

    model_load_name, local_model_path_used = resolve_model_name(args.model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_load_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    base_model = AutoModelForCausalLM.from_pretrained(model_load_name, torch_dtype=dtype, trust_remote_code=True)
    model = PeftModel.from_pretrained(base_model, args.adapter_dir)
    model.to(device)
    model.eval()
    tokenization_info = label_tokenization_info(tokenizer)

    queries = load_jsonl(args.test_path)[: args.max_queries]
    ranking_rows = []
    flat_rows = []
    parse_failure_count = 0
    generated_label_counts: Counter[str] = Counter()
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    start = time.time()
    for query in tqdm(queries, desc="LoRA reranking eval"):
        ranked_candidates = []
        for candidate in query["candidates"]:
            try:
                prompt = build_prompt(query["query"], candidate["passage"])
                rel_lp = None
                irr_lp = None
                generated_text = None
                parsed_label = None
                if args.scoring_method == "generate_parse":
                    generated_text, parsed_label = generate_label(
                        model,
                        tokenizer,
                        prompt,
                        max_length=args.max_length,
                        max_new_tokens=args.max_new_tokens,
                        device=device,
                    )
                    if parsed_label is None:
                        parse_failure_count += 1
                        score = -1.0
                        generated_label_counts["parse_failure"] += 1
                    else:
                        score = 1.0 if parsed_label == "Relevant" else 0.0
                        generated_label_counts[parsed_label] += 1
                else:
                    rel_lp = compute_label_logprob(model, tokenizer, prompt, "Relevant", max_length=args.max_length, device=device)
                    if args.scoring_method == "relevant_logprob":
                        score = rel_lp
                    else:
                        irr_lp = compute_label_logprob(model, tokenizer, prompt, "Irrelevant", max_length=args.max_length, device=device)
                        score = rel_lp - irr_lp
            except Exception:
                parse_failure_count += 1
                raise
            row = {
                "query_id": query["query_id"],
                "passage_id": candidate["passage_id"],
                "score": score,
                "label": int(candidate["label"]),
                "logprob_relevant": rel_lp,
                "logprob_irrelevant": irr_lp,
                "generated_text": generated_text,
                "parsed_label": parsed_label,
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
            "device_name": torch.cuda.get_device_name(0) if device.type == "cuda" else None,
            "torch_dtype": str(dtype),
            "model_name": args.model_name,
            "model_load_name": model_load_name,
            "local_model_path_used": local_model_path_used,
            "git_commit": resolve_git_commit(),
            "adapter_dir": args.adapter_dir,
            "test_path": args.test_path,
            "max_queries": args.max_queries,
            "max_length": args.max_length,
            "scoring_method": args.scoring_method,
            "max_new_tokens": args.max_new_tokens if args.scoring_method == "generate_parse" else None,
            "label_tokenization": tokenization_info,
            "inference_time_sec": elapsed,
            "pairs_per_second": len(flat_rows) / elapsed if elapsed > 0 else 0.0,
            "peak_gpu_memory_gib": torch.cuda.max_memory_allocated(device) / 1024**3 if device.type == "cuda" else None,
            "parse_failure_count": parse_failure_count,
            "parse_failure_rate": parse_failure_count / max(len(flat_rows), 1),
            "generated_label_distribution": dict(sorted(generated_label_counts.items())),
            **tie_summary(ranking_rows),
        }
    )

    output_metrics.parent.mkdir(parents=True, exist_ok=True)
    output_rankings.parent.mkdir(parents=True, exist_ok=True)
    output_metrics.write_text(json.dumps(metrics, indent=2, ensure_ascii=False), encoding="utf-8")
    output_rankings.write_text(json.dumps(ranking_rows, indent=2, ensure_ascii=False), encoding="utf-8")
    if args.output_tokenization:
        Path(args.output_tokenization).parent.mkdir(parents=True, exist_ok=True)
        Path(args.output_tokenization).write_text(json.dumps(tokenization_info, indent=2, ensure_ascii=False), encoding="utf-8")
    save_markdown_metrics(metrics, output_metrics.with_suffix(".md"))
    print(json.dumps(metrics, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
