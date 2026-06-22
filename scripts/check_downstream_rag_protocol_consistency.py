"""Check protocol consistency between original and strict-prompt RAG runs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

try:
    from src.rag_answer_generation import load_config, read_jsonl
    from src.utils import read_json, resolve_path, write_json
except ModuleNotFoundError:
    import sys

    sys.path.append(str(Path(__file__).resolve().parents[1]))
    from src.rag_answer_generation import load_config, read_jsonl
    from src.utils import read_json, resolve_path, write_json


SETTING_KEYS = ["top_k", "do_sample", "temperature", "max_new_tokens", "max_context_tokens"]


def rows_by_method_query(output_dir: Path, methods: list[str], top_k: int) -> dict[str, dict[str, dict[str, Any]]]:
    result = {}
    for method in methods:
        rows = read_jsonl(output_dir / f"{method}_top{top_k}_answers.jsonl")
        result[method] = {row["query_id"]: row for row in rows}
    return result


def load_reranking_metrics(config: dict[str, Any]) -> dict[str, Any]:
    metrics = {}
    for method, path in config.get("reranking_metrics", {}).items():
        metrics[method] = read_json(path)
    return metrics


def compare_rows(
    original_rows: dict[str, dict[str, dict[str, Any]]],
    strict_rows: dict[str, dict[str, dict[str, Any]]],
) -> tuple[dict[str, bool], list[str]]:
    errors = []
    flags = {
        "same_questions": True,
        "same_top3_contexts": True,
        "same_context_order": True,
        "same_gold_answers": True,
    }
    for method, original_by_qid in original_rows.items():
        strict_by_qid = strict_rows.get(method, {})
        if set(original_by_qid) != set(strict_by_qid):
            flags["same_questions"] = False
            errors.append(f"{method}: query id set mismatch")
            continue
        for qid, original in original_by_qid.items():
            strict = strict_by_qid[qid]
            if original.get("question") != strict.get("question"):
                flags["same_questions"] = False
                errors.append(f"{method}/{qid}: question text mismatch")
            if original.get("context_candidate_ids") != strict.get("context_candidate_ids"):
                flags["same_top3_contexts"] = False
                flags["same_context_order"] = False
                errors.append(f"{method}/{qid}: top-3 context ids mismatch")
            if original.get("contexts") != strict.get("contexts"):
                flags["same_context_order"] = False
                errors.append(f"{method}/{qid}: context text/order mismatch")
            if original.get("gold_answers") != strict.get("gold_answers"):
                flags["same_gold_answers"] = False
                errors.append(f"{method}/{qid}: gold answers mismatch")
            if original.get("gold_answer_in_context") != strict.get("gold_answer_in_context"):
                errors.append(f"{method}/{qid}: Gold@3 mismatch")
    return flags, errors


def generation_settings_match(original_summary: dict[str, Any], strict_summary: dict[str, Any]) -> tuple[bool, list[str]]:
    errors = []
    for key in SETTING_KEYS:
        if original_summary.get(key) != strict_summary.get(key):
            errors.append(f"generation setting differs: {key}")
    return not errors, errors


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--original-config", default="configs/downstream_rag_50q.yaml")
    parser.add_argument("--strict-config", default="configs/downstream_rag_50q_qwen2_1_5b_strict_prompt.yaml")
    parser.add_argument("--original-dir", default="outputs/downstream_rag_eval")
    parser.add_argument("--strict-dir", default="outputs/downstream_rag_eval_qwen2_1_5b_strict_prompt")
    parser.add_argument("--top-k", type=int, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    original_config = load_config(args.original_config)
    strict_config = load_config(args.strict_config)
    methods = list(strict_config["methods"])
    top_k = int(args.top_k or strict_config["top_k"])
    original_dir = resolve_path(args.original_dir)
    strict_dir = resolve_path(args.strict_dir)

    original_rows = rows_by_method_query(original_dir, methods, top_k)
    strict_rows = rows_by_method_query(strict_dir, methods, top_k)
    row_flags, errors = compare_rows(original_rows, strict_rows)

    original_summary = read_json(original_dir / "generation_run_summary.json")
    strict_summary = read_json(strict_dir / "generation_run_summary.json")
    same_settings, setting_errors = generation_settings_match(original_summary, strict_summary)
    errors.extend(setting_errors)

    same_rankings = original_config.get("rankings") == strict_config.get("rankings")
    same_reranking_metrics = load_reranking_metrics(original_config) == load_reranking_metrics(strict_config)
    if not same_rankings:
        errors.append("ranking config differs")
    if not same_reranking_metrics:
        errors.append("reranking metrics differ")

    payload = {
        "status": "passed" if not errors else "failed",
        "generator": strict_config.get("generator_model"),
        "same_questions": row_flags["same_questions"],
        "same_rankings": same_rankings,
        "same_top3_contexts": row_flags["same_top3_contexts"],
        "same_context_order": row_flags["same_context_order"],
        "same_gold_answers": row_flags["same_gold_answers"],
        "same_gold_at_3": not any("Gold@3 mismatch" in error for error in errors),
        "same_reranking_metrics": same_reranking_metrics,
        "same_generation_settings_except_prompt": same_settings,
        "differences": {
            "prompt_style": [
                original_summary.get("prompt_style", original_config.get("prompt_style", "original")),
                strict_summary.get("prompt_style", strict_config.get("prompt_style")),
            ],
            "prompt_version": [
                original_summary.get("prompt_version", original_config.get("prompt_version", "original_v1")),
                strict_summary.get("prompt_version", strict_config.get("prompt_version")),
            ],
        },
        "errors": errors,
    }
    write_json(strict_dir / "protocol_consistency_check.json", payload)
    print(json.dumps(payload, indent=2, ensure_ascii=False))
    if errors:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
