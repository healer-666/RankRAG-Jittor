"""Validate downstream RAG answer generation outputs."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from statistics import mean

try:
    from src.evaluate_answer_quality import answer_metrics, classify_failure
    from src.rag_answer_generation import load_config, read_jsonl
    from src.utils import resolve_path, write_json
except ModuleNotFoundError:
    import sys

    sys.path.append(str(Path(__file__).resolve().parents[1]))
    from src.evaluate_answer_quality import answer_metrics, classify_failure
    from src.rag_answer_generation import load_config, read_jsonl
    from src.utils import resolve_path, write_json


TOL = 1e-9


def validate_method(method: str, rows: list[dict], qa_pool: dict, summary: dict) -> list[str]:
    errors = []
    seen = set()
    expected_qids = set(qa_pool)
    actual_qids = {row["query_id"] for row in rows}
    if actual_qids != expected_qids:
        errors.append(f"{method}: query id set mismatch")
    for row in rows:
        key = (row["method"], row["query_id"], row["top_k"])
        if key in seen:
            errors.append(f"{method}: duplicate result key {key}")
        seen.add(key)
        if row["method"] != method:
            errors.append(f"{method}: row method mismatch {row['method']}")
        if len(row["context_candidate_ids"]) != int(row["top_k"]):
            errors.append(f"{method}/{row['query_id']}: top-k count mismatch")
        valid_candidates = {candidate["candidate_id"] for candidate in qa_pool[row["query_id"]]["candidates"]}
        if not set(row["context_candidate_ids"]).issubset(valid_candidates):
            errors.append(f"{method}/{row['query_id']}: context candidate id not in QA pool")
        if not str(row.get("generated_answer", "")).strip():
            errors.append(f"{method}/{row['query_id']}: generated answer is empty")
        metrics = answer_metrics(row["generated_answer"], row["gold_answers"])
        for key_metric in ["exact_match", "gold_containment", "answer_hit"]:
            if int(row[key_metric]) != int(metrics[key_metric]):
                errors.append(f"{method}/{row['query_id']}: {key_metric} mismatch")
        if abs(float(row["token_f1"]) - float(metrics["token_f1"])) > TOL:
            errors.append(f"{method}/{row['query_id']}: token_f1 mismatch")
        expected_failure = classify_failure(
            gold_answer_in_context=bool(row["gold_answer_in_context"]),
            answer_hit=int(row["answer_hit"]),
        )
        if row["failure_type"] != expected_failure:
            errors.append(f"{method}/{row['query_id']}: failure_type mismatch")
        for metric_name in ["token_f1"]:
            if not math.isfinite(float(row[metric_name])):
                errors.append(f"{method}/{row['query_id']}: non-finite {metric_name}")

    recomputed = {
        "questions": len(rows),
        "positive_in_context": mean(int(row["contains_labeled_positive"]) for row in rows),
        "gold_in_context": mean(int(row["gold_answer_in_context"]) for row in rows),
        "answer_hit": mean(int(row["answer_hit"]) for row in rows),
        "exact_match": mean(int(row["exact_match"]) for row in rows),
        "token_f1": mean(float(row["token_f1"]) for row in rows),
    }
    summary_metrics = summary["metrics"]
    for key, value in recomputed.items():
        if abs(float(summary_metrics[key]) - float(value)) > TOL:
            errors.append(f"{method}: aggregate {key} mismatch")
    return errors


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/downstream_rag_50q.yaml")
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--top-k", type=int, default=None)
    parser.add_argument("--max-questions", type=int, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    output_dir = resolve_path(args.output_dir or config["output_dir"])
    top_k = int(args.top_k or config["top_k"])
    qa_rows = read_jsonl(config["dataset_path"])
    if args.max_questions is not None:
        qa_rows = qa_rows[: args.max_questions]
    qa_pool = {row["query_id"]: row for row in qa_rows}
    summary_path = output_dir / "downstream_rag_eval_results.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    summary_by_method = {row["method"]: row for row in summary["methods"]}

    all_errors = []
    method_counts = {}
    for method in config["methods"]:
        rows = read_jsonl(output_dir / f"{method}_top{top_k}_answers.jsonl")
        method_counts[method] = len(rows)
        all_errors.extend(validate_method(method, rows, qa_pool, summary_by_method[method]))

    qid_sets = []
    for method in config["methods"]:
        rows = read_jsonl(output_dir / f"{method}_top{top_k}_answers.jsonl")
        qid_sets.append({row["query_id"] for row in rows})
    if len({tuple(sorted(qids)) for qids in qid_sets}) != 1:
        all_errors.append("method query id sets are not identical")

    validation = {
        "status": "ready" if not all_errors else "failed",
        "methods": list(config["methods"]),
        "method_counts": method_counts,
        "errors": all_errors,
    }
    write_json(output_dir / "validation.json", validation)
    print(json.dumps(validation, indent=2, ensure_ascii=False))
    if all_errors:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
