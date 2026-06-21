"""Aggregate downstream RAG answer generation results."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from statistics import mean
from typing import Any

import matplotlib.pyplot as plt

try:
    from src.rag_answer_generation import load_config, read_jsonl
    from src.utils import ensure_parent, read_json, resolve_path, write_json
except ModuleNotFoundError:
    import sys

    sys.path.append(str(Path(__file__).resolve().parents[1]))
    from src.rag_answer_generation import load_config, read_jsonl
    from src.utils import ensure_parent, read_json, resolve_path, write_json


METRIC_COLUMNS = [
    "contains_labeled_positive",
    "gold_answer_in_context",
    "answer_hit",
    "exact_match",
    "token_f1",
]


def load_method_rows(output_dir: Path, method: str, top_k: int) -> list[dict[str, Any]]:
    path = output_dir / f"{method}_top{top_k}_answers.jsonl"
    if not path.exists():
        raise FileNotFoundError(f"Missing generated answers for {method}: {path}")
    return read_jsonl(path)


def aggregate_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    questions = len(rows)
    if questions == 0:
        raise ValueError("Cannot aggregate empty result rows")
    generation_failures = [
        row
        for row in rows
        if row["gold_answer_in_context"] and row["failure_type"] == "generation_failure"
    ]
    insufficient = [
        row
        for row in rows
        if str(row.get("generated_answer", "")).strip().lower() == "insufficient information"
    ]
    empty = [row for row in rows if not str(row.get("generated_answer", "")).strip()]
    return {
        "questions": questions,
        "top_k": rows[0]["top_k"],
        "positive_in_context": mean(int(row["contains_labeled_positive"]) for row in rows),
        "gold_in_context": mean(int(row["gold_answer_in_context"]) for row in rows),
        "answer_hit": mean(int(row["answer_hit"]) for row in rows),
        "exact_match": mean(int(row["exact_match"]) for row in rows),
        "token_f1": mean(float(row["token_f1"]) for row in rows),
        "empty_answer_rate": len(empty) / questions,
        "insufficient_information_rate": len(insufficient) / questions,
        "average_answer_length": mean(len(str(row.get("generated_answer", "")).split()) for row in rows),
        "generation_failure_rate": len(generation_failures) / max(
            sum(int(row["gold_answer_in_context"]) for row in rows),
            1,
        ),
        "average_gold_answer_context_rank": mean(
            int(row["gold_answer_context_rank"])
            for row in rows
            if row.get("gold_answer_context_rank") is not None
        )
        if any(row.get("gold_answer_context_rank") is not None for row in rows)
        else None,
        "retrieval_failures": sum(1 for row in rows if row["failure_type"] == "retrieval_failure"),
        "generation_failures": len(generation_failures),
        "successes": sum(1 for row in rows if row["failure_type"] == "success"),
    }


def load_rerank_metric(config: dict[str, Any], method: str, metric: str) -> float | None:
    path = config.get("reranking_metrics", {}).get(method)
    if not path or not resolve_path(path).exists():
        return None
    payload = read_json(path)
    if method == "bm25" and "bm25" in payload:
        payload = payload["bm25"].get("metrics", {})
    return float(payload[metric]) if metric in payload else None


def write_markdown(results: list[dict[str, Any]], path: Path) -> None:
    lines = [
        "# Downstream RAG Evaluation Results",
        "",
        "| Method | Questions | Top-k | Positive in Context | Gold in Context | Answer Hit | EM | Token F1 | Generation Failure |",
        "| ------ | --------: | ----: | ------------------: | --------------: | ---------: | -: | -------: | -----------------: |",
    ]
    for row in results:
        m = row["metrics"]
        lines.append(
            f"| {row['method']} | {m['questions']} | {m['top_k']} | "
            f"{m['positive_in_context']:.4f} | {m['gold_in_context']:.4f} | "
            f"{m['answer_hit']:.4f} | {m['exact_match']:.4f} | {m['token_f1']:.4f} | "
            f"{m['generation_failure_rate']:.4f} |"
        )
    lines.extend(
        [
            "",
            "## Reranking to Answer",
            "",
            "| Method | R@1 | R@3 | R@5 | MRR | Gold in Context@3 | Answer Hit |",
            "| ------ | --: | --: | --: | --: | ----------------: | ---------: |",
        ]
    )
    for row in results:
        rr = row["reranking_metrics"]
        m = row["metrics"]
        lines.append(
            f"| {row['method']} | {format_optional(rr.get('recall@1'))} | {format_optional(rr.get('recall@3'))} | "
            f"{format_optional(rr.get('recall@5'))} | {format_optional(rr.get('mrr'))} | "
            f"{m['gold_in_context']:.4f} | {m['answer_hit']:.4f} |"
        )
    ensure_parent(path).write_text("\n".join(lines) + "\n", encoding="utf-8")


def format_optional(value: float | None) -> str:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return "N/A"
    return f"{value:.4f}"


def plot_results(results: list[dict[str, Any]], path: Path) -> None:
    methods = [row["method"] for row in results]
    metrics = ["gold_in_context", "answer_hit", "token_f1"]
    labels = ["Gold in context", "Answer hit", "Token F1"]
    x = range(len(methods))
    width = 0.24
    fig, ax = plt.subplots(figsize=(9, 4.8))
    for offset, (metric, label) in enumerate(zip(metrics, labels)):
        values = [row["metrics"][metric] for row in results]
        ax.bar([i + (offset - 1) * width for i in x], values, width=width, label=label)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Rate / score")
    ax.set_title("Downstream RAG answer quality on 50 MS MARCO questions")
    ax.set_xticks(list(x))
    ax.set_xticklabels(methods)
    ax.grid(axis="y", alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(ensure_parent(path), dpi=180)
    plt.close(fig)


def select_case_studies(all_rows: dict[str, list[dict[str, Any]]]) -> list[dict[str, Any]]:
    by_method_query = {
        method: {row["query_id"]: row for row in rows}
        for method, rows in all_rows.items()
    }
    methods = list(all_rows)
    qids = sorted(set.intersection(*(set(rows) for rows in by_method_query.values())))
    cases = []

    def add_case(kind: str, qid: str) -> None:
        if sum(1 for case in cases if case["case_type"] == kind) >= 5:
            return
        per_method = {method: by_method_query[method][qid] for method in methods}
        first = next(iter(per_method.values()))
        cases.append(
            {
                "case_type": kind,
                "query_id": qid,
                "question": first["question"],
                "gold_answers": first["gold_answers"],
                "methods": {
                    method: {
                        "context_candidate_ids": row["context_candidate_ids"],
                        "generated_answer": row["generated_answer"],
                        "answer_hit": row["answer_hit"],
                        "token_f1": row["token_f1"],
                        "failure_type": row["failure_type"],
                    }
                    for method, row in per_method.items()
                },
                "analysis": "Automatically selected for manual review.",
            }
        )

    for qid in qids:
        if "lora_v3" in methods and "bm25" in methods:
            if by_method_query["lora_v3"][qid]["answer_hit"] and not by_method_query["bm25"][qid]["answer_hit"]:
                add_case("lora_better_than_bm25", qid)
        if "cross_encoder" in methods and "lora_v3" in methods:
            if by_method_query["cross_encoder"][qid]["answer_hit"] and not by_method_query["lora_v3"][qid]["answer_hit"]:
                add_case("cross_encoder_better_than_lora", qid)
        if any(
            row["gold_answer_in_context"] and not row["answer_hit"]
            for row in (by_method_query[method][qid] for method in methods)
        ):
            add_case("generation_failure_with_answer_in_context", qid)
        if all(not by_method_query[method][qid]["answer_hit"] for method in methods):
            add_case("all_methods_failed", qid)
    return cases


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/downstream_rag_50q.yaml")
    parser.add_argument("--input-dir", default=None)
    parser.add_argument("--top-k", type=int, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    output_dir = resolve_path(args.input_dir or config["output_dir"])
    top_k = int(args.top_k or config["top_k"])
    methods = list(config["methods"])
    all_rows = {method: load_method_rows(output_dir, method, top_k) for method in methods}
    results = []
    for method, rows in all_rows.items():
        metrics = aggregate_rows(rows)
        reranking_metrics = {
            metric: load_rerank_metric(config, method, metric)
            for metric in ["recall@1", "recall@3", "recall@5", "mrr"]
        }
        results.append({"method": method, "metrics": metrics, "reranking_metrics": reranking_metrics})
    payload = {"methods": results}
    write_json(output_dir / "downstream_rag_eval_results.json", payload)
    write_markdown(results, output_dir / "downstream_rag_eval_results.md")
    plot_results(results, output_dir / "downstream_rag_eval_results.png")
    write_json(output_dir / "downstream_rag_case_study.json", select_case_studies(all_rows))
    print(json.dumps(payload, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
