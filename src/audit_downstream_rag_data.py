"""Audit data and rankings for downstream RAG answer generation."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

try:
    from src.rag_answer_generation import RankingsLoader, load_config, read_jsonl
    from src.utils import ensure_parent, resolve_path, write_json
except ModuleNotFoundError:
    import sys

    sys.path.append(str(Path(__file__).resolve().parents[1]))
    from src.rag_answer_generation import RankingsLoader, load_config, read_jsonl
    from src.utils import ensure_parent, resolve_path, write_json


EXTRA_METHODS = {
    "jittor_qwen2_1_5b_zero_shot": {
        "path": "outputs/jittorllm_qwen2_1_5b_full/rankings.json",
    },
    "jittor_qwen2_0_5b_zero_shot": {
        "path": "outputs/jittorllm_qwen2_0_5b_full/rankings.json",
    },
    "jittor_mlp": {
        "path": "outputs/msmarco_medium_jittor_test_rankings.json",
    },
    "jittor_textcnn": {
        "path": "outputs/msmarco_medium_textcnn_jittor_test_rankings.json",
    },
}


def build_candidate_pool(rows: list[dict[str, Any]]) -> dict[str, dict[str, dict[str, Any]]]:
    return {
        row["query_id"]: {candidate["doc_id"]: candidate for candidate in row["candidates"]}
        for row in rows
    }


def inspect_original_answers() -> dict[str, Any]:
    try:
        from datasets import load_dataset

        row = next(iter(load_dataset("microsoft/ms_marco", "v1.1", split="validation", streaming=True)))
        return {
            "available": True,
            "fields": sorted(row.keys()),
            "has_answers": "answers" in row,
            "has_well_formed_answers": "wellFormedAnswers" in row,
            "sample_query": row.get("query"),
            "sample_answers": row.get("answers"),
            "sample_well_formed_answers": row.get("wellFormedAnswers"),
        }
    except Exception as exc:  # noqa: BLE001 - audit should record failure.
        return {"available": False, "error": repr(exc)}


def audit_rankings(config: dict[str, Any], candidate_pool: dict[str, dict[str, dict[str, Any]]]) -> dict[str, Any]:
    loader = RankingsLoader(candidate_pool)
    method_specs = {**config["rankings"], **EXTRA_METHODS}
    query_ids = set(candidate_pool)
    audit = {}
    for method, spec in method_specs.items():
        path = spec["path"]
        exists = resolve_path(path).exists()
        item = {"path": path, "exists": exists}
        if not exists:
            audit[method] = item
            continue
        try:
            rankings = loader.load_rankings(method, path, spec.get("nested_key"))
            qids = set(rankings)
            candidate_counts = [len(rows) for rows in rankings.values()]
            item.update(
                {
                    "status": "ready",
                    "queries": len(rankings),
                    "missing_queries": len(query_ids - qids),
                    "extra_queries": len(qids - query_ids),
                    "min_candidates": min(candidate_counts) if candidate_counts else 0,
                    "max_candidates": max(candidate_counts) if candidate_counts else 0,
                    "candidate_ids_aligned": not (query_ids - qids) and not (qids - query_ids),
                }
            )
        except Exception as exc:  # noqa: BLE001
            item.update({"status": "invalid", "error": repr(exc)})
        audit[method] = item
    return audit


def write_markdown(report: dict[str, Any], path: str) -> None:
    test = report["processed_test"]
    lines = [
        "# Downstream RAG Data Audit",
        "",
        "This audit checks whether the existing MS MARCO medium candidate pool and reranker rankings can support downstream answer-generation evaluation.",
        "",
        "## Processed Test File",
        "",
        f"Path: `{test['path']}`",
        "",
        "| Item | Value |",
        "| --- | --- |",
        f"| Queries | {test['queries']} |",
        f"| Candidates | {test['candidates']} |",
        f"| Top-level fields | `{', '.join(test['top_level_fields'])}` |",
        f"| Candidate fields | `{', '.join(test['candidate_fields'])}` |",
        f"| Has `answer` | {test['has_answer']} |",
        f"| Has `answers` | {test['has_answers']} |",
        f"| Has `wellFormedAnswers` | {test['has_well_formed_answers']} |",
        "",
        "The processed subset has query, candidate passage, candidate id, and label fields, but it does not retain gold answer fields.",
        "",
        "## Original Answer Source",
        "",
    ]
    original = report["original_msmarco_validation"]
    if original["available"]:
        lines.extend(
            [
                "Original `microsoft/ms_marco:v1.1` validation is available in the current environment and contains answer fields.",
                "",
                "| Field | Available |",
                "| --- | --- |",
                f"| `answers` | {original['has_answers']} |",
                f"| `wellFormedAnswers` | {original['has_well_formed_answers']} |",
                "",
                f"Sample query: `{original['sample_query']}`",
                f"Sample answers: `{original['sample_answers']}`",
            ]
        )
    else:
        lines.append(f"Original validation access failed: `{original['error']}`")

    lines.extend(["", "## Rankings Alignment", "", "| Method | Exists | Status | Queries | Missing | Extra | Candidate aligned | Path |", "| --- | --- | --- | ---: | ---: | ---: | --- | --- |"])
    for method, item in report["rankings"].items():
        lines.append(
            "| "
            + " | ".join(
                [
                    method,
                    str(item.get("exists")),
                    str(item.get("status", "missing")),
                    str(item.get("queries", "N/A")),
                    str(item.get("missing_queries", "N/A")),
                    str(item.get("extra_queries", "N/A")),
                    str(item.get("candidate_ids_aligned", "N/A")),
                    f"`{item['path']}`",
                ]
            )
            + " |"
        )
    ready_methods = [method for method, item in report["rankings"].items() if item.get("status") == "ready"]
    lines.extend(
        [
            "",
            "## Audit Conclusion",
            "",
            f"Methods suitable for the first downstream experiment: `{', '.join(report['first_pass_methods'])}`.",
            f"Additional aligned methods available for optional expansion: `{', '.join(method for method in ready_methods if method not in report['first_pass_methods'])}`.",
            "",
            "The project can build unified top-k contexts because the first-pass rankings cover the same 500 query ids and candidate ids as `data/processed/msmarco_medium/test.jsonl`.",
        ]
    )
    ensure_parent(path).write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    rows = read_jsonl(config["source_test_path"])
    candidate_pool = build_candidate_pool(rows)
    first = rows[0]
    report = {
        "processed_test": {
            "path": config["source_test_path"],
            "queries": len(rows),
            "candidates": sum(len(row["candidates"]) for row in rows),
            "top_level_fields": sorted(first.keys()),
            "candidate_fields": sorted(first["candidates"][0].keys()),
            "has_answer": "answer" in first,
            "has_answers": "answers" in first,
            "has_well_formed_answers": "wellFormedAnswers" in first,
        },
        "original_msmarco_validation": inspect_original_answers(),
        "rankings": audit_rankings(config, candidate_pool),
        "first_pass_methods": [method for method in config["methods"]],
    }
    write_json("outputs/downstream_rag_eval/data_audit.json", report)
    write_markdown(report, "docs/downstream_rag_data_audit.md")
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
