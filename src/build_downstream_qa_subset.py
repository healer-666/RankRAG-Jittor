"""Build a small answer-generation QA subset aligned to MS MARCO medium rankings."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any

try:
    from src.evaluate_answer_quality import gold_in_context, normalize_answer
    from src.rag_answer_generation import load_config
    from src.utils import ensure_parent, resolve_path, write_json
except ModuleNotFoundError:
    import sys

    sys.path.append(str(Path(__file__).resolve().parents[1]))
    from src.evaluate_answer_quality import gold_in_context, normalize_answer
    from src.rag_answer_generation import load_config
    from src.utils import ensure_parent, resolve_path, write_json


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    rows = []
    with resolve_path(path).open("r", encoding="utf-8") as file:
        for line in file:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def write_jsonl(path: str | Path, rows: list[dict[str, Any]]) -> None:
    resolved = ensure_parent(path)
    with resolved.open("w", encoding="utf-8") as file:
        for row in rows:
            file.write(json.dumps(row, ensure_ascii=False) + "\n")


def candidate_passage(candidate: dict[str, Any]) -> str:
    return str(candidate.get("text") or candidate.get("passage") or "")


def normalize_candidate(candidate: dict[str, Any]) -> dict[str, Any]:
    return {
        "candidate_id": str(candidate.get("doc_id") or candidate.get("candidate_id") or candidate.get("passage_id")),
        "passage": candidate_passage(candidate),
        "label": int(candidate.get("label", 0)),
    }


def answer_is_usable(answer: str) -> bool:
    normalized = normalize_answer(answer)
    if not normalized:
        return False
    tokens = normalized.split()
    if len(tokens) > 12:
        return False
    blocked = ["current", "today", "weather", "price", "stock", "latest"]
    return not any(word in normalized for word in blocked)


def stream_validation_answers(max_records: int) -> dict[str, dict[str, Any]]:
    from datasets import load_dataset

    stream = load_dataset("microsoft/ms_marco", "v1.1", split="validation", streaming=True)
    rows = {}
    for row in stream:
        answers = row.get("wellFormedAnswers") or row.get("answers") or []
        answers = [str(answer).strip() for answer in answers if str(answer).strip()]
        query = str(row.get("query", "")).strip()
        if query:
            rows[normalize_answer(query)] = {
                "query": query,
                "answers": answers,
                "query_type": row.get("query_type"),
            }
        if len(rows) >= max_records:
            break
    return rows


def build_subset(config: dict[str, Any]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    source_rows = read_jsonl(config["source_test_path"])
    original_by_query = stream_validation_answers(max(len(source_rows) * 20, 10000))
    num_questions = int(config.get("num_questions", 50))
    min_candidates = int(config.get("top_k", 3))

    kept = []
    excluded: Counter[str] = Counter()
    for index, source in enumerate(source_rows):
        original = original_by_query.get(normalize_answer(source["query"]))
        if original is None:
            excluded["missing_original_answer_row"] += 1
            continue
        answers = [answer for answer in original["answers"] if answer_is_usable(answer)]
        if not answers:
            excluded["no_usable_gold_answer"] += 1
            continue
        candidates = [normalize_candidate(candidate) for candidate in source["candidates"]]
        if not candidates:
            excluded["no_candidates"] += 1
            continue
        if len(candidates) < min_candidates:
            excluded["fewer_than_top_k_candidates"] += 1
            continue
        if not gold_in_context([candidate["passage"] for candidate in candidates], answers):
            excluded["gold_not_in_candidate_pool"] += 1
            continue
        kept.append(
            {
                "query_id": source["query_id"],
                "question": source["query"],
                "gold_answers": answers,
                "candidates": candidates,
                "source": "microsoft/ms_marco:v1.1 validation aligned by streaming order",
                "selection_reason": "non-empty short answer and gold answer appears in candidate pool",
            }
        )
        if len(kept) >= num_questions:
            break

    manifest = {
        "dataset_path": config["dataset_path"],
        "source_test_path": config["source_test_path"],
        "num_requested": num_questions,
        "num_kept": len(kept),
        "source_rows_scanned": len(original_by_query),
        "excluded": dict(excluded),
        "requirements": {
            "gold_answer_non_empty": True,
            "short_answer": True,
            "gold_answer_in_candidate_pool": True,
            "candidate_pool_reused_from_msmarco_medium_test": True,
        },
    }
    if len(kept) < num_questions:
        raise SystemExit(f"Only built {len(kept)} QA rows; excluded={dict(excluded)}")
    return kept, manifest


def write_report(rows: list[dict[str, Any]], manifest: dict[str, Any]) -> None:
    type_counts = Counter()
    for row in rows:
        question = row["question"].lower()
        if question.startswith("what"):
            type_counts["what"] += 1
        elif question.startswith("who"):
            type_counts["who"] += 1
        elif question.startswith("when"):
            type_counts["when"] += 1
        elif question.startswith("where"):
            type_counts["where"] += 1
        elif question.startswith("how many") or question.startswith("how much"):
            type_counts["how many/much"] += 1
        elif question.startswith(("is ", "are ", "do ", "does ", "did ", "can ", "was ", "were ")):
            type_counts["yes/no"] += 1
        else:
            type_counts["other"] += 1

    lines = [
        "# Downstream RAG QA Subset Report",
        "",
        f"Dataset path: `{manifest['dataset_path']}`",
        f"Source processed test path: `{manifest['source_test_path']}`",
        f"Gold answer source: `microsoft/ms_marco:v1.1` validation, aligned by streaming order.",
        "",
        "| Item | Value |",
        "| --- | ---: |",
        f"| Requested questions | {manifest['num_requested']} |",
        f"| Kept questions | {manifest['num_kept']} |",
        f"| Source rows scanned | {manifest['source_rows_scanned']} |",
    ]
    for key, value in manifest["excluded"].items():
        lines.append(f"| Excluded: {key} | {value} |")
    lines.extend(["", "Question type distribution:", "", "| Type | Count |", "| --- | ---: |"])
    for key, value in sorted(type_counts.items()):
        lines.append(f"| {key} | {value} |")
    lines.extend(
        [
            "",
            "Selection rules:",
            "- Gold answer must be non-empty.",
            "- Gold answer is kept short enough for automatic evaluation.",
            "- Real-time/current-information queries are filtered by simple keyword rules.",
            "- At least one candidate passage in the reused MS MARCO medium candidate pool must contain the gold answer string after normalization.",
            "- The original `data/processed/msmarco_medium/test.jsonl` file is not modified.",
        ]
    )
    ensure_parent("docs/downstream_rag_subset_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    rows, manifest = build_subset(config)
    write_jsonl(config["dataset_path"], rows)
    write_json(config["qa_manifest_path"], manifest)
    write_report(rows, manifest)
    print(json.dumps(manifest, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
