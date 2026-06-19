"""External pretrained Cross-Encoder reranker reference for MS MARCO medium.

This experiment is not the Jittor reproduction body. It is an external
pretrained semantic reranker reference used to analyze why RankRAG-style
LLM reranking matters.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

from dataset import read_jsonl
from metrics import evaluate_grouped
from utils import ensure_parent, resolve_path, write_json


METRIC_ORDER = [
    "recall@1",
    "recall@3",
    "recall@5",
    "recall@10",
    "ndcg@1",
    "ndcg@3",
    "ndcg@5",
    "ndcg@10",
    "mrr",
    "pairwise_accuracy",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", default="data/processed/msmarco_medium/test.jsonl")
    parser.add_argument("--model_name", default="cross-encoder/ms-marco-MiniLM-L6-v2")
    parser.add_argument("--output_metrics", default="outputs/msmarco_medium_cross_encoder_metrics.json")
    parser.add_argument("--output_rankings", default="outputs/msmarco_medium_cross_encoder_rankings.json")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--max_queries", type=int, default=500)
    parser.add_argument("--device", default=None, choices=[None, "cpu", "cuda"])
    return parser.parse_args()


def import_cross_encoder():
    try:
        import torch
        from sentence_transformers import CrossEncoder
    except Exception as exc:
        raise RuntimeError(
            "Failed to import torch/sentence-transformers. Install dependencies with "
            "`pip install sentence-transformers transformers torch` before running this reference."
        ) from exc
    return torch, CrossEncoder


def choose_device(torch_module, requested: str | None) -> str:
    if requested:
        return requested
    return "cuda" if torch_module.cuda.is_available() else "cpu"


def score_record(model, record: dict[str, Any], batch_size: int) -> tuple[list[dict[str, Any]], int]:
    pairs = [(record["query"], candidate["text"]) for candidate in record["candidates"]]
    scores = model.predict(pairs, batch_size=batch_size, show_progress_bar=False)
    ranking = []
    for candidate, score in zip(record["candidates"], scores):
        ranking.append(
            {
                "query_id": record["query_id"],
                "query": record["query"],
                "doc_id": candidate["doc_id"],
                "title": candidate.get("title", ""),
                "text": candidate["text"],
                "label": int(candidate["label"]),
                "score": float(score),
            }
        )
    ranking.sort(key=lambda row: row["score"], reverse=True)
    for rank, row in enumerate(ranking, start=1):
        row["rank"] = rank
    return ranking, len(pairs)


def write_markdown(path: str | Path, payload: dict[str, Any]) -> None:
    lines = [
        "# MS MARCO Medium Cross-Encoder Reference",
        "",
        "This is an external pretrained semantic reranker reference, not the Jittor reproduction body.",
        "",
        "| Metric | Value |",
        "| --- | ---: |",
    ]
    for metric in METRIC_ORDER:
        value = payload.get(metric)
        if value is not None:
            lines.append(f"| {metric} | {float(value):.4f} |")
    lines.extend(
        [
            "",
            "| Runtime | Value |",
            "| --- | ---: |",
            f"| total_time_sec | {payload['total_time_sec']:.4f} |",
            f"| num_queries | {payload['num_queries']} |",
            f"| num_pairs | {payload['num_pairs']} |",
            f"| pairs_per_second | {payload['pairs_per_second']:.4f} |",
            f"| avg_time_per_query | {payload['avg_time_per_query']:.4f} |",
            f"| device | {payload['device']} |",
            f"| model_name | {payload['model_name']} |",
            f"| sampled_evaluation | {payload['sampled_evaluation']} |",
        ]
    )
    ensure_parent(path).write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    try:
        torch, CrossEncoder = import_cross_encoder()
        device = choose_device(torch, args.device)
        model = CrossEncoder(args.model_name, device=device)
    except Exception as exc:
        print(f"Cross-Encoder setup failed: {exc}", file=sys.stderr)
        raise SystemExit(2) from None

    records = read_jsonl(args.data_path)
    if args.max_queries is not None:
        records = records[: args.max_queries]

    start = time.perf_counter()
    all_rows: list[dict[str, Any]] = []
    ranking_payload = []
    num_pairs = 0
    for record in records:
        ranking, pair_count = score_record(model, record, args.batch_size)
        num_pairs += pair_count
        all_rows.extend(ranking)
        ranking_payload.append(
            {
                "query_id": record["query_id"],
                "query": record["query"],
                "ranking": [
                    {
                        "rank": row["rank"],
                        "doc_id": row["doc_id"],
                        "title": row["title"],
                        "text": row["text"],
                        "label": row["label"],
                        "score": row["score"],
                    }
                    for row in ranking
                ],
            }
        )
    total_time = time.perf_counter() - start

    metrics = evaluate_grouped(all_rows, topk=[1, 3, 5, 10])
    payload = {
        **metrics,
        "model_name": args.model_name,
        "data_path": str(resolve_path(args.data_path)),
        "device": device,
        "total_time_sec": total_time,
        "num_queries": len(records),
        "num_pairs": num_pairs,
        "pairs_per_second": num_pairs / total_time if total_time > 0 else 0.0,
        "avg_time_per_query": total_time / len(records) if records else 0.0,
        "batch_size": args.batch_size,
        "sampled_evaluation": args.max_queries is not None and args.max_queries < 500,
        "role": "external pretrained semantic reranker reference",
    }

    write_json(args.output_metrics, payload)
    write_json(args.output_rankings, ranking_payload)
    markdown_path = Path(args.output_metrics).with_suffix(".md")
    write_markdown(markdown_path, payload)
    print(json.dumps(payload, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
