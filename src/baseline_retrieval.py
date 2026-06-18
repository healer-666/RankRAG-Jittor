"""TF-IDF and BM25 retrieval baselines for grouped ranking JSONL data."""

from __future__ import annotations

import argparse
import json
import re
from collections import defaultdict
from pathlib import Path

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from dataset import candidate_feature_text, read_jsonl
from metrics import evaluate_grouped
from utils import ensure_parent, write_json

try:
    from rank_bm25 import BM25Okapi
except Exception as exc:  # pragma: no cover - depends on optional dependency.
    BM25Okapi = None
    BM25_IMPORT_ERROR = exc
else:
    BM25_IMPORT_ERROR = None


TOKEN_PATTERN = re.compile(r"[a-z0-9]+")
METRIC_ORDER = [
    "recall@1",
    "ndcg@1",
    "recall@3",
    "ndcg@3",
    "recall@5",
    "ndcg@5",
    "recall@10",
    "ndcg@10",
    "mrr",
    "pairwise_accuracy",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", default="data/processed/msmarco/test.jsonl")
    parser.add_argument("--output_path", default="outputs/msmarco_retrieval_baseline_metrics.json")
    parser.add_argument("--rankings_path", default="outputs/msmarco_retrieval_baseline_rankings.json")
    parser.add_argument("--run_name", default="msmarco")
    return parser.parse_args()


def tokenize(text: str) -> list[str]:
    return TOKEN_PATTERN.findall(text.lower())


def score_tfidf(query: str, candidates: list[dict]) -> list[float]:
    texts = [query] + [candidate_feature_text(candidate) for candidate in candidates]
    vectorizer = TfidfVectorizer(lowercase=True, ngram_range=(1, 2))
    matrix = vectorizer.fit_transform(texts)
    scores = cosine_similarity(matrix[0:1], matrix[1:]).reshape(-1)
    return [float(score) for score in scores]


def score_bm25(query: str, candidates: list[dict]) -> list[float]:
    if BM25Okapi is None:
        raise RuntimeError(f"rank_bm25 is unavailable: {BM25_IMPORT_ERROR}")
    corpus = [tokenize(candidate_feature_text(candidate)) for candidate in candidates]
    bm25 = BM25Okapi(corpus)
    return [float(score) for score in bm25.get_scores(tokenize(query))]


def build_rows(records: list[dict], method: str) -> tuple[list[dict], list[dict]]:
    rows: list[dict] = []
    rankings: list[dict] = []
    scorer = score_tfidf if method == "tfidf" else score_bm25

    for record in records:
        candidates = record["candidates"]
        scores = scorer(record["query"], candidates)
        scored_candidates = []
        for candidate, score in zip(candidates, scores):
            row = {
                "query_id": record["query_id"],
                "query": record["query"],
                "doc_id": candidate["doc_id"],
                "title": candidate.get("title", ""),
                "text": candidate["text"],
                "label": int(candidate["label"]),
                "score": float(score),
            }
            rows.append(row)
            scored_candidates.append(row)

        ranked = sorted(scored_candidates, key=lambda item: item["score"], reverse=True)
        rankings.append(
            {
                "query_id": record["query_id"],
                "query": record["query"],
                "ranking": [
                    {
                        "rank": index,
                        "doc_id": item["doc_id"],
                        "title": item.get("title", ""),
                        "text": item["text"],
                        "label": item["label"],
                        "score": item["score"],
                    }
                    for index, item in enumerate(ranked, start=1)
                ],
            }
        )

    return rows, rankings


def markdown_table(metrics_by_method: dict[str, dict]) -> str:
    headers = ["Method", "Status", *METRIC_ORDER]
    lines = ["| " + " | ".join(headers) + " |", "| --- | --- | " + " | ".join(["---:"] * len(METRIC_ORDER)) + " |"]
    for method, payload in metrics_by_method.items():
        status = payload.get("status", "ready")
        metrics = payload.get("metrics", {})
        values = [f"{float(metrics[name]):.4f}" if name in metrics else "N/A" for name in METRIC_ORDER]
        lines.append("| " + " | ".join([method, status, *values]) + " |")
    return "\n".join(lines) + "\n"


def main() -> None:
    args = parse_args()
    records = read_jsonl(args.data_path)
    max_candidates = max((len(record.get("candidates", [])) for record in records), default=0)
    topk = [1, 3, 5, 10] if max_candidates >= 10 or args.run_name.endswith("medium") else [1, 3, 5]

    metrics_by_method: dict[str, dict] = {}
    rankings_by_method: dict[str, list[dict]] = {}

    for method in ["tfidf", "bm25"]:
        try:
            rows, rankings = build_rows(records, method)
            metrics_by_method[method] = {
                "run_name": args.run_name,
                "status": "ready",
                "metrics": evaluate_grouped(rows, topk=topk),
            }
            rankings_by_method[method] = rankings
        except Exception as exc:
            metrics_by_method[method] = {
                "run_name": args.run_name,
                "status": "unavailable",
                "error": str(exc),
                "metrics": {},
            }
            rankings_by_method[method] = []
            print(f"{method} unavailable: {exc}")

    write_json(args.output_path, metrics_by_method)
    write_json(args.rankings_path, rankings_by_method)
    md_path = Path(args.output_path).with_suffix(".md")
    ensure_parent(md_path).write_text(markdown_table(metrics_by_method), encoding="utf-8")

    print(f"Saved retrieval baseline metrics to {args.output_path}")
    print(markdown_table(metrics_by_method))


if __name__ == "__main__":
    main()
