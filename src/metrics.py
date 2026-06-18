"""Ranking metrics shared by the PyTorch and Jittor evaluation paths."""

from __future__ import annotations

from collections import defaultdict
from math import log2
from typing import Iterable


def _ranked_labels(scores: Iterable[float], labels: Iterable[int]) -> list[int]:
    pairs = sorted(zip(scores, labels), key=lambda item: item[0], reverse=True)
    return [int(label) for _, label in pairs]


def recall_at_k(scores: Iterable[float], labels: Iterable[int], k: int) -> float:
    ranked = _ranked_labels(scores, labels)
    positives = sum(ranked)
    if positives == 0:
        return 0.0
    return sum(ranked[:k]) / positives


def mrr(scores: Iterable[float], labels: Iterable[int]) -> float:
    ranked = _ranked_labels(scores, labels)
    for index, label in enumerate(ranked, start=1):
        if label == 1:
            return 1.0 / index
    return 0.0


def ndcg_at_k(scores: Iterable[float], labels: Iterable[int], k: int) -> float:
    ranked = _ranked_labels(scores, labels)
    dcg = sum((2**label - 1) / log2(rank + 1) for rank, label in enumerate(ranked[:k], start=1))
    ideal = sorted([int(label) for label in labels], reverse=True)
    idcg = sum((2**label - 1) / log2(rank + 1) for rank, label in enumerate(ideal[:k], start=1))
    return 0.0 if idcg == 0 else dcg / idcg


def pairwise_accuracy(scores: Iterable[float], labels: Iterable[int]) -> float:
    positives = [score for score, label in zip(scores, labels) if int(label) == 1]
    negatives = [score for score, label in zip(scores, labels) if int(label) == 0]
    total = 0
    correct = 0
    for pos_score in positives:
        for neg_score in negatives:
            total += 1
            correct += int(pos_score > neg_score)
    return 0.0 if total == 0 else correct / total


def evaluate_grouped(rows: list[dict], topk: list[int]) -> dict[str, float]:
    """Compute ranking metrics from rows with query_id, score, and label fields."""

    grouped: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        grouped[row["query_id"]].append(row)

    results: dict[str, float] = {}
    for k in topk:
        results[f"recall@{k}"] = 0.0
        results[f"ndcg@{k}"] = 0.0
    results["mrr"] = 0.0
    results["pairwise_accuracy"] = 0.0

    for candidates in grouped.values():
        scores = [float(row["score"]) for row in candidates]
        labels = [int(row["label"]) for row in candidates]
        for k in topk:
            results[f"recall@{k}"] += recall_at_k(scores, labels, k)
            results[f"ndcg@{k}"] += ndcg_at_k(scores, labels, k)
        results["mrr"] += mrr(scores, labels)
        results["pairwise_accuracy"] += pairwise_accuracy(scores, labels)

    query_count = max(len(grouped), 1)
    return {name: value / query_count for name, value in results.items()}
